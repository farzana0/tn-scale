#!/usr/bin/env python3
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn

from torchmps import MPS
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


########################################
# Miscellaneous initialization
########################################

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

start_time = time.time()

# MPS parameters
bond_dim = 20
adaptive_mode = False
periodic_bc = False

# Training parameters
alpha = 4.0               # multiplier in alpha * n * log2(n)
num_test_masks = 1000
batch_size = 64
num_epochs = 20
learn_rate = 1e-3
l2_reg = 0.0

########################################
# Load DistilBERT IMDB model
########################################

MODEL_NAME = "textattack/distilbert-base-uncased-imdb"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

print(f"Using model: {MODEL_NAME}")
print(f"Device: {DEVICE}")

########################################
# Pick one reasonably long IMDB review
########################################

ds = load_dataset("imdb")

def tokenize_length(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        add_special_tokens=True
    )
    example["input_ids"] = tokens["input_ids"]
    example["attention_mask"] = tokens["attention_mask"]
    example["len"] = len(tokens["input_ids"])
    return example

test = ds["test"].map(tokenize_length)

candidates = [ex for ex in test if 256 <= ex["len"] <= 512]
if len(candidates) == 0:
    candidates = [ex for ex in test if ex["len"] > 64]

review = candidates[0]
input_ids_base = torch.tensor(review["input_ids"], dtype=torch.long, device=DEVICE)
attention_mask_base = torch.tensor(review["attention_mask"], dtype=torch.long, device=DEVICE)
n_sites = input_ids_base.shape[0]

print(f"Selected review with length n = {n_sites} tokens")

MASK_TOKEN_ID = tokenizer.mask_token_id

@torch.no_grad()
def f_value_function(keep_mask_np: np.ndarray) -> float:
    """
    keep_mask_np: shape (n_sites,), bool or {0,1}
                  True/1 = keep token, False/0 = mask
    Returns: scalar float (logit of positive class)
    """
    keep_mask = torch.tensor(keep_mask_np.astype(bool), device=DEVICE)
    masked_ids = input_ids_base.clone()
    masked_ids[~keep_mask] = MASK_TOKEN_ID

    ids_t = masked_ids.unsqueeze(0)               # (1, n)
    attn_t = attention_mask_base.unsqueeze(0)     # (1, n)

    outputs = model(input_ids=ids_t, attention_mask=attn_t)
    logits = outputs.logits                       # (1, 2)
    positive_logit = logits[0, 1].item()
    return positive_logit


########################################
# Dataset construction (in mask space)
########################################

def sample_random_mask(n: int) -> np.ndarray:
    """
    Each token independently kept/dropped with p = 0.5
    Returns a boolean np.array of shape (n,)
    """
    return (np.random.rand(n) < 0.5)


def build_dataset(alpha: float, num_test: int):
    """
    Build:
      - training dataset of size L = alpha * n * log2(n)
      - test dataset of size num_test (for R^2)
    Each sample: (z, label) where
      - z in {-1, +1}^n encodes the subset S
      - label is DistilBERT logit f(S)
    The feature map will be applied later inside the training loop.
    """
    L = int(alpha * n_sites * math.log2(n_sites))
    print(f"Building dataset: L_train = {L}, L_test = {num_test}")

    # Train
    Z_train = []
    y_train = []

    for _ in range(L):
        keep_mask = sample_random_mask(n_sites)              # bool
        z = np.where(keep_mask, 1.0, -1.0).astype(np.float32)  # (n,)
        y = f_value_function(keep_mask)

        Z_train.append(z)
        y_train.append(y)

    Z_train = np.stack(Z_train, axis=0)  # (L, n)
    y_train = np.array(y_train, dtype=np.float32)  # (L,)

    # Test
    Z_test = []
    y_test = []

    for _ in range(num_test):
        keep_mask = sample_random_mask(n_sites)
        z = np.where(keep_mask, 1.0, -1.0).astype(np.float32)
        y = f_value_function(keep_mask)

        Z_test.append(z)
        y_test.append(y)

    Z_test = np.stack(Z_test, axis=0)
    y_test = np.array(y_test, dtype=np.float32)

    return Z_train, y_train, Z_test, y_test


########################################
# Build dataset
########################################

Z_train, y_train, Z_test, y_test = build_dataset(alpha=alpha, num_test=num_test_masks)

# Convert to tensors
X_train = torch.tensor(Z_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(Z_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

train_ds = torch.utils.data.TensorDataset(X_train, y_train_t)
test_ds = torch.utils.data.TensorDataset(X_test, y_test_t)

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=batch_size, shuffle=True, drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=batch_size, shuffle=False, drop_last=False
)

num_train_batches = len(train_loader)
num_test_batches = len(test_loader)

########################################
# Learnable MLP feature map
########################################

class MLPFeatureMap(nn.Module):
    def __init__(self, n_sites: int, hidden_dim: int = 128):
        """
        Simple MLP: R^n -> R^n (same dimension, but richer nonlinear features).
        """
        super().__init__()
        self.n_sites = n_sites
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(n_sites, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_sites)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (batch, n_sites)
        returns: (batch, n_sites) transformed features
        """
        return self.net(z)


feature_map = MLPFeatureMap(n_sites=n_sites, hidden_dim=128).to(DEVICE)

########################################
# Initialize the MPS module
########################################

mps = MPS(
    input_dim=n_sites,          # feature dimension after MLP = n_sites
    output_dim=1,               # scalar regression
    bond_dim=bond_dim,
    adaptive_mode=adaptive_mode,
    periodic_bc=periodic_bc,
)

mps = mps.to(DEVICE)

# Loss and optimizer (joint over MPS + feature map)
loss_fun = nn.MSELoss()
optimizer = torch.optim.Adam(
    list(mps.parameters()) + list(feature_map.parameters()),
    lr=learn_rate,
    weight_decay=l2_reg
)


########################################
# Training loop
########################################

print(
    f"Training MPS on DistilBERT mask surrogate (MLP feature map) \n"
    f"n_sites = {n_sites}, alpha = {alpha}, L_train = {len(train_ds)}, "
    f"L_test = {len(test_ds)}, epochs = {num_epochs}"
)
print(f"Maximum MPS bond dimension = {bond_dim}")
print(f" * {'Adaptive' if adaptive_mode else 'Fixed'} bond dimensions")
print(f" * {'Periodic' if periodic_bc else 'Open'} boundary conditions")
print(f"Using Adam w/ learning rate = {learn_rate:.1e}")
if l2_reg > 0:
    print(f" * L2 regularization = {l2_reg:.2e}")
print()

for epoch_num in range(1, num_epochs + 1):
    mps.train()
    feature_map.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(DEVICE)   # (B, n_sites), z in {-1, +1}
        labels = labels.to(DEVICE)   # (B,)

        feats = feature_map(inputs)            # (B, n_sites)
        scores = mps(feats).squeeze(-1)        # (B,)
        loss = loss_fun(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    avg_loss = running_loss / len(train_ds)

    # Evaluate R^2 on test set
    mps.eval()
    feature_map.eval()
    with torch.no_grad():
        y_true_list = []
        y_pred_list = []

        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            feats = feature_map(inputs)
            preds = mps(feats).squeeze(-1)

            y_true_list.append(labels.cpu().numpy())
            y_pred_list.append(preds.cpu().numpy())

        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)

        y_mean = y_true.mean()
        num = np.sum((y_pred - y_true) ** 2)
        den = np.sum((y_true - y_mean) ** 2)
        R2 = 1.0 - num / den if den > 0 else float("nan")

    print(f"### Epoch {epoch_num} ###")
    print(f"Average train loss: {avg_loss:.4f}")
    print(f"Test R^2:           {R2:.4f}")
    print(f"Runtime so far:     {int(time.time() - start_time)} sec\n")
