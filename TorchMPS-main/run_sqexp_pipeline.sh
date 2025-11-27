#!/usr/bin/env bash
set -e

# Config for exponential experiment
PREFIX="sqexp_D50"
MAX_DEGREE=5
N_TARGETS=5

echo ">>> Training MPS for sqexp with feature map exp(x_i^2)..."
python train_mps_sqexp_paths.py \
  --prefix "${PREFIX}" \
  --max-degree "${MAX_DEGREE}" \
  --n-targets "${N_TARGETS}" \
  --batch-size 4096 \
  --num-epochs 300 \
  --bond-dim 40 \
  --lr 1e-3 \
  --l2-reg 0.0 \
  --lr-factor 0.8 \
  --lr-patience 5 \
  --min-lr 1e-7 \
  --early-stop-patience 30 \
  --grad-clip 5.0

echo ">>> Evaluating TN-SHAP on teacher vs MPS (sqexp)..."
python tnshap_sqexp_vandermonde_eval.py \
  --prefix "${PREFIX}" \
  --max-degree "${MAX_DEGREE}" \
  --n-targets "${N_TARGETS}" \
  --eval-batch-size 4096
