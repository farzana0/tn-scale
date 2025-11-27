#!/usr/bin/env bash
set -e

mkdir -p logs

TASKS=("poly5" "poly10" "sqexp")
DS=(50 100)   # Only consider D = 50, 100

# Total number of datapoints
N_TOTAL=1
N_TEST=0         # 5% of all datapoints as test set
N_TRAIN=1     # remaining as train
N_TARGETS=1    # calculate Shap on 100 datapoints

# Seed for reproducibility (change this to get different random splits)
SEED_BASE=42

for task in "${TASKS[@]}"; do
  for D in "${DS[@]}"; do

    PREFIX="${task}_D${D}"

    echo "==============================================="
    echo "Task=${task}, D=${D}, PREFIX=${PREFIX}"
    echo "==============================================="

    # 1) Generate teacher & data
    echo "[1/3] Generating teacher and data..."
    python poly_teacher.py \
      --prefix "${PREFIX}" \
      --task "${task}" \
      --D "${D}" \
      --n-train "${N_TRAIN}" \
      --n-test "${N_TEST}" \
      --noise-std 0.0 \
      --seed-base "${SEED_BASE}" \
      > "logs/${PREFIX}_gen.log" 

    # Choose max_degree for TN-SHAP interpolation
    if [ "${task}" = "poly5" ]; then
      MAX_DEG=5
    elif [ "${task}" = "poly10" ]; then
      MAX_DEG=5
    else
      MAX_DEG=5  # sqexp approx
    fi

    # 2) Train MPS
    echo "[2/3] Training MPS..."
    python train_mps_paths.py \
      --prefix "${PREFIX}" \
      --max-degree "${MAX_DEG}" \
      --n-targets "${N_TARGETS}" \
      --batch-size 306 \
      --num-epochs 150 \
      --bond-dim 20 \
      --lr 1e-4 \
      --l2-reg 0 \
      --lr-factor 0.8 \
      --lr-patience 3 \
      --min-lr 1e-7 \
      --early-stop-patience 50 \
      --grad-clip 1.0 \
      > "logs/${PREFIX}_train.log"

    # 3) Evaluate TN-SHAP
    echo "[3/3] Evaluating TN-SHAP..."
    python tnshap_vandermonde_compare_sparse_poly.py \
      --prefix "${PREFIX}" \
      --max-degree "${MAX_DEG}" \
      --n-targets "${N_TARGETS}" \
      > "logs/${PREFIX}_eval.log" 2>&1

    echo "Finished Task=${task}, D=${D}. Logs in logs/${PREFIX}_*.log"
    echo

  done
done
