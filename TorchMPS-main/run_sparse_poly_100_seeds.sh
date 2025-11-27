#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

TASKS=("poly5" "poly10" "sqexp")
DS=(50 100)   # D = 50, 100

# Total number of datapoints
N_TOTAL=1
N_TEST=0          # you can change this if you want a test split
N_TRAIN=${N_TOTAL}
N_TARGETS=1       # number of points whose Shapley we compute

N_SEEDS=100       # seeds 0..99

for task in "${TASKS[@]}"; do
  for D in "${DS[@]}"; do

    echo "==============================================="
    echo "Task=${task}, D=${D}"
    echo "Running ${N_SEEDS} seeds"
    echo "==============================================="

    # choose max_degree for TN-SHAP interpolation
    if [ "${task}" = "poly5" ]; then
      MAX_DEG=5
    elif [ "${task}" = "poly10" ]; then
      MAX_DEG=5
    else
      MAX_DEG=5  # sqexp approx
    fi

    for SEED in $(seq 0 $((N_SEEDS - 1))); do
      PREFIX="${task}_D${D}_seed${SEED}"

      echo
      echo "-----------------------------------------------"
      echo " Seed=${SEED}, PREFIX=${PREFIX}"
      echo "-----------------------------------------------"

      # 1) Generate teacher & data
      echo "[1/3] Generating teacher and data..."
      python poly_teacher.py \
        --prefix "${PREFIX}" \
        --task "${task}" \
        --D "${D}" \
        --n-train "${N_TRAIN}" \
        --n-test "${N_TEST}" \
        --noise-std 0.0 \
        --seed-base "${SEED}" \
        > "logs/${PREFIX}_gen.log" 2>&1

      # 2) Train MPS
      echo "[2/3] Training MPS..."
      TRAIN_START=$(date +%s)
      python train_mps_paths.py \
        --prefix "${PREFIX}" \
        --max-degree "${MAX_DEG}" \
        --n-targets "${N_TARGETS}" \
        --batch-size 306 \
        --num-epochs 150 \
        --bond-dim 30 \
        --lr 2e-4 \
        --l2-reg 0 \
        --lr-factor 0.8 \
        --lr-patience 3 \
        --min-lr 1e-7 \
        --early-stop-patience 50 \
        --grad-clip 1.0 \
        > "logs/${PREFIX}_train.log" 2>&1
      TRAIN_END=$(date +%s)
      TRAIN_TIME=$((TRAIN_END - TRAIN_START))
      echo "TRAIN_TIME_SECONDS=${TRAIN_TIME}" >> "logs/${PREFIX}_train.log"

      # 3) Evaluate TN-SHAP
      echo "[3/3] Evaluating TN-SHAP..."
      EVAL_START=$(date +%s)
      python tnshap_vandermonde_compare_sparse_poly.py \
        --prefix "${PREFIX}" \
        --max-degree "${MAX_DEG}" \
        --n-targets "${N_TARGETS}" \
        > "logs/${PREFIX}_eval.log" 2>&1
      EVAL_END=$(date +%s)
      EVAL_TIME=$((EVAL_END - EVAL_START))
      echo "EVAL_TIME_SECONDS=${EVAL_TIME}" >> "logs/${PREFIX}_eval.log"

      echo "Finished Task=${task}, D=${D}, seed=${SEED}."
      echo "Logs in logs/${PREFIX}_*.log"
    done

    echo
    echo "Completed all ${N_SEEDS} seeds for Task=${task}, D=${D}."
    echo

  done
done
