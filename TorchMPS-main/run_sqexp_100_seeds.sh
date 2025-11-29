#!/usr/bin/env bash
set -e  # Exit on error

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate concepts

EXPO_DIR="expo"
LOG_DIR="${EXPO_DIR}/logs"
mkdir -p "${EXPO_DIR}" "${LOG_DIR}"

TASK="sqexp"
D=100

# Total number of datapoints used to build the teacher dataset
N_TOTAL=1
N_TEST=0
N_TRAIN=${N_TOTAL}
N_TARGETS=1
MAX_DEG=5

N_SEEDS=100   # <-- change if you want fewer/more

echo "==============================================="
echo "Task=${TASK}, D=${D}"
echo "Running ${N_SEEDS} different seeds"
echo "Results will be stored under ${EXPO_DIR}/"
echo "Logs under ${LOG_DIR}/"
echo "==============================================="

for SEED in $(seq 0 $((N_SEEDS - 1))); do
  PREFIX="${TASK}_D${D}_seed${SEED}"

  echo
  echo "==============================================="
  echo "  Seed ${SEED} -> PREFIX=${PREFIX}"
  echo "==============================================="

  # 1) Generate teacher & data
  echo "[1/3] Generating teacher and data (seed=${SEED})..."
  python poly_teacher.py \
    --prefix "${PREFIX}" \
    --task "${TASK}" \
    --D "${D}" \
    --n-train "${N_TRAIN}" \
    --n-test "${N_TEST}" \
    --noise-std 0.0 \
    --seed-base "${SEED}" \
    2>&1 | tee "${LOG_DIR}/${PREFIX}_gen.log"

  if [ $? -ne 0 ]; then
      echo "ERROR: Teacher generation failed for seed=${SEED}! Check ${LOG_DIR}/${PREFIX}_gen.log"
      exit 1
  fi

  # 2) Build Chebyshev path-aug dataset + train MPS
  echo "[2/3] Building paths and training MPS (sqexp, seed=${SEED})..."

  TRAIN_START=$(date +%s)
  python train_mps_sqexp_paths.py \
    --prefix "${PREFIX}" \
    --max-degree "${MAX_DEG}" \
    --n-targets "${N_TARGETS}" \
    --batch-size 306 \
    --num-epochs 100 \
    --bond-dim 50 \
    --lr 2e-4 \
    --l2-reg 0.0 \
    --lr-factor 0.8 \
    --lr-patience 5 \
    --min-lr 1e-7 \
    --early-stop-patience 100 \
    --grad-clip 5.0 \
    --seed "${SEED}" \
    2>&1 | tee "${LOG_DIR}/${PREFIX}_train_sqexp.log"
  TRAIN_END=$(date +%s)
  TRAIN_TIME=$((TRAIN_END - TRAIN_START))
  echo "TRAIN_TIME_SECONDS=${TRAIN_TIME}" | tee -a "${LOG_DIR}/${PREFIX}_train_sqexp.log"

  if [ $? -ne 0 ]; then
      echo "ERROR: Training failed for seed=${SEED}! Check ${LOG_DIR}/${PREFIX}_train_sqexp.log"
      exit 1
  fi

  # 3) Evaluate TN-SHAP
  echo "[3/3] Evaluating TN-SHAP (teacher vs MPS, seed=${SEED})..."

  EVAL_START=$(date +%s)
  python tnshap_sqexp_vandermonde_eval_gi.py \
    --prefix "${PREFIX}" \
    --max-degree "${MAX_DEG}" \
    --n-targets "${N_TARGETS}" \
    --expo-dir "${EXPO_DIR}" \
    2>&1 | tee "${LOG_DIR}/${PREFIX}_eval_sqexp.log"
  EVAL_END=$(date +%s)
  EVAL_TIME=$((EVAL_END - EVAL_START))
  echo "EVAL_TIME_SECONDS=${EVAL_TIME}" | tee -a "${LOG_DIR}/${PREFIX}_eval_sqexp.log"

  if [ $? -ne 0 ]; then
      echo "ERROR: Evaluation failed for seed=${SEED}! Check ${LOG_DIR}/${PREFIX}_eval_sqexp.log"
      exit 1
  fi

  echo
  echo "Finished seed=${SEED}, PREFIX=${PREFIX}."
  echo "Artifacts: ${EXPO_DIR}/${PREFIX}_tnshap_targets.pt, ${EXPO_DIR}/${PREFIX}_mps.pt"
  echo "Logs:      ${LOG_DIR}/${PREFIX}_*.log"
done

echo
echo "All seeds finished. Next: run aggregate_sqexp_stats.py to compute averages."
echo
