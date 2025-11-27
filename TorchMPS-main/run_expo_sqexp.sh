#!/usr/bin/env bash
set -e  # Exit on error

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate concepts

EXPO_DIR="expo"
LOG_DIR="${EXPO_DIR}/logs"
mkdir -p "${EXPO_DIR}" "${LOG_DIR}"

TASK="sqexp"
D=50

# Total number of datapoints used to build the teacher dataset
N_TOTAL=1
N_TEST=0            # you can change this if you want a test split
N_TRAIN=${N_TOTAL}  # here we just use all as "train" for simplicity
N_TARGETS=1     # M points whose Shapley we care about

PREFIX="${TASK}_D${D}"
MAX_DEG=5        # polynomial degree in t for sqexp interpolation

echo "==============================================="
echo "Task=${TASK}, D=${D}, PREFIX=${PREFIX}"
echo "Results will be stored under ${EXPO_DIR}/"
echo "==============================================="

# 1) Generate teacher & data (Gaussian) via poly_teacher.py
echo "[1/3] Generating teacher and data..."
python poly_teacher.py \
  --prefix "${PREFIX}" \
  --task "${TASK}" \
  --D "${D}" \
  --n-train "${N_TRAIN}" \
  --n-test "${N_TEST}" \
  --noise-std 0.0 \
  --seed-base 2 \
  2>&1 | tee "${LOG_DIR}/${PREFIX}_gen.log"

if [ $? -ne 0 ]; then
    echo "ERROR: Teacher generation failed! Check ${LOG_DIR}/${PREFIX}_gen.log"
    exit 1
fi

# 2) Build Chebyshev path-aug dataset + train MPS (saved under expo/)
echo "[2/3] Building paths and training MPS (sqexp)..."
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
  2>&1 | tee "${LOG_DIR}/${PREFIX}_train_sqexp.log"

if [ $? -ne 0 ]; then
    echo "ERROR: Training failed! Check ${LOG_DIR}/${PREFIX}_train_sqexp.log"
    exit 1
fi

# 3) Evaluate TN-SHAP using Vandermonde and the same Chebyshev grid
echo "[3/3] Evaluating TN-SHAP (teacher vs MPS)..."
python tnshap_sqexp_vandermonde_eval_gi.py \
  --prefix "${PREFIX}" \
  --max-degree "${MAX_DEG}" \
  --n-targets "${N_TARGETS}" \
  --expo-dir "${EXPO_DIR}" \
  2>&1 | tee "${LOG_DIR}/${PREFIX}_eval_sqexp.log"

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation failed! Check ${LOG_DIR}/${PREFIX}_eval_sqexp.log"
    exit 1
fi

echo
echo "Finished Task=${TASK}, D=${D}."
echo "Artifacts: ${EXPO_DIR}/${PREFIX}_tnshap_targets.pt, ${EXPO_DIR}/${PREFIX}_mps.pt"
echo "Logs:      ${LOG_DIR}/${PREFIX}_*.log"
echo
