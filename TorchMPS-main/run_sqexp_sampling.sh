#!/usr/bin/env bash
set -e

EVAL_FILE="fixed_eval_indices.txt"
LOG_DIR="logs_sampling_sqexp"
mkdir -p "$LOG_DIR"

# ---------------------------------------------
# 1. Create a fixed list of 100 evaluation indices
#    -> always 0, 1, 2, ..., 99
# ---------------------------------------------
rm -f "$EVAL_FILE"
for i in $(seq 0 99); do
    echo "$i" >> "$EVAL_FILE"
done
echo "[INFO] Fixed eval indices written to $EVAL_FILE"

# ---------------------------------------------
# 2. sqexp tasks (D = 50, 100)
# ---------------------------------------------
PREFIXES=(
  "sqexp_D50"
  "sqexp_D100"
)

for PREFIX in "${PREFIXES[@]}"; do
    echo
    echo "========================================"
    echo "[INFO] Running prefix: $PREFIX"
    echo "========================================"

    # Infer support size from dimension in prefix name
    if [[ "$PREFIX" == *"D50" ]]; then
        SUPPORT_SIZE=17
    elif [[ "$PREFIX" == *"D100" ]]; then
        SUPPORT_SIZE=33
    else
        echo "[WARN] Could not infer D from prefix $PREFIX, defaulting SUPPORT_SIZE=17"
        SUPPORT_SIZE=17
    fi

    # For sqexp we use t_min = 0.2, t_max = 1.0
    T_MIN=0.2
    T_MAX=1.0

    LOG_FILE="${LOG_DIR}/sampling_${PREFIX}.log"

    echo "[INFO] SUPPORT_SIZE=$SUPPORT_SIZE, T_MIN=$T_MIN, T_MAX=$T_MAX"
    echo "[INFO] Logging to $LOG_FILE"

    python local_neighborhood_vs_coalitions_treeshap_sqexp.py \
        --prefix "$PREFIX" \
        --support-size "$SUPPORT_SIZE" \
        --t-min "$T_MIN" \
        --t-max "$T_MAX" \
        --n-eval 100 \
        --seed 0 \
        --eval-indices "$EVAL_FILE" \
        > "$LOG_FILE" 2>&1

    echo "[INFO] Done: $PREFIX"
done

echo
echo "[INFO] All sqexp runs completed. Logs are in $LOG_DIR/"
