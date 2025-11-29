#!/usr/bin/env bash
set -e

# ---------------------------------------------
# 0. Config
# ---------------------------------------------

EVAL_FILE="fixed_eval_indices.txt"
LOG_DIR="logs_sampling"
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
# 2. Define tasks
#    poly5, poly10, sqexp
#    D = 50, 100
#
#   Prefixes assumed:
#     poly5_D50, poly5_D100
#     poly10_D50, poly10_D100
#     sqexp_D50, sqexp_D100
# ---------------------------------------------

PREFIXES=(
  "poly5_D50"
  "poly5_D100"
  "poly10_D50"
  "poly10_D100"
  "sqexp_D50"
  "sqexp_D100"
)


# ---------------------------------------------
# 3. Loop over tasks and run Python
#    - support-size: 17 for D=50, 33 for D=100
#    - t-min: 0.0 for poly*, 0.2 for sqexp*
#    - t-max: 1.0 for all
#    - n-eval: 100
#    - same eval indices file
#    - seed: 0
# ---------------------------------------------

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

    # t-min depends on teacher type
    if [[ "$PREFIX" == sqexp_* ]]; then
        T_MIN=0.2
    else
        T_MIN=0.0
    fi

    T_MAX=1.0

    LOG_FILE="${LOG_DIR}/sampling_${PREFIX}.log"

    echo "[INFO] SUPPORT_SIZE=$SUPPORT_SIZE, T_MIN=$T_MIN, T_MAX=$T_MAX"
    echo "[INFO] Logging to $LOG_FILE"

    python local_neighborhood_vs_coalitions_treeshap.py \
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
echo "[INFO] All runs completed. Logs are in $LOG_DIR/"
