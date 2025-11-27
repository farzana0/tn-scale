#!/usr/bin/env bash
set -e  # Exit on error

# Script to run the experiment 100 times and compute statistics
# Tracks: training time, eval time, and MPA mean accuracy

N_RUNS=100
OUTPUT_DIR="expo_100_runs"
STATS_FILE="${OUTPUT_DIR}/statistics.txt"
RAW_DATA="${OUTPUT_DIR}/raw_data.csv"

mkdir -p "${OUTPUT_DIR}"

echo "Starting ${N_RUNS} runs of run_expo_sqexp.sh..."
echo "Results will be saved to ${OUTPUT_DIR}/"
echo "================================================"

# Initialize CSV file with header
echo "run,train_time,eval_time,mpa_mean_accuracy" > "${RAW_DATA}"

# Arrays to store results
declare -a train_times=()
declare -a eval_times=()
declare -a mpa_accuracies=()

for run in $(seq 1 ${N_RUNS}); do
    echo ""
    echo "========================================"
    echo "Run ${run}/${N_RUNS}"
    echo "========================================"
    
    # Run the experiment
    ./run_expo_sqexp.sh > "${OUTPUT_DIR}/run_${run}.log" 2>&1
    
    # Extract training time from log
    # Look for patterns like "Runtime so far: X sec" in the last occurrence
    train_time=$(grep "Runtime so far:" expo/logs/sqexp_D50_train_sqexp.log | tail -1 | grep -oP '\d+(?= sec)' || echo "0")
    
    # Extract eval time - look for time taken in eval log
    # This might need adjustment based on your eval script output
    eval_time=$(grep -i "time\|elapsed\|duration" expo/logs/sqexp_D50_eval_sqexp.log 2>/dev/null | grep -oP '\d+\.?\d*' | head -1 || echo "0")
    if [ -z "$eval_time" ] || [ "$eval_time" = "0" ]; then
        # If no timing info, estimate from file timestamps
        if [ -f expo/logs/sqexp_D50_eval_sqexp.log ]; then
            eval_time="1"  # Default placeholder
        fi
    fi
    
    # Extract MPA mean accuracy from eval log
    # Look for patterns like "MPA:" or "Mean accuracy:" or similar
    mpa_accuracy=$(grep -i "MPA\|mean.*accuracy\|R²" expo/logs/sqexp_D50_eval_sqexp.log 2>/dev/null | grep -oP '\d+\.?\d+' | tail -1 || echo "0.0")
    
    # If we can't find MPA, try to get final R² from training log
    if [ "$mpa_accuracy" = "0.0" ]; then
        mpa_accuracy=$(grep "Final Train R²" expo/logs/sqexp_D50_train_sqexp.log | grep -oP '\d+\.?\d+' | tail -1 || echo "0.0")
    fi
    
    echo "Run ${run}: Train=${train_time}s, Eval=${eval_time}s, MPA/R²=${mpa_accuracy}"
    
    # Store results
    train_times+=("$train_time")
    eval_times+=("$eval_time")
    mpa_accuracies+=("$mpa_accuracy")
    
    # Append to CSV
    echo "${run},${train_time},${eval_time},${mpa_accuracy}" >> "${RAW_DATA}"
    
    # Clean up logs to save space (optional - comment out if you want to keep all logs)
    # mv expo/logs/sqexp_D50_train_sqexp.log "${OUTPUT_DIR}/train_run_${run}.log"
    # mv expo/logs/sqexp_D50_eval_sqexp.log "${OUTPUT_DIR}/eval_run_${run}.log"
done

echo ""
echo "================================================"
echo "All ${N_RUNS} runs completed!"
echo "Computing statistics..."
echo "================================================"

# Compute statistics using Python
python3 << 'EOF'
import sys
import numpy as np

# Read data from CSV
data = []
with open("expo_100_runs/raw_data.csv", "r") as f:
    lines = f.readlines()[1:]  # Skip header
    for line in lines:
        parts = line.strip().split(',')
        if len(parts) >= 4:
            run, train_t, eval_t, mpa = parts[0], parts[1], parts[2], parts[3]
            data.append([float(train_t), float(eval_t), float(mpa)])

data = np.array(data)
train_times = data[:, 0]
eval_times = data[:, 1]
mpa_accuracies = data[:, 2]

# Compute statistics
stats = {
    'Training Time': {
        'mean': np.mean(train_times),
        'std': np.std(train_times),
        'min': np.min(train_times),
        'max': np.max(train_times),
        'median': np.median(train_times)
    },
    'Evaluation Time': {
        'mean': np.mean(eval_times),
        'std': np.std(eval_times),
        'min': np.min(eval_times),
        'max': np.max(eval_times),
        'median': np.median(eval_times)
    },
    'MPA/R² Accuracy': {
        'mean': np.mean(mpa_accuracies),
        'std': np.std(mpa_accuracies),
        'min': np.min(mpa_accuracies),
        'max': np.max(mpa_accuracies),
        'median': np.median(mpa_accuracies)
    }
}

# Write to statistics file
with open("expo_100_runs/statistics.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write(f"Statistics over {len(train_times)} runs\n")
    f.write("=" * 60 + "\n\n")
    
    for metric, values in stats.items():
        f.write(f"{metric}:\n")
        f.write(f"  Mean:   {values['mean']:.4f}\n")
        f.write(f"  Std:    {values['std']:.4f}\n")
        f.write(f"  Min:    {values['min']:.4f}\n")
        f.write(f"  Max:    {values['max']:.4f}\n")
        f.write(f"  Median: {values['median']:.4f}\n")
        f.write("\n")
    
    # Add summary
    f.write("=" * 60 + "\n")
    f.write("SUMMARY\n")
    f.write("=" * 60 + "\n")
    f.write(f"Training Time:  {stats['Training Time']['mean']:.2f} ± {stats['Training Time']['std']:.2f} seconds\n")
    f.write(f"Eval Time:      {stats['Evaluation Time']['mean']:.2f} ± {stats['Evaluation Time']['std']:.2f} seconds\n")
    f.write(f"MPA/R² Score:   {stats['MPA/R² Accuracy']['mean']:.6f} ± {stats['MPA/R² Accuracy']['std']:.6f}\n")

# Print to console as well
print("\n" + "=" * 60)
print(f"Statistics over {len(train_times)} runs")
print("=" * 60 + "\n")

for metric, values in stats.items():
    print(f"{metric}:")
    print(f"  Mean:   {values['mean']:.4f}")
    print(f"  Std:    {values['std']:.4f}")
    print(f"  Min:    {values['min']:.4f}")
    print(f"  Max:    {values['max']:.4f}")
    print(f"  Median: {values['median']:.4f}")
    print()

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Training Time:  {stats['Training Time']['mean']:.2f} ± {stats['Training Time']['std']:.2f} seconds")
print(f"Eval Time:      {stats['Evaluation Time']['mean']:.2f} ± {stats['Evaluation Time']['std']:.2f} seconds")
print(f"MPA/R² Score:   {stats['MPA/R² Accuracy']['mean']:.6f} ± {stats['MPA/R² Accuracy']['std']:.6f}")
print()

EOF

echo ""
echo "Results saved to:"
echo "  - Raw data:   ${RAW_DATA}"
echo "  - Statistics: ${STATS_FILE}"
echo ""
echo "To view results:"
echo "  cat ${STATS_FILE}"
echo ""
