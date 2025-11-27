#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np

def parse_train_time(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r") as f:
        for line in f:
            if "TRAIN_TIME_SECONDS=" in line:
                try:
                    return float(line.strip().split("=")[-1])
                except ValueError:
                    return None
    return None

def parse_eval_time(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r") as f:
        for line in f:
            if "EVAL_TIME_SECONDS=" in line:
                try:
                    return float(line.strip().split("=")[-1])
                except ValueError:
                    return None
    return None

def parse_mps_accuracy(log_path):
    """
    For logs with lines like:
      MPS Mean Acc:     0.8824
    """
    if not os.path.exists(log_path):
        return None
    acc = None
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("MPS Mean Acc:"):
                try:
                    acc = float(line.split(":")[1])
                except (IndexError, ValueError):
                    pass
    return acc

def summarize(arr):
    if len(arr) == 0:
        return None, None
    arr = np.array(arr, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    return mean, std

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sqexp")
    parser.add_argument("--D", type=int, default=50)
    parser.add_argument("--log-dir", type=str, default="expo/logs",
                        help="Directory where *_eval_sqexp.log and *_train_sqexp.log live")
    parser.add_argument("--out-dir", type=str, default=".",
                        help="Where to write the summary txt file")
    args = parser.parse_args()

    # patterns we try for eval logs:
    #   sqexp_D50_eval_sqexp.log
    #   sqexp_D50_seed0_eval_sqexp.log, etc.
    base = f"{args.task}_D{args.D}"
    pattern1 = os.path.join(args.log_dir, f"{base}_eval_sqexp.log")
    pattern2 = os.path.join(args.log_dir, f"{base}_seed*_eval_sqexp.log")

    eval_logs = []
    if os.path.exists(pattern1):
        eval_logs.append(pattern1)
    eval_logs.extend(glob.glob(pattern2))
    eval_logs = sorted(set(eval_logs))

    if not eval_logs:
        print(f"No eval logs found matching {pattern1} or {pattern2}")
        return

    accs = []
    train_times = []
    eval_times = []

    for eval_log in eval_logs:
        # infer train log name by mirroring suffix
        train_log = eval_log.replace("_eval_sqexp.log", "_train_sqexp.log")

        acc = parse_mps_accuracy(eval_log)
        t_train = parse_train_time(train_log)
        t_eval = parse_eval_time(eval_log)

        if acc is None or t_train is None or t_eval is None:
            print(f"[WARN] Skipping {os.path.basename(eval_log)} "
                  f"(acc={acc}, train={t_train}, eval={t_eval})")
            continue

        accs.append(acc)
        train_times.append(t_train)
        eval_times.append(t_eval)

    print("========================================")
    print(f" Task={args.task}, D={args.D}")
    print(f" Completed runs with full metrics: {len(accs)} / {len(eval_logs)}")
    print("========================================")

    if len(accs) == 0:
        print("No valid runs found, aborting.")
        return

    acc_mean, acc_std = summarize(accs)
    train_mean, _ = summarize(train_times)
    eval_mean, _ = summarize(eval_times)

    print(f"MPS Shapley accuracy: mean={acc_mean:.4f}, std={acc_std:.4f}")
    print(f"Training time (s):    mean={train_mean:.2f}")
    print(f"Evaluation time (s):  mean={eval_mean:.2f}")

    # Write summary
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(
        args.out_dir, f"{args.task}_D{args.D}_summary_auto.txt"
    )
    with open(summary_path, "w") as f:
        f.write(f"Task={args.task}, D={args.D}\n")
        f.write(f"Runs used: {len(accs)} of {len(eval_logs)}\n")
        f.write("\n")
        f.write(f"MPS Shapley accuracy mean: {acc_mean:.6f}\n")
        f.write(f"MPS Shapley accuracy std:  {acc_std:.6f}\n")
        f.write(f"Training time mean (s):    {train_mean:.6f}\n")
        f.write(f"Evaluation time mean (s):  {eval_mean:.6f}\n")

    print()
    print(f"Wrote summary to {summary_path}")

if __name__ == "__main__":
    main()
