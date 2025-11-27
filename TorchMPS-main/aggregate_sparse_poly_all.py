#!/usr/bin/env python3
import os
import re
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
    Parse 'MPS Mean Acc: 0.7059' from eval log.
    """
    if not os.path.exists(log_path):
        return None

    acc = None
    with open(log_path, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("MPS Mean Acc:"):
                try:
                    acc = float(stripped.split(":")[1])
                except (IndexError, ValueError):
                    pass
    return acc


def parse_mps_r2_from_eval(log_path):
    """
    Parse 'MPS vs path-aug ground truth:     R2 = 0.9236'
    from eval log. We treat this as the relevant train/path R².
    """
    if not os.path.exists(log_path):
        return None

    r2 = None
    pattern = re.compile(r"MPS vs path-aug ground truth:\s*R2\s*=\s*([0-9]*\.?[0-9]+)")
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                try:
                    r2 = float(m.group(1))
                except ValueError:
                    pass
    return r2


def summarize(values):
    if len(values) == 0:
        return None, None
    arr = np.array(values, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["poly5", "poly10", "sqexp"],
    )
    parser.add_argument(
        "--Ds",
        type=int,
        nargs="+",
        default=[50, 100],
    )
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument(
        "--out",
        type=str,
        default="sparse_poly_all_summary.tsv",
    )
    args = parser.parse_args()

    rows = []

    print("=========================================================")
    print(" Aggregating metrics over tasks, D, seeds")
    print(" log_dir = ", args.log_dir)
    print(" tasks   = ", args.tasks)
    print(" Ds      = ", args.Ds)
    print(" n_seeds = ", args.n_seeds)
    print("=========================================================\n")

    for task in args.tasks:
        for D in args.Ds:
            accs = []
            r2s = []
            train_times = []
            eval_times = []

            for seed in range(args.n_seeds):
                prefix = f"{task}_D{D}_seed{seed}"
                train_log = os.path.join(args.log_dir, f"{prefix}_train.log")
                eval_log = os.path.join(args.log_dir, f"{prefix}_eval.log")

                acc = parse_mps_accuracy(eval_log)
                r2 = parse_mps_r2_from_eval(eval_log)
                t_train = parse_train_time(train_log)
                t_eval = parse_eval_time(eval_log)

                if acc is None or r2 is None or t_train is None or t_eval is None:
                    # uncomment for debugging:
                    # print(f"[WARN] Missing metrics for {prefix}: "
                    #       f"acc={acc}, r2={r2}, train_time={t_train}, eval_time={t_eval}")
                    continue

                accs.append(acc)
                r2s.append(r2)
                train_times.append(t_train)
                eval_times.append(t_eval)

            n_used = len(accs)
            print(f"Task={task}, D={D}: used {n_used} / {args.n_seeds} seeds")
            if n_used == 0:
                print("  -> No complete seeds; skipping.\n")
                continue

            acc_mean, acc_std = summarize(accs)
            r2_mean, r2_std = summarize(r2s)
            train_time_mean, _ = summarize(train_times)
            eval_time_mean, _ = summarize(eval_times)

            print(f"  MPS mean accuracy     mean={acc_mean:.4f}, std={acc_std:.4f}")
            print(f"  MPS R2 (path-aug)     mean={r2_mean:.4f}, std={r2_std:.4f}")
            print(f"  Training time (sec)   mean={train_time_mean:.2f}")
            print(f"  Evaluation time (sec) mean={eval_time_mean:.2f}")
            print()

            rows.append(
                {
                    "task": task,
                    "D": D,
                    "n_seeds_used": n_used,
                    "n_seeds_total": args.n_seeds,
                    "mps_acc_mean": acc_mean,
                    "mps_acc_std": acc_std,
                    "mps_r2_mean": r2_mean,
                    "mps_r2_std": r2_std,
                    "train_time_mean_s": train_time_mean,
                    "eval_time_mean_s": eval_time_mean,
                }
            )

    if len(rows) == 0:
        print("No data aggregated; not writing summary file.")
        return

    header = [
        "task",
        "D",
        "n_seeds_used",
        "n_seeds_total",
        "mps_acc_mean",
        "mps_acc_std",
        "mps_r2_mean",
        "mps_r2_std",
        "train_time_mean_s",
        "eval_time_mean_s",
    ]

    with open(args.out, "w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write(
                "\t".join(
                    [
                        str(row[col]) if row[col] is not None else "NA"
                        for col in header
                    ]
                )
                + "\n"
            )

    print(f"Wrote summary to {args.out}")


if __name__ == "__main__":
    main()
