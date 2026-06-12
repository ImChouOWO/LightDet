# watch_train.py

import os
import time
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def read_jsonl(path):
    rows = []

    if not os.path.exists(path):
        return pd.DataFrame()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    return pd.DataFrame(rows)


def plot_epoch_metrics(metrics_path, output_path):
    df = read_jsonl(metrics_path)

    if df.empty:
        return

    if "epoch" not in df.columns:
        return

    plt.figure(figsize=(12, 6))

    if "train_loss" in df.columns:
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")

    if "val_loss" in df.columns:
        plt.plot(df["epoch"], df["val_loss"], label="val_loss")

    if "map50" in df.columns:
        plt.plot(df["epoch"], df["map50"], label="mAP50")

    if "precision" in df.columns:
        plt.plot(df["epoch"], df["precision"], label="precision")

    if "recall" in df.columns:
        plt.plot(df["epoch"], df["recall"], label="recall")

    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--output", type=str, default="training_curve.png")
    parser.add_argument("--interval", type=float, default=5.0)

    args = parser.parse_args()

    while True:
        plot_epoch_metrics(args.metrics, args.output)
        print(f"updated: {args.output}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()