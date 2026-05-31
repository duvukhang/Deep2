import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def plot_learning_curve(history, output_path, title="Learning curve"):
    if not history:
        return

    output_path = Path(output_path)
    ensure_parent(output_path)

    epochs = [row.get("epoch", idx + 1) for idx, row in enumerate(history)]
    keys = [
        key
        for key in history[0].keys()
        if key != "epoch" and isinstance(history[0].get(key), (int, float))
    ]

    plt.figure(figsize=(10, 6))
    for key in keys:
        values = [row.get(key) for row in history]
        if any(value is not None for value in values):
            plt.plot(epochs, values, marker="o", linewidth=1.8, label=key)

    plt.title(title)
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_confusion_matrix(cm, class_names, output_path, title="Confusion matrix"):
    output_path = Path(output_path)
    ensure_parent(output_path)

    cm = np.asarray(cm)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=30, ha="right")
    plt.yticks(tick_marks, class_names)

    threshold = cm.max() / 2.0 if cm.size and cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_csv(rows, output_path, fieldnames=None):
    output_path = Path(output_path)
    ensure_parent(output_path)
    if not rows:
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_yolo_results_csv(results_csv, output_path):
    results_csv = Path(results_csv)
    if not results_csv.exists():
        return False

    import pandas as pd

    df = pd.read_csv(results_csv)
    df.columns = [column.strip() for column in df.columns]
    if "epoch" not in df.columns:
        df.insert(0, "epoch", range(1, len(df) + 1))

    preferred = [
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]
    columns = [column for column in preferred if column in df.columns]
    if not columns:
        return False

    output_path = Path(output_path)
    ensure_parent(output_path)

    plt.figure(figsize=(11, 7))
    for column in columns:
        plt.plot(df["epoch"], df[column], linewidth=1.7, label=column)

    plt.title("YOLO train/val metrics")
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True
