import argparse
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.hybrid_model import DriverMonitoringSystem
from train_hybrid import SequenceHybridDataset, run_epoch
from utils.metrics import classification_report_text, save_json
from utils.plotting import plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Hybrid model tren dataset_extracted/test.")
    parser.add_argument("--data_dir", default="dataset_extracted")
    parser.add_argument("--weights", default="weights/hybrid_best.pth")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def configure_logging():
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/evaluate_hybrid.log", encoding="utf-8"),
        ],
    )


def select_device(device_arg):
    if device_arg:
        text = str(device_arg).lower()
        if text.isdigit():
            return torch.device(f"cuda:{text}" if torch.cuda.is_available() else "cpu")
        return torch.device(text)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_state_dict_flexible(path, device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def main():
    configure_logging()
    args = parse_args()
    device = select_device(args.device)

    dataset = SequenceHybridDataset(args.data_dir, args.split, args.seq_len, args.strict)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = DriverMonitoringSystem(feature_dim=args.feature_dim).to(device)
    model.load_state_dict(load_state_dict_flexible(args.weights, device))
    model.eval()

    criterion = nn.CrossEntropyLoss()
    metrics, cm, y_true, y_pred, y_score = run_epoch(
        model,
        loader,
        criterion,
        device,
        optimizer=None,
    )

    report = classification_report_text(y_true, y_pred, ["normal", "drowsy"])
    logging.info("Metrics %s: %s", args.split, metrics)
    logging.info("Classification report:\n%s", report)

    plot_confusion_matrix(
        cm,
        ["normal", "drowsy"],
        f"plots/hybrid_confusion_matrix_{args.split}.png",
        f"Hybrid confusion matrix ({args.split})",
    )
    save_json(
        {
            "split": args.split,
            "weights": args.weights,
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        },
        f"logs/hybrid_eval_{args.split}.json",
    )

    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Precision:", round(metrics["precision"], 4))
    print("Recall:", round(metrics["recall"], 4))
    print("F1:", round(metrics["f1"], 4))
    print("ROC-AUC:", None if metrics["roc_auc"] is None else round(metrics["roc_auc"], 4))
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    main()
