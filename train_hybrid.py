import argparse
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.hybrid_model import DriverMonitoringSystem
from utils.metrics import binary_classification_metrics, classification_report_text, save_json
from utils.plotting import plot_confusion_matrix, plot_learning_curve, write_csv


class SequenceHybridDataset(Dataset):
    def __init__(self, data_dir="dataset_extracted", mode="train", seq_len=30, strict=False):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.seq_len = seq_len
        self.strict = strict
        self.mode_dir = self.data_dir / mode
        self.samples = []
        self.class_counts = Counter()
        self.bad_files = []

        if not self.mode_dir.exists():
            raise FileNotFoundError(
                f"Khong tim thay thu muc {self.mode_dir}. "
                "Hay chay extract_features.py truoc khi train Hybrid."
            )

        files = sorted(list(self.mode_dir.glob("*.npy")) + list(self.mode_dir.glob("*.npz")))
        if not files:
            raise FileNotFoundError(
                f"Khong co file .npy/.npz trong {self.mode_dir}. "
                "Training that khong duoc dung dummy data."
            )

        for file_path in files:
            try:
                item = self._read_and_validate(file_path)
            except Exception as exc:
                self.bad_files.append({"file": str(file_path), "error": str(exc)})
                if strict:
                    raise
                continue
            self.samples.append(item)
            self.class_counts[item["label"]] += 1

        if not self.samples:
            raise ValueError(
                f"Khong co sample hop le trong {self.mode_dir}. "
                f"So file loi: {len(self.bad_files)}"
            )

        missing_classes = [label for label in [0, 1] if self.class_counts.get(label, 0) == 0]
        if missing_classes:
            raise ValueError(
                f"Split {mode} thieu class {missing_classes}. "
                "Can co ca normal(0) va drowsy(1) de train/evaluate tin cay."
            )

    def _load_file(self, file_path):
        loaded = np.load(file_path, allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            return {key: loaded[key] for key in loaded.files}
        value = loaded.item() if loaded.shape == () else loaded
        if not isinstance(value, dict):
            raise ValueError("File phai luu dict gom spatial, geo, label.")
        return value

    def _read_and_validate(self, file_path):
        data = self._load_file(file_path)
        for key in ["spatial", "geo", "label"]:
            if key not in data:
                raise KeyError(f"Thieu key '{key}'")

        spatial = np.asarray(data["spatial"], dtype=np.float32)
        geo = np.asarray(data["geo"], dtype=np.float32)
        label = int(data["label"])

        expected_spatial = (self.seq_len, 512)
        expected_geo = (self.seq_len, 6)
        if spatial.shape != expected_spatial:
            raise ValueError(f"spatial shape {spatial.shape}, ky vong {expected_spatial}")
        if geo.shape != expected_geo:
            raise ValueError(f"geo shape {geo.shape}, ky vong {expected_geo}")
        if label not in [0, 1]:
            raise ValueError("label phai la 0 hoac 1")
        if not np.isfinite(spatial).all() or not np.isfinite(geo).all():
            raise ValueError("spatial/geo co NaN hoac Inf")

        geo[:, 5] = np.clip(geo[:, 5], 0.0, 1.0)
        return {
            "spatial": spatial,
            "geo": geo,
            "label": label,
            "file": str(file_path),
            "meta": data.get("meta", {}),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return (
            torch.tensor(item["spatial"], dtype=torch.float32),
            torch.tensor(item["geo"], dtype=torch.float32),
            torch.tensor(item["label"], dtype=torch.long),
        )


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Hybrid spatial+geometric temporal model.")
    parser.add_argument("--data_dir", default="dataset_extracted")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    parser.add_argument("--loss", choices=["auto", "ce", "focal"], default="auto")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--strict", action="store_true", help="Gap file loi thi dung ngay.")
    return parser.parse_args()


def configure_logging():
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/train_hybrid.log", encoding="utf-8"),
        ],
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(device_arg):
    if device_arg:
        text = str(device_arg).lower()
        if text.isdigit():
            return torch.device(f"cuda:{text}" if torch.cuda.is_available() else "cpu")
        return torch.device(text)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(args):
    train_dataset = SequenceHybridDataset(args.data_dir, "train", args.seq_len, args.strict)
    val_dataset = SequenceHybridDataset(args.data_dir, "val", args.seq_len, args.strict)

    logging.info("Train samples: %d | class_counts=%s", len(train_dataset), dict(train_dataset.class_counts))
    logging.info("Val samples: %d | class_counts=%s", len(val_dataset), dict(val_dataset.class_counts))
    if train_dataset.bad_files:
        logging.warning("Train co %d file loi da bo qua.", len(train_dataset.bad_files))
    if val_dataset.bad_files:
        logging.warning("Val co %d file loi da bo qua.", len(val_dataset.bad_files))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_dataset, val_dataset, train_loader, val_loader


def class_weights_from_counts(class_counts, device):
    counts = torch.tensor(
        [class_counts.get(0, 0), class_counts.get(1, 0)],
        dtype=torch.float32,
        device=device,
    )
    if torch.any(counts <= 0):
        raise ValueError("Moi class can co it nhat 1 sample de tinh class weights.")
    weights = counts.sum() / (len(counts) * counts)
    return weights / weights.mean()


def choose_loss(args, class_counts, device):
    class_weights = class_weights_from_counts(class_counts, device)
    counts = [class_counts.get(0, 0), class_counts.get(1, 0)]
    imbalance_ratio = max(counts) / max(1, min(counts))
    use_focal = args.loss == "focal" or (args.loss == "auto" and imbalance_ratio >= 3.0)

    logging.info("Class weights: %s | imbalance_ratio=%.2f", class_weights.detach().cpu().tolist(), imbalance_ratio)
    if use_focal:
        logging.info("Dung FocalLoss vi class imbalance dang ke.")
        return FocalLoss(class_weights, gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)


def run_epoch(model, loader, criterion, device, optimizer=None, grad_clip=5.0):
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    y_true = []
    y_pred = []
    y_score = []

    loop = tqdm(loader, leave=False, desc="train" if is_train else "eval")
    for spatial_seq, geo_seq, labels in loop:
        spatial_seq = spatial_seq.to(device, non_blocking=True)
        geo_seq = geo_seq.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(spatial_seq, geo_seq)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        probabilities = torch.softmax(logits.detach(), dim=1)[:, 1]
        predictions = torch.argmax(logits.detach(), dim=1)

        losses.append(float(loss.item()))
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(predictions.cpu().numpy().tolist())
        y_score.extend(probabilities.cpu().numpy().tolist())
        loop.set_postfix(loss=float(loss.item()))

    metrics, cm = binary_classification_metrics(y_true, y_pred, y_score)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics, cm, y_true, y_pred, y_score


def save_model_state(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def train_hybrid():
    configure_logging()
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    logging.info("Train Hybrid tren device: %s", device)

    Path("weights").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)

    train_dataset, val_dataset, train_loader, val_loader = make_loaders(args)
    model = DriverMonitoringSystem(
        feature_dim=args.feature_dim,
        dropout=args.dropout,
    ).to(device)
    criterion = choose_loss(args, train_dataset.class_counts, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, args.patience // 3),
    )

    history = []
    best_f1 = -1.0
    best_val_loss = float("inf")
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
        )
        val_metrics, val_cm, y_true, y_pred, _ = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            grad_clip=args.grad_clip,
        )
        scheduler.step(val_metrics["f1"])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_f1": train_metrics["f1"],
            "train_recall": train_metrics["recall"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        history.append(row)
        write_csv(history, "logs/hybrid_metrics.csv")
        plot_learning_curve(history, "plots/hybrid_learning_curve.png", "Hybrid learning curve")
        plot_confusion_matrix(val_cm, ["normal", "drowsy"], "plots/hybrid_confusion_matrix.png")

        logging.info(
            "Epoch %03d | train_loss=%.4f train_f1=%.4f | val_loss=%.4f val_f1=%.4f val_recall=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["f1"],
            val_metrics["loss"],
            val_metrics["f1"],
            val_metrics["recall"],
        )

        improved = (
            val_metrics["f1"] > best_f1
            or (np.isclose(val_metrics["f1"], best_f1) and val_metrics["loss"] < best_val_loss)
        )
        if improved:
            best_f1 = val_metrics["f1"]
            best_val_loss = val_metrics["loss"]
            patience_count = 0
            save_model_state(model, "weights/hybrid_best.pth")
            save_json(
                {
                    "args": vars(args),
                    "best_epoch": epoch,
                    "best_metrics": val_metrics,
                    "train_class_counts": dict(train_dataset.class_counts),
                    "val_class_counts": dict(val_dataset.class_counts),
                    "classification_report": classification_report_text(y_true, y_pred, ["normal", "drowsy"]),
                },
                "weights/hybrid_best_meta.json",
            )
            logging.info("Da luu weights/hybrid_best.pth theo val F1.")
        else:
            patience_count += 1
            logging.info("Val F1 khong cai thien. Patience %d/%d", patience_count, args.patience)
            if patience_count >= args.patience:
                logging.info("Dung som de giam overfitting.")
                break

    save_model_state(model, "weights/hybrid_final.pth")
    save_json(
        {
            "args": vars(args),
            "last_epoch": history[-1]["epoch"] if history else 0,
            "history": history,
        },
        "weights/hybrid_final_meta.json",
    )
    logging.info("Hoan tat train Hybrid. Best F1=%.4f", best_f1)


if __name__ == "__main__":
    train_hybrid()
