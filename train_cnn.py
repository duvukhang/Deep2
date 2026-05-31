import argparse
import logging
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from models.cnn_feature_extractor import CNNFeatureExtractor
from utils.dataset_audit import image_to_label_path, label_dir_from_image_dir, load_yolo_config, parse_label_file
from utils.metrics import multiclass_classification_metrics, classification_report_text, save_json
from utils.plotting import plot_confusion_matrix, plot_learning_curve, write_csv


DEFAULT_CLASS_NAMES = ["eye_closed", "eye_open", "face", "mouth_open"]


class YoloRoiDataset(Dataset):
    def __init__(self, root, split="train", transform=None, padding=0.20, class_names=None):
        self.root = Path(root)
        self.split = "valid" if split == "val" else split
        self.transform = transform
        self.padding = padding
        self.image_dir = self.root / self.split / "images"
        self.label_dir = label_dir_from_image_dir(self.image_dir)
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.samples = []
        self.targets = []

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Khong tim thay {self.image_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Khong tim thay {self.label_dir}")

        image_paths = sorted(
            path
            for path in self.image_dir.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        for image_path in image_paths:
            label_path = image_to_label_path(image_path, self.image_dir, self.label_dir)
            parsed = parse_label_file(label_path, len(self.class_names))
            for instance in parsed["instances"]:
                label = int(instance["class_id"])
                self.samples.append((image_path, label, instance["bbox"]))
                self.targets.append(label)

        if not self.samples:
            raise ValueError(f"Khong co ROI label hop le trong {self.image_dir}")

    def __len__(self):
        return len(self.samples)

    def _crop(self, image, bbox):
        width, height = image.size
        cx, cy, bw, bh = bbox
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height

        pad_x = (x2 - x1) * self.padding
        pad_y = (y2 - y1) * self.padding
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(width, int(x2 + pad_x))
        y2 = min(height, int(y2 + pad_y))

        if x2 <= x1 or y2 <= y1:
            return image
        return image.crop((x1, y1, x2, y2))

    def __getitem__(self, idx):
        image_path, label, bbox = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        image = self._crop(image, bbox)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN feature extractor 512-d cho Hybrid.")
    parser.add_argument("--data_dir", default="driver_drowsiness_Computer_Vision_Model")
    parser.add_argument("--data_yaml", default=None, help="Neu co, doc class names/path tu data.yaml.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--backbone", default="mobilenet_v3_small")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--no_pretrained", action="store_true")
    return parser.parse_args()


def configure_logging():
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/train_cnn.log", encoding="utf-8"),
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


def build_transforms(image_size):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.20, hue=0.04),
        transforms.RandomRotation(degrees=15),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.25),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.14), ratio=(0.3, 3.3)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.12)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def resolve_root_and_classes(args):
    if args.data_yaml:
        config = load_yolo_config(args.data_yaml)
        return Path(config["base_path"]), config["names"]

    root = Path(args.data_dir)
    yaml_path = root / "data.yaml"
    if yaml_path.exists():
        config = load_yolo_config(yaml_path)
        return Path(config["base_path"]), config["names"]

    return root, DEFAULT_CLASS_NAMES


def build_dataset(root, split, transform, class_names):
    split_name = "valid" if split == "val" else split
    yolo_image_dir = Path(root) / split_name / "images"
    imagefolder_dir = Path(root) / split_name

    if yolo_image_dir.exists():
        return YoloRoiDataset(root, split_name, transform=transform, class_names=class_names)

    if imagefolder_dir.exists():
        dataset = datasets.ImageFolder(imagefolder_dir, transform=transform)
        dataset.class_names = dataset.classes
        dataset.targets = [target for _, target in dataset.samples]
        return dataset

    raise FileNotFoundError(f"Khong tim thay split {split_name} trong {root}")


def class_weights(targets, num_classes, device):
    counts = Counter(targets)
    values = []
    total = len(targets)
    for class_id in range(num_classes):
        count = counts.get(class_id, 0)
        if count == 0:
            raise ValueError(f"Class {class_id} khong co sample trong train split.")
        values.append(total / (num_classes * count))
    weights = torch.tensor(values, dtype=torch.float32, device=device)
    return weights / weights.mean(), counts


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    y_true = []
    y_pred = []

    loop = tqdm(loader, leave=False, desc="train" if is_train else "eval")
    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        predictions = torch.argmax(logits.detach(), dim=1)
        losses.append(float(loss.item()))
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(predictions.cpu().numpy().tolist())
        loop.set_postfix(loss=float(loss.item()))

    metrics, cm = multiclass_classification_metrics(y_true, y_pred, labels=list(range(model.num_classes)))
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics, cm, y_true, y_pred


def save_checkpoint(model, path, args, class_names, metrics=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone": args.backbone,
            "feature_dim": args.feature_dim,
            "num_classes": len(class_names),
            "class_names": class_names,
            "image_size": args.image_size,
            "metrics": metrics or {},
        },
        path,
    )


def train_cnn():
    configure_logging()
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)

    root, class_names = resolve_root_and_classes(args)
    train_tf, eval_tf = build_transforms(args.image_size)
    train_dataset = build_dataset(root, "train", train_tf, class_names)
    val_dataset = build_dataset(root, "val", eval_tf, class_names)
    test_dataset = None
    try:
        test_dataset = build_dataset(root, "test", eval_tf, class_names)
    except FileNotFoundError:
        logging.warning("Khong co test split cho CNN, chi validate tren val.")

    if hasattr(train_dataset, "class_names"):
        class_names = train_dataset.class_names

    weights, counts = class_weights(train_dataset.targets, len(class_names), device)
    logging.info("CNN train samples=%d val=%d class_counts=%s", len(train_dataset), len(val_dataset), dict(counts))
    logging.info("Class names: %s", class_names)
    logging.info("Class weights: %s", weights.detach().cpu().tolist())

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

    model = CNNFeatureExtractor(
        backbone=args.backbone,
        feature_dim=args.feature_dim,
        num_classes=len(class_names),
        dropout=args.dropout,
        pretrained=not args.no_pretrained,
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, args.patience // 3),
    )

    Path("weights").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    history = []
    best_f1 = -1.0
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics, val_cm, y_true, y_pred = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["f1"])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_f1": train_metrics["f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        }
        history.append(row)
        write_csv(history, "logs/cnn_metrics.csv")
        plot_learning_curve(history, "plots/cnn_learning_curve.png", "CNN learning curve")
        plot_confusion_matrix(val_cm, class_names, "plots/cnn_confusion_matrix.png", "CNN confusion matrix")

        logging.info(
            "Epoch %03d | train_loss=%.4f train_f1=%.4f | val_loss=%.4f val_f1=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["f1"],
            val_metrics["loss"],
            val_metrics["f1"],
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_count = 0
            save_checkpoint(model, "weights/cnn_best.pth", args, class_names, val_metrics)
            save_json(
                {
                    "best_epoch": epoch,
                    "best_metrics": val_metrics,
                    "classification_report": classification_report_text(y_true, y_pred, class_names),
                    "class_names": class_names,
                    "class_counts": dict(counts),
                },
                "weights/cnn_best_meta.json",
            )
            logging.info("Da luu weights/cnn_best.pth")
        else:
            patience_count += 1
            logging.info("Val F1 khong cai thien. Patience %d/%d", patience_count, args.patience)
            if patience_count >= args.patience:
                logging.info("Dung som CNN de giam overfitting.")
                break

    save_checkpoint(model, "weights/cnn_final.pth", args, class_names, history[-1] if history else {})

    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        test_metrics, test_cm, y_true, y_pred = run_epoch(model, test_loader, criterion, device)
        plot_confusion_matrix(test_cm, class_names, "plots/cnn_confusion_matrix_test.png", "CNN confusion matrix test")
        save_json(
            {
                "metrics": test_metrics,
                "classification_report": classification_report_text(y_true, y_pred, class_names),
            },
            "logs/cnn_test_metrics.json",
        )
        logging.info("CNN test metrics: %s", test_metrics)


if __name__ == "__main__":
    train_cnn()
