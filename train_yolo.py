import argparse
import logging
import shutil
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

from utils.dataset_audit import audit_yolo_dataset, write_audit_report
from utils.plotting import plot_yolo_results_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO driver-drowsiness detector.")
    parser.add_argument("--data", default="configs/data.yaml", help="YOLO data.yaml.")
    parser.add_argument("--model", default="yolo11n.pt", help="yolo11n.pt, yolo11s.pt hoac path local.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="'0', 'cpu', 'cuda'... Neu bo trong se tu chon.")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="yolo_drowsy")
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0, help="Windows nen de 0 neu hay loi dataloader.")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--freeze_layers", type=int, default=0, help="Dong bang N layer dau neu dataset nho.")
    parser.add_argument("--freeze_epochs", type=int, default=0, help="Warmup freeze N epoch truoc khi train full.")
    parser.add_argument("--close_mosaic", type=int, default=15)
    parser.add_argument("--skip_audit", action="store_true", help="Bo qua audit dataset truoc train.")
    parser.add_argument("--skip_test_val", action="store_true", help="Khong chay validation test sau train.")
    return parser.parse_args()


def configure_logging():
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/train_yolo.log", encoding="utf-8"),
        ],
    )


def select_device(device_arg):
    if device_arg:
        return device_arg
    return 0 if torch.cuda.is_available() else "cpu"


def resolve_model_path(model_name):
    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)

    local_weight = Path("weights") / model_name
    if local_weight.exists():
        return str(local_weight)

    return model_name


def supported_ultralytics_keys():
    try:
        from ultralytics.cfg import DEFAULT_CFG

        if isinstance(DEFAULT_CFG, dict):
            return set(DEFAULT_CFG.keys())
        return set(vars(DEFAULT_CFG).keys())
    except Exception:
        return None


def filter_train_kwargs(kwargs):
    keys = supported_ultralytics_keys()
    if not keys:
        return kwargs

    filtered = {}
    for key, value in kwargs.items():
        if key in keys:
            filtered[key] = value
        else:
            logging.warning("Bo qua tham so YOLO khong duoc version Ultralytics nay ho tro: %s", key)
    return filtered


def build_train_kwargs(args, name, epochs, freeze_layers=0):
    kwargs = {
        "data": args.data,
        "epochs": epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": select_device(args.device),
        "workers": args.workers,
        "project": args.project,
        "name": name,
        "exist_ok": True,
        "seed": args.seed,
        "deterministic": True,
        "patience": args.patience,
        "optimizer": "AdamW",
        "weight_decay": args.weight_decay,
        "cos_lr": True,
        "close_mosaic": args.close_mosaic,
        "freeze": freeze_layers,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        # Driver-facing augmentation: lighting, viewpoint, mild occlusion.
        "hsv_h": 0.015,
        "hsv_s": 0.60,
        "hsv_v": 0.50,
        "degrees": 12.0,
        "translate": 0.10,
        "scale": 0.45,
        "shear": 3.0,
        "perspective": 0.0007,
        "flipud": 0.0,
        "fliplr": 0.50,
        "mosaic": 0.85,
        "mixup": 0.05,
        "copy_paste": 0.05,
        "erasing": 0.20,
        "amp": True,
        "plots": True,
        "save": True,
        "val": True,
    }
    return filter_train_kwargs(kwargs)


def summarize_audit(report):
    for split, item in report["splits"].items():
        logging.info(
            "%s: %d images, %d labels, class_counts=%s",
            split,
            item["num_images"],
            item["num_labels"],
            item["class_counts"],
        )
    if report["warnings"]:
        logging.warning("Audit co %d canh bao. Nen xem dataset_report.txt truoc khi train dai.", len(report["warnings"]))
        for warning in report["warnings"][:10]:
            logging.warning(" - %s", warning)


def make_resolved_data_yaml(data_yaml, report):
    if report is None:
        return data_yaml

    source = Path(data_yaml)
    with source.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data["path"] = report["base_path"]
    output_path = Path("logs") / "yolo_data_resolved.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    logging.info("Da tao YAML train YOLO voi path tuyet doi: %s", output_path)
    return str(output_path)


def copy_training_artifacts(save_dir):
    save_dir = Path(save_dir)
    Path("weights").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)

    best_src = save_dir / "weights" / "best.pt"
    last_src = save_dir / "weights" / "last.pt"
    if best_src.exists():
        shutil.copy2(best_src, "weights/yolo_drowsy_best.pt")
        logging.info("Da copy best.pt -> weights/yolo_drowsy_best.pt")
    else:
        logging.warning("Khong tim thay best.pt tai %s", best_src)

    if last_src.exists():
        shutil.copy2(last_src, "weights/yolo_drowsy_last.pt")

    metrics_src = save_dir / "results.csv"
    if metrics_src.exists():
        shutil.copy2(metrics_src, "logs/yolo_metrics.csv")
        plot_yolo_results_csv(metrics_src, "plots/yolo_learning_curve.png")

    results_png = save_dir / "results.png"
    if results_png.exists():
        shutil.copy2(results_png, "plots/yolo_results.png")


def print_detection_metrics(metrics, split_name):
    box = getattr(metrics, "box", None)
    if box is None:
        logging.info("%s metrics: %s", split_name, metrics)
        return

    precision = getattr(box, "mp", None)
    recall = getattr(box, "mr", None)
    map50 = getattr(box, "map50", None)
    map5095 = getattr(box, "map", None)
    logging.info(
        "%s | precision=%.4f recall=%.4f mAP50=%.4f mAP50-95=%.4f",
        split_name,
        float(precision) if precision is not None else -1.0,
        float(recall) if recall is not None else -1.0,
        float(map50) if map50 is not None else -1.0,
        float(map5095) if map5095 is not None else -1.0,
    )


def run_validation(weights_path, args, report):
    model = YOLO(str(weights_path))
    common = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": select_device(args.device),
        "workers": args.workers,
        "plots": True,
    }

    val_metrics = model.val(split="val", **common)
    print_detection_metrics(val_metrics, "valid")

    test_info = report["splits"].get("test") if report else None
    if test_info and test_info["num_images"] > 0:
        test_metrics = model.val(split="test", **common)
        print_detection_metrics(test_metrics, "test")
    else:
        logging.warning("Khong co test split hoac test split rong, bo qua validate test.")


def train_yolo():
    configure_logging()
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs phai > 0")

    report = None
    if not args.skip_audit:
        report = audit_yolo_dataset(args.data, compute_phash=True, max_phash_images=3000)
        write_audit_report(report, "dataset_report.json", "dataset_report.txt")
        summarize_audit(report)
        args.data = make_resolved_data_yaml(args.data, report)

    model_path = resolve_model_path(args.model)
    logging.info("Load YOLO model: %s", model_path)
    model = YOLO(model_path)

    if args.freeze_epochs > 0 and args.freeze_layers > 0:
        warmup_epochs = min(args.freeze_epochs, args.epochs)
        logging.info(
            "Warmup freeze %d layer trong %d epoch.",
            args.freeze_layers,
            warmup_epochs,
        )
        warmup_kwargs = build_train_kwargs(
            args,
            name=f"{args.name}_warmup",
            epochs=warmup_epochs,
            freeze_layers=args.freeze_layers,
        )
        warmup_results = model.train(**warmup_kwargs)
        warmup_save_dir = Path(getattr(warmup_results, "save_dir", Path(args.project) / f"{args.name}_warmup"))
        warmup_last = warmup_save_dir / "weights" / "last.pt"
        if warmup_last.exists() and warmup_epochs < args.epochs:
            model = YOLO(str(warmup_last))
        remaining_epochs = args.epochs - warmup_epochs
    else:
        remaining_epochs = args.epochs

    if remaining_epochs > 0:
        train_kwargs = build_train_kwargs(
            args,
            name=args.name,
            epochs=remaining_epochs,
            freeze_layers=0,
        )
        results = model.train(**train_kwargs)
        save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    else:
        save_dir = Path(args.project) / f"{args.name}_warmup"

    copy_training_artifacts(save_dir)

    best_path = Path("weights/yolo_drowsy_best.pt")
    if best_path.exists() and not args.skip_test_val:
        run_validation(best_path, args, report)

    logging.info("Hoan tat train YOLO. Neu dataset nho, hay uu tien yolo11n truoc; chi doi yolo11s khi du lieu da du da dang.")


if __name__ == "__main__":
    train_yolo()
