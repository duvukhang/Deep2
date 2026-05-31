import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

try:
    import cv2
except ImportError:
    cv2 = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}
SCENARIO_KEYWORDS = {
    "glasses": ["glasses", "sunglasses", "kinh"],
    "mask": ["mask", "khau_trang", "khẩu_trang"],
    "low_light": ["lowlight", "low_light", "night", "dark", "toi", "thiếu_sáng"],
    "side_pose": ["side", "left", "right", "yaw", "nghieng"],
    "head_down": ["head_down", "nodding", "nod", "down", "cui"],
    "leaning": ["lean", "chom", "leaning"],
    "yawning": ["yawn", "yawning", "ngap"],
}


def load_yolo_config(data_yaml):
    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Khong tim thay file data yaml: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    raw_base_path = Path(config.get("path", data_yaml.parent)).expanduser()
    if raw_base_path.is_absolute():
        base_path = raw_base_path.resolve()
    else:
        candidates = [
            (data_yaml.parent / raw_base_path).resolve(),
            (Path.cwd() / raw_base_path).resolve(),
            (data_yaml.parent.parent / raw_base_path).resolve(),
        ]
        base_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])

    names = config.get("names", [])
    if isinstance(names, dict):
        names = [names[idx] for idx in sorted(names)]

    splits = {}
    for key in ["train", "val", "valid", "test"]:
        if key not in config:
            continue
        split_name = SPLIT_ALIASES[key]
        split_path = Path(config[key])
        if not split_path.is_absolute():
            split_path = base_path / split_path
        splits[split_name] = split_path.resolve()

    return {
        "yaml_path": str(data_yaml.resolve()),
        "base_path": str(base_path.resolve()),
        "names": names,
        "nc": int(config.get("nc", len(names))),
        "splits": splits,
    }


def list_images(image_dir):
    image_dir = Path(image_dir)
    if not image_dir.exists():
        return []
    return sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def label_dir_from_image_dir(image_dir):
    image_dir = Path(image_dir)
    parts = list(image_dir.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx].lower() == "images":
            parts[idx] = "labels"
            return Path(*parts)
    return image_dir.parent / "labels"


def image_to_label_path(image_path, image_dir=None, label_dir=None):
    image_path = Path(image_path)
    if label_dir is None:
        if image_dir is None:
            label_dir = label_dir_from_image_dir(image_path.parent)
        else:
            label_dir = label_dir_from_image_dir(image_dir)
    return Path(label_dir) / f"{image_path.stem}.txt"


def parse_label_file(label_path, num_classes):
    label_path = Path(label_path)
    result = {
        "exists": label_path.exists(),
        "empty": False,
        "invalid_lines": [],
        "instances": [],
    }

    if not label_path.exists():
        return result

    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        result["empty"] = True
        return result

    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.strip().split()
        if len(parts) < 5:
            result["invalid_lines"].append({"line": line_no, "reason": "Khong du 5 cot"})
            continue
        try:
            class_id = int(float(parts[0]))
            bbox = [float(value) for value in parts[1:5]]
        except ValueError:
            result["invalid_lines"].append({"line": line_no, "reason": "Gia tri khong hop le"})
            continue

        if class_id < 0 or class_id >= num_classes:
            result["invalid_lines"].append({"line": line_no, "reason": "Class id ngoai pham vi"})
            continue

        if any(value < 0.0 or value > 1.0 for value in bbox):
            result["invalid_lines"].append({"line": line_no, "reason": "BBox khong nam trong [0, 1]"})
            continue

        if bbox[2] <= 0 or bbox[3] <= 0:
            result["invalid_lines"].append({"line": line_no, "reason": "BBox co kich thuoc <= 0"})
            continue

        result["instances"].append({"class_id": class_id, "bbox": bbox})

    return result


def file_sha1(path, block_size=1024 * 1024):
    sha1 = hashlib.sha1()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            sha1.update(block)
    return sha1.hexdigest()


def perceptual_hash(path):
    if cv2 is None:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    image = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
    mean_value = float(np.mean(image))
    bits = image > mean_value
    return "".join("1" if bit else "0" for bit in bits.flatten())


def roboflow_origin_key(path):
    stem = Path(path).stem
    if ".rf." in stem:
        return stem.split(".rf.", 1)[0]
    return stem


def detect_scenarios(path):
    text = str(path).lower()
    found = []
    for scenario, keywords in SCENARIO_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            found.append(scenario)
    return found


def imbalance_warnings(class_counts, names):
    warnings = []
    non_zero = {key: value for key, value in class_counts.items() if value > 0}
    if len(non_zero) <= 1:
        warnings.append("Chi co 0-1 class co instance, dataset khong du de train can bang.")
        return warnings

    min_count = min(non_zero.values())
    max_count = max(non_zero.values())
    ratio = max_count / max(1, min_count)
    if ratio >= 5:
        warnings.append(f"Class imbalance rat nang: max/min = {ratio:.1f}.")
    elif ratio >= 3:
        warnings.append(f"Class imbalance dang ke: max/min = {ratio:.1f}.")

    for idx, name in enumerate(names):
        count = class_counts.get(idx, 0)
        if count == 0:
            warnings.append(f"Class '{name}' khong co instance.")
        elif count < 50:
            warnings.append(f"Class '{name}' chi co {count} instance, rat de overfit.")
    return warnings


def audit_yolo_dataset(data_yaml, compute_phash=True, max_phash_images=3000):
    config = load_yolo_config(data_yaml)
    names = config["names"]
    num_classes = config["nc"]

    report = {
        "data_yaml": config["yaml_path"],
        "base_path": config["base_path"],
        "class_names": names,
        "num_classes": num_classes,
        "splits": {},
        "warnings": [],
        "leakage": {
            "exact_hash": [],
            "roboflow_origin": [],
            "perceptual_hash": [],
        },
    }
    if cv2 is None:
        report["warnings"].append(
            "Chua cai opencv-python nen bo qua kiem tra anh loi va perceptual hash."
        )

    hashes_by_split = defaultdict(dict)
    origins_by_split = defaultdict(dict)
    phash_by_split = defaultdict(dict)
    total_phash_seen = 0

    for split, image_dir in config["splits"].items():
        label_dir = label_dir_from_image_dir(image_dir)
        images = list_images(image_dir)
        class_counts = Counter()
        scenario_counts = Counter()
        missing_labels = []
        empty_labels = []
        invalid_labels = []
        unreadable_images = []
        labels_without_image = []
        labels_seen = set()

        if not image_dir.exists():
            report["warnings"].append(f"Thieu thu muc anh cho split {split}: {image_dir}")
        if not label_dir.exists():
            report["warnings"].append(f"Thieu thu muc label cho split {split}: {label_dir}")

        for image_path in images:
            if cv2 is not None:
                image = cv2.imread(str(image_path))
                if image is None:
                    unreadable_images.append(str(image_path))

            for scenario in detect_scenarios(image_path):
                scenario_counts[scenario] += 1

            label_path = image_to_label_path(image_path, image_dir, label_dir)
            labels_seen.add(str(label_path.resolve()))
            parsed = parse_label_file(label_path, num_classes)

            if not parsed["exists"]:
                missing_labels.append(str(image_path))
            elif parsed["empty"]:
                empty_labels.append(str(label_path))

            if parsed["invalid_lines"]:
                invalid_labels.append({
                    "label": str(label_path),
                    "errors": parsed["invalid_lines"],
                })

            for item in parsed["instances"]:
                class_counts[item["class_id"]] += 1

            try:
                hashes_by_split[split][file_sha1(image_path)] = str(image_path)
            except OSError:
                pass

            origin = roboflow_origin_key(image_path)
            origins_by_split[split].setdefault(origin, []).append(str(image_path))

            if compute_phash and total_phash_seen < max_phash_images:
                phash = perceptual_hash(image_path)
                total_phash_seen += 1
                if phash:
                    phash_by_split[split].setdefault(phash, []).append(str(image_path))

        if label_dir.exists():
            image_stems = {image.stem for image in images}
            for label_path in sorted(label_dir.glob("*.txt")):
                if label_path.stem not in image_stems:
                    labels_without_image.append(str(label_path))

        split_report = {
            "image_dir": str(image_dir),
            "label_dir": str(label_dir),
            "num_images": len(images),
            "num_labels": len(list(label_dir.glob("*.txt"))) if label_dir.exists() else 0,
            "missing_labels": missing_labels[:100],
            "missing_labels_count": len(missing_labels),
            "empty_labels": empty_labels[:100],
            "empty_labels_count": len(empty_labels),
            "invalid_labels": invalid_labels[:100],
            "invalid_labels_count": len(invalid_labels),
            "labels_without_image": labels_without_image[:100],
            "labels_without_image_count": len(labels_without_image),
            "unreadable_images": unreadable_images[:100],
            "unreadable_images_count": len(unreadable_images),
            "class_counts": {names[idx] if idx < len(names) else str(idx): int(class_counts[idx]) for idx in range(num_classes)},
            "class_counts_by_id": {str(idx): int(class_counts[idx]) for idx in range(num_classes)},
            "scenario_counts": dict(scenario_counts),
        }
        split_report["warnings"] = imbalance_warnings(class_counts, names)
        report["splits"][split] = split_report
        report["warnings"].extend([f"{split}: {warning}" for warning in split_report["warnings"]])

    split_names = list(report["splits"].keys())
    for i, split_a in enumerate(split_names):
        for split_b in split_names[i + 1:]:
            exact_overlap = set(hashes_by_split[split_a]) & set(hashes_by_split[split_b])
            if exact_overlap:
                report["leakage"]["exact_hash"].append({
                    "splits": [split_a, split_b],
                    "count": len(exact_overlap),
                    "examples": [
                        [hashes_by_split[split_a][hash_value], hashes_by_split[split_b][hash_value]]
                        for hash_value in list(exact_overlap)[:20]
                    ],
                })

            origin_overlap = set(origins_by_split[split_a]) & set(origins_by_split[split_b])
            origin_overlap = {
                key for key in origin_overlap
                if key and len(origins_by_split[split_a][key]) and len(origins_by_split[split_b][key])
            }
            if origin_overlap:
                report["leakage"]["roboflow_origin"].append({
                    "splits": [split_a, split_b],
                    "count": len(origin_overlap),
                    "examples": [
                        [origins_by_split[split_a][key][0], origins_by_split[split_b][key][0]]
                        for key in list(origin_overlap)[:20]
                    ],
                })

            phash_overlap = set(phash_by_split[split_a]) & set(phash_by_split[split_b])
            if phash_overlap:
                report["leakage"]["perceptual_hash"].append({
                    "splits": [split_a, split_b],
                    "count": len(phash_overlap),
                    "examples": [
                        [phash_by_split[split_a][key][0], phash_by_split[split_b][key][0]]
                        for key in list(phash_overlap)[:20]
                    ],
                })

    if report["leakage"]["exact_hash"]:
        report["warnings"].append("Phat hien anh trung hash giua cac split.")
    if report["leakage"]["roboflow_origin"]:
        report["warnings"].append("Phat hien anh cung origin Roboflow giua train/valid/test, co nguy co data leakage.")
    if report["leakage"]["perceptual_hash"]:
        report["warnings"].append("Phat hien anh gan trung nhau giua cac split theo perceptual hash.")

    for split, split_report in report["splits"].items():
        scenarios = split_report["scenario_counts"]
        for scenario in ["glasses", "mask", "low_light", "side_pose", "head_down", "leaning"]:
            if scenarios.get(scenario, 0) < 20:
                report["warnings"].append(
                    f"{split}: It mau du lieu '{scenario}' trong ten file; can bo sung neu muon robust thuc te."
                )

    return report


def write_audit_report(report, json_path="dataset_report.json", text_path="dataset_report.txt"):
    json_path = Path(json_path)
    text_path = Path(text_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "DATASET AUDIT REPORT",
        f"Data yaml: {report['data_yaml']}",
        f"Base path: {report['base_path']}",
        f"Classes: {', '.join(report['class_names'])}",
        "",
    ]

    for split, item in report["splits"].items():
        lines.extend([
            f"[{split}]",
            f"Images: {item['num_images']}",
            f"Labels: {item['num_labels']}",
            f"Missing labels: {item['missing_labels_count']}",
            f"Empty labels: {item['empty_labels_count']}",
            f"Invalid labels: {item['invalid_labels_count']}",
            f"Unreadable images: {item['unreadable_images_count']}",
            f"Labels without image: {item['labels_without_image_count']}",
            f"Class counts: {item['class_counts']}",
            f"Scenario keyword counts: {item['scenario_counts']}",
            "",
        ])

    lines.append("[Leakage]")
    for key, values in report["leakage"].items():
        total = sum(item["count"] for item in values)
        lines.append(f"{key}: {total}")

    lines.extend(["", "[Warnings]"])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- Khong co canh bao lon.")

    text_path.write_text("\n".join(lines), encoding="utf-8")
