import argparse
import logging
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from ultralytics import YOLO

from models.cnn_feature_extractor import CNNFeatureExtractor
from utils.dataset_audit import (
    image_to_label_path,
    label_dir_from_image_dir,
    list_images,
    load_yolo_config,
    parse_label_file,
)


DEFAULT_CLASS_NAMES = ["eye_closed", "eye_open", "face", "mouth_open"]
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class GeometryExtractor:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

    @staticmethod
    def point(landmarks, idx, w, h):
        return int(landmarks[idx].x * w), int(landmarks[idx].y * h)

    @staticmethod
    def distance(p1, p2):
        return math.dist(p1, p2)

    def eye_ear(self, landmarks, indices, w, h):
        try:
            pts = [self.point(landmarks, idx, w, h) for idx in indices]
            a = self.distance(pts[1], pts[5])
            b = self.distance(pts[2], pts[4])
            c = self.distance(pts[0], pts[3])
            if c <= 1:
                return None
            ear = (a + b) / (2.0 * c)
            if ear <= 0 or ear > 0.75:
                return None
            return float(ear)
        except Exception:
            return None

    def ear(self, landmarks, w, h):
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]
        values = [
            self.eye_ear(landmarks, left_eye, w, h),
            self.eye_ear(landmarks, right_eye, w, h),
        ]
        valid = [value for value in values if value is not None]
        if not valid:
            return 0.0, 0
        return float(np.mean(valid)), len(valid)

    def mar(self, landmarks, w, h):
        try:
            upper_lip = self.point(landmarks, 13, w, h)
            lower_lip = self.point(landmarks, 14, w, h)
            left_mouth = self.point(landmarks, 78, w, h)
            right_mouth = self.point(landmarks, 308, w, h)
            vertical = self.distance(upper_lip, lower_lip)
            horizontal = self.distance(left_mouth, right_mouth)
            if horizontal <= 1:
                return 0.0
            return float(vertical / horizontal)
        except Exception:
            return 0.0

    @staticmethod
    def normalize_angle(angle):
        angle = ((float(angle) + 180.0) % 360.0) - 180.0
        if angle > 90.0:
            angle = 180.0 - angle
        elif angle < -90.0:
            angle = -180.0 - angle
        return angle

    def head_pose(self, landmarks, w, h):
        try:
            image_points = np.array([
                self.point(landmarks, 1, w, h),
                self.point(landmarks, 152, w, h),
                self.point(landmarks, 33, w, h),
                self.point(landmarks, 263, w, h),
                self.point(landmarks, 61, w, h),
                self.point(landmarks, 291, w, h),
            ], dtype=np.float64)
            model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (-28.9, -28.9, -24.1),
                (28.9, -28.9, -24.1),
            ], dtype=np.float64)
            focal_length = float(w)
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success:
                return 0.0, 0.0, 0.0
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            projection_matrix = np.hstack((rotation_matrix, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)
            pitch, yaw, roll = euler_angles.flatten()
            return (
                self.normalize_angle(pitch),
                self.normalize_angle(yaw),
                self.normalize_angle(roll),
            )
        except Exception:
            return 0.0, 0.0, 0.0

    def extract(self, frame_bgr, yolo_confidence=0.0):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            confidence = float(np.clip(yolo_confidence * 0.35, 0.0, 1.0))
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, confidence], dtype=np.float32)

        landmarks = results.multi_face_landmarks[0].landmark
        ear, visible_eye_count = self.ear(landmarks, w, h)
        mar = self.mar(landmarks, w, h)
        pitch, yaw, roll = self.head_pose(landmarks, w, h)

        eye_score = visible_eye_count / 2.0
        confidence = 0.50 + 0.30 * float(np.clip(yolo_confidence, 0.0, 1.0)) + 0.20 * eye_score
        confidence = float(np.clip(confidence, 0.0, 1.0))
        return np.array([ear, mar, pitch, yaw, roll, confidence], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract spatial/geometric sequences cho Hybrid.")
    parser.add_argument("--source", default="driver_drowsiness_Computer_Vision_Model", help="Dataset root, data.yaml, video, hoac thu muc anh.")
    parser.add_argument("--yolo", default="weights/yolo_drowsy_best.pt")
    parser.add_argument("--cnn", default="weights/cnn_best.pth")
    parser.add_argument("--output", default="dataset_extracted")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--cnn_backbone", default="mobilenet_v3_small")
    parser.add_argument("--cnn_feature_dim", type=int, default=512)
    parser.add_argument("--frame_stride", type=int, default=3)
    parser.add_argument("--split", default="train", choices=["train", "val", "valid", "test"], help="Split dung khi source la video/thu muc anh thuong.")
    parser.add_argument("--label", default="auto", choices=["auto", "0", "1"], help="Label sequence khi source khong co nhan.")
    parser.add_argument("--pad_last", action="store_true", help="Pad sequence cuoi bang frame cuoi neu chua du seq_len.")
    return parser.parse_args()


def configure_logging():
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/extract_features.log", encoding="utf-8"),
        ],
    )


def select_device(device_arg):
    if device_arg:
        text = str(device_arg).lower()
        if text.isdigit():
            return torch.device(f"cuda:{text}" if torch.cuda.is_available() else "cpu")
        return torch.device(text)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def yolo_device_arg(device):
    if device.type == "cpu":
        return "cpu"
    return 0 if device.index is None else device.index


def build_transform(image_size):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.12)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_cnn_model(path, args, device):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Khong tim thay CNN weights: {path}. Hay train: python train_cnn.py ..."
        )
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        backbone = checkpoint.get("backbone", args.cnn_backbone)
        feature_dim = int(checkpoint.get("feature_dim", args.cnn_feature_dim))
        num_classes = int(checkpoint.get("num_classes", 4))
        image_size = int(checkpoint.get("image_size", args.image_size))
    else:
        state_dict = checkpoint
        backbone = args.cnn_backbone
        feature_dim = args.cnn_feature_dim
        num_classes = 4
        image_size = args.image_size

    model = CNNFeatureExtractor(
        backbone=backbone,
        feature_dim=feature_dim,
        num_classes=num_classes,
        pretrained=False,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, image_size


def xyxy_from_detection(box):
    return [int(value) for value in box]


def crop_frame(frame_bgr, xyxy):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return frame_bgr
    return frame_bgr[y1:y2, x1:x2]


def parse_yolo_detections(result, names):
    detections = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        cls_id = int(box.cls.detach().cpu().item())
        conf = float(box.conf.detach().cpu().item())
        xyxy = xyxy_from_detection(box.xyxy.detach().cpu().numpy()[0])
        class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
        detections.append({
            "class_id": cls_id,
            "class_name": str(class_name),
            "confidence": conf,
            "xyxy": xyxy,
            "area": max(1, (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])),
        })
    return detections


def choose_face_box(detections):
    faces = [item for item in detections if item["class_name"] == "face"]
    candidates = faces if faces else detections
    if not candidates:
        return None, 0.0
    best = max(candidates, key=lambda item: item["area"] * item["confidence"])
    return best["xyxy"], best["confidence"]


@torch.no_grad()
def extract_spatial(frame_bgr, detections, cnn_model, transform, device):
    priority = {"face": 0, "eye_closed": 1, "eye_open": 1, "mouth_open": 2}
    selected = [
        det for det in detections
        if det["class_name"] in priority and det["confidence"] >= 0.05
    ]
    selected = sorted(selected, key=lambda det: (priority.get(det["class_name"], 9), -det["confidence"]))[:6]

    if not selected:
        selected = [{"xyxy": [0, 0, frame_bgr.shape[1], frame_bgr.shape[0]], "confidence": 0.25}]

    tensors = []
    weights = []
    for det in selected:
        crop = crop_frame(frame_bgr, det["xyxy"])
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        tensors.append(transform(image))
        weights.append(max(0.05, float(det.get("confidence", 0.25))))

    batch = torch.stack(tensors).to(device)
    features = cnn_model.extract_features(batch).detach().cpu().numpy()
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / max(float(weights.sum()), 1e-6)
    return np.sum(features * weights[:, None], axis=0).astype(np.float32)


def infer_label_from_name(path):
    text = Path(path).stem.lower()
    if "notdrowsy" in text or "non_sleepy" in text or "nonsleepy" in text or "normal" in text:
        return 0
    if "drowsy" in text or "sleepy" in text or "yawn" in text or "nodding" in text:
        return 1
    return None


def infer_label_from_yolo_label(image_path, image_dir, class_names):
    label_path = image_to_label_path(image_path, image_dir, label_dir_from_image_dir(image_dir))
    parsed = parse_label_file(label_path, len(class_names))
    drowsy_classes = {"eye_closed", "mouth_open"}
    for instance in parsed["instances"]:
        class_id = int(instance["class_id"])
        if class_id < len(class_names) and class_names[class_id] in drowsy_classes:
            return 1
    if parsed["instances"]:
        return 0
    return None


def resolve_label(path, forced_label, image_dir=None, class_names=None):
    if forced_label in {"0", "1"}:
        return int(forced_label)

    by_name = infer_label_from_name(path)
    if by_name is not None:
        return by_name

    if image_dir is not None and class_names is not None:
        by_yolo = infer_label_from_yolo_label(path, image_dir, class_names)
        if by_yolo is not None:
            return by_yolo

    raise ValueError(
        f"Khong suy ra duoc label cho {path}. "
        "Hay dat ten file co drowsy/notdrowsy hoac truyen --label 0/1."
    )


def save_sequence(output_dir, split, index, spatial_items, geo_items, labels, metas):
    split_dir = Path(output_dir) / ("val" if split == "valid" else split)
    split_dir.mkdir(parents=True, exist_ok=True)
    label = int(round(float(np.mean(labels))))
    item = {
        "spatial": np.asarray(spatial_items, dtype=np.float32),
        "geo": np.asarray(geo_items, dtype=np.float32),
        "label": label,
        "meta": {
            "source_frames": metas,
            "label_votes": [int(value) for value in labels],
        },
    }
    path = split_dir / f"seq_{index:06d}.npy"
    np.save(path, item)
    return path


def process_frame(frame_bgr, yolo_model, cnn_model, transform, geometry, args, device):
    result = yolo_model.predict(
        frame_bgr,
        imgsz=args.imgsz,
        conf=args.conf,
        device=yolo_device_arg(device),
        verbose=False,
    )[0]
    detections = parse_yolo_detections(result, yolo_model.names)
    _, yolo_confidence = choose_face_box(detections)
    spatial = extract_spatial(frame_bgr, detections, cnn_model, transform, device)
    geo = geometry.extract(frame_bgr, yolo_confidence=yolo_confidence)
    return spatial, geo, detections


def dataset_split_dirs(source):
    source = Path(source)
    if source.suffix.lower() in {".yaml", ".yml"}:
        config = load_yolo_config(source)
        return config["splits"], config["names"]

    yaml_path = source / "data.yaml"
    if yaml_path.exists():
        config = load_yolo_config(yaml_path)
        return config["splits"], config["names"]

    splits = {}
    for split in ["train", "valid", "test"]:
        image_dir = source / split / "images"
        if image_dir.exists():
            splits["val" if split == "valid" else split] = image_dir
    if splits:
        return splits, DEFAULT_CLASS_NAMES

    return {None: source}, DEFAULT_CLASS_NAMES


def flush_sequence(output, split, seq_index, spatial_buffer, geo_buffer, label_buffer, meta_buffer, seq_len, pad_last):
    if len(spatial_buffer) < seq_len:
        if not pad_last or not spatial_buffer:
            return seq_index, 0
        while len(spatial_buffer) < seq_len:
            spatial_buffer.append(spatial_buffer[-1])
            geo_buffer.append(geo_buffer[-1])
            label_buffer.append(label_buffer[-1])
            meta_buffer.append({**meta_buffer[-1], "padded": True})

    path = save_sequence(
        output,
        split,
        seq_index,
        spatial_buffer[:seq_len],
        geo_buffer[:seq_len],
        label_buffer[:seq_len],
        meta_buffer[:seq_len],
    )
    logging.info("Da luu %s", path)
    del spatial_buffer[:seq_len]
    del geo_buffer[:seq_len]
    del label_buffer[:seq_len]
    del meta_buffer[:seq_len]
    return seq_index + 1, 1


def process_image_dataset(args, yolo_model, cnn_model, transform, geometry, device):
    split_dirs, class_names = dataset_split_dirs(args.source)
    total_saved = 0

    for split, image_dir in split_dirs.items():
        split_name = split or args.split
        image_paths = list_images(image_dir)
        if not image_paths:
            logging.warning("Khong co anh trong %s", image_dir)
            continue

        spatial_buffer = []
        geo_buffer = []
        label_buffer = []
        meta_buffer = []
        seq_index = 0

        for image_path in tqdm(image_paths, desc=f"extract {split_name}"):
            frame = cv2.imread(str(image_path))
            if frame is None:
                logging.warning("Bo qua anh loi: %s", image_path)
                continue

            label = resolve_label(image_path, args.label, image_dir, class_names)
            spatial, geo, detections = process_frame(
                frame,
                yolo_model,
                cnn_model,
                transform,
                geometry,
                args,
                device,
            )
            spatial_buffer.append(spatial)
            geo_buffer.append(geo)
            label_buffer.append(label)
            meta_buffer.append({
                "path": str(image_path),
                "detections": [
                    {
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "xyxy": det["xyxy"],
                    }
                    for det in detections
                ],
            })

            seq_index, saved = flush_sequence(
                args.output,
                split_name,
                seq_index,
                spatial_buffer,
                geo_buffer,
                label_buffer,
                meta_buffer,
                args.seq_len,
                pad_last=False,
            )
            total_saved += saved

        seq_index, saved = flush_sequence(
            args.output,
            split_name,
            seq_index,
            spatial_buffer,
            geo_buffer,
            label_buffer,
            meta_buffer,
            args.seq_len,
            pad_last=args.pad_last,
        )
        total_saved += saved

    return total_saved


def process_video(args, yolo_model, cnn_model, transform, geometry, device):
    source = Path(args.source)
    label = resolve_label(source, args.label)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Khong mo duoc video: {source}")

    spatial_buffer = []
    geo_buffer = []
    label_buffer = []
    meta_buffer = []
    frame_index = 0
    seq_index = 0
    saved_total = 0

    pbar = tqdm(desc=f"extract video {source.name}")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % args.frame_stride != 0:
            frame_index += 1
            continue

        spatial, geo, detections = process_frame(
            frame,
            yolo_model,
            cnn_model,
            transform,
            geometry,
            args,
            device,
        )
        spatial_buffer.append(spatial)
        geo_buffer.append(geo)
        label_buffer.append(label)
        meta_buffer.append({
            "video": str(source),
            "frame_index": frame_index,
            "detections": [
                {
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "xyxy": det["xyxy"],
                }
                for det in detections
            ],
        })
        seq_index, saved = flush_sequence(
            args.output,
            args.split,
            seq_index,
            spatial_buffer,
            geo_buffer,
            label_buffer,
            meta_buffer,
            args.seq_len,
            pad_last=False,
        )
        saved_total += saved
        frame_index += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    seq_index, saved = flush_sequence(
        args.output,
        args.split,
        seq_index,
        spatial_buffer,
        geo_buffer,
        label_buffer,
        meta_buffer,
        args.seq_len,
        pad_last=args.pad_last,
    )
    return saved_total + saved


def main():
    configure_logging()
    args = parse_args()
    device = select_device(args.device)

    if not Path(args.yolo).exists():
        raise FileNotFoundError(
            f"Khong tim thay YOLO weights: {args.yolo}. Hay train_yolo truoc."
        )

    yolo_model = YOLO(args.yolo)
    cnn_model, image_size = load_cnn_model(args.cnn, args, device)
    transform = build_transform(image_size)
    geometry = GeometryExtractor()

    source = Path(args.source)
    if source.is_file() and source.suffix.lower() in VIDEO_SUFFIXES:
        saved = process_video(args, yolo_model, cnn_model, transform, geometry, device)
    else:
        saved = process_image_dataset(args, yolo_model, cnn_model, transform, geometry, device)

    logging.info("Hoan tat extract. Tong sequence da luu: %d", saved)
    if saved == 0:
        logging.warning("Chua luu sequence nao. Kiem tra --seq_len, --pad_last, label va source.")


if __name__ == "__main__":
    main()
