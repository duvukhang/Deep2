# services/driver_selector.py

import threading
from config import DriverConfig


class DriverSelector:
    def __init__(self):
        self.lock = threading.Lock()

        self.config = {
            "side": DriverConfig.DEFAULT_SIDE,
            "lock_driver": DriverConfig.LOCK_DRIVER_DEFAULT,
            "custom_roi": None
        }

        self.last_driver_box = None
        self.current_driver_roi = None

    def get_config(self):
        with self.lock:
            return {
                "side": self.config["side"],
                "lock_driver": self.config["lock_driver"],
                "custom_roi": self.config["custom_roi"],
                "has_custom_roi": self.config["custom_roi"] is not None
            }

    def update_config(self, side=None, lock_driver=None):
        with self.lock:
            if side in ["left", "right"]:
                self.config["side"] = side
                self.config["custom_roi"] = None
                self.last_driver_box = None
                self.current_driver_roi = None

            if isinstance(lock_driver, bool):
                self.config["lock_driver"] = lock_driver

            return self.get_config()

    def lock_current_driver(self):
        with self.lock:
            if self.current_driver_roi is None:
                return {
                    "success": False,
                    "message": "Chưa nhận diện được tài xế để khóa vùng.",
                    **self.get_config()
                }

            self.config["custom_roi"] = self.current_driver_roi
            self.config["lock_driver"] = True

            return {
                "success": True,
                "message": "Đã khóa vùng tài xế hiện tại.",
                **self.get_config()
            }

    def unlock_driver_roi(self):
        with self.lock:
            self.config["custom_roi"] = None
            self.last_driver_box = None
            self.current_driver_roi = None

            return {
                "success": True,
                "message": "Đã mở khóa vùng tài xế.",
                **self.get_config()
            }

    def get_default_roi(self, side):
        if side == "left":
            return DriverConfig.DEFAULT_ROI_LEFT

        return DriverConfig.DEFAULT_ROI_RIGHT

    def get_active_roi(self):
        with self.lock:
            custom_roi = self.config["custom_roi"]
            side = self.config["side"]

        if custom_roi:
            return custom_roi

        return self.get_default_roi(side)

    def denormalize_roi(self, roi, w, h):
        return (
            int(roi["x1"] * w),
            int(roi["y1"] * h),
            int(roi["x2"] * w),
            int(roi["y2"] * h)
        )

    def get_face_box(self, landmarks, w, h):
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]

        x1 = int(max(0, min(xs) * w))
        y1 = int(max(0, min(ys) * h))
        x2 = int(min(w, max(xs) * w))
        y2 = int(min(h, max(ys) * h))

        return x1, y1, x2, y2

    def get_face_center_and_area(self, face_box):
        x1, y1, x2, y2 = face_box

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        area = max(1, (x2 - x1) * (y2 - y1))

        return cx, cy, area

    def box_center(self, face_box):
        x1, y1, x2, y2 = face_box
        return (x1 + x2) / 2, (y1 + y2) / 2

    def is_box_in_roi(self, face_box, roi, w, h):
        cx, cy = self.box_center(face_box)
        rx1, ry1, rx2, ry2 = self.denormalize_roi(roi, w, h)

        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    def box_iou(self, box_a, box_b):
        if box_a is None or box_b is None:
            return 0.0

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))

        return inter_area / float(area_a + area_b - inter_area)

    def create_roi_from_face_box(self, face_box, w, h):
        x1, y1, x2, y2 = face_box

        bw = x2 - x1
        bh = y2 - y1

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        new_w = bw * DriverConfig.LOCK_ROI_PADDING_X
        new_h = bh * DriverConfig.LOCK_ROI_PADDING_Y

        nx1 = max(0, cx - new_w / 2)
        ny1 = max(0, cy - new_h / 2)
        nx2 = min(w, cx + new_w / 2)
        ny2 = min(h, cy + new_h / 2)

        return {
            "x1": nx1 / w,
            "y1": ny1 / h,
            "x2": nx2 / w,
            "y2": ny2 / h
        }

    def select_driver_face(self, multi_face_landmarks, frame_w, frame_h):
        if not multi_face_landmarks:
            self.last_driver_box = None
            return None

        with self.lock:
            side = self.config["side"]
            lock_driver = self.config["lock_driver"]
            custom_roi = self.config["custom_roi"]

        roi = custom_roi if custom_roi else self.get_default_roi(side)

        candidates = []

        for face_landmarks in multi_face_landmarks:
            face_box = self.get_face_box(face_landmarks.landmark, frame_w, frame_h)
            cx, cy, area = self.get_face_center_and_area(face_box)

            area_ratio = area / (frame_w * frame_h)
            in_roi = self.is_box_in_roi(face_box, roi, frame_w, frame_h)

            if lock_driver and not in_roi:
                continue

            iou_score = self.box_iou(face_box, self.last_driver_box)

            if side == "right":
                side_score = cx / frame_w
            else:
                side_score = 1 - (cx / frame_w)

            y_score = cy / frame_h
            roi_score = 2.0 if in_roi else 0.0
            custom_roi_bonus = 1.0 if custom_roi and in_roi else 0.0

            if area_ratio < DriverConfig.SMALL_FACE_RATIO:
                small_face_penalty = -2.0
            elif area_ratio < DriverConfig.MEDIUM_FACE_RATIO:
                small_face_penalty = -0.8
            else:
                small_face_penalty = 0.0

            score = (
                area_ratio * 5.0
                + iou_score * 2.8
                + side_score * 0.8
                + y_score * 0.3
                + roi_score
                + custom_roi_bonus
                + small_face_penalty
            )

            candidates.append({
                "score": score,
                "face": face_landmarks,
                "box": face_box,
                "in_roi": in_roi,
                "area_ratio": area_ratio
            })

        if not candidates:
            self.last_driver_box = None
            return None

        candidates.sort(key=lambda item: item["score"], reverse=True)

        selected = candidates[0]

        self.last_driver_box = selected["box"]
        self.current_driver_roi = self.create_roi_from_face_box(
            selected["box"],
            frame_w,
            frame_h
        )

        return selected