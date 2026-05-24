# services/drowsiness_detector.py

import time
import math
import numpy as np
from config import DrowsinessConfig


class DrowsinessDetector:
    def __init__(self):
        self.ear_buffer = []
        self.base_ear = None

        self.eye_closed_start = None
        self.yawn_start = None

        self.drowsy_score = 0.0
        self.last_alert = 0.0

        self.prev_face_center_y = None
        self.prev_face_area = None
        self.leaning_start = None
        self.bad_camera_start = None

        self.head_down_start = None
        self.no_eye_yawn_start = None
        self.no_eye_start = None

    def distance(self, p1, p2):
        return math.dist(p1, p2)

    def point(self, landmarks, idx, w, h):
        return int(landmarks[idx].x * w), int(landmarks[idx].y * h)

    def calculate_eye_ear(self, landmarks, eye_indices, w, h):
        try:
            pts = [self.point(landmarks, idx, w, h) for idx in eye_indices]

            A = self.distance(pts[1], pts[5])
            B = self.distance(pts[2], pts[4])
            C = self.distance(pts[0], pts[3])

            if C <= 1:
                return None

            ear = (A + B) / (2.0 * C)

            if ear <= 0 or ear > 0.65:
                return None

            return ear
        except Exception:
            return None

    def calculate_ear(self, landmarks, w, h):
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        left_ear = self.calculate_eye_ear(landmarks, left_eye, w, h)
        right_ear = self.calculate_eye_ear(landmarks, right_eye, w, h)

        valid_ears = [ear for ear in [left_ear, right_ear] if ear is not None]

        if len(valid_ears) == 0:
            return 0.0, 0, False

        avg_ear = float(np.mean(valid_ears))

        self.ear_buffer.append(avg_ear)

        if len(self.ear_buffer) > 10:
            self.ear_buffer.pop(0)

        smooth_ear = float(np.mean(self.ear_buffer))

        if self.base_ear is None and len(self.ear_buffer) >= 10:
            self.base_ear = smooth_ear

        return smooth_ear, len(valid_ears), True

    def calculate_mar(self, landmarks, w, h):
        try:
            upper_lip = self.point(landmarks, 13, w, h)
            lower_lip = self.point(landmarks, 14, w, h)
            left_mouth = self.point(landmarks, 78, w, h)
            right_mouth = self.point(landmarks, 308, w, h)

            vertical = self.distance(upper_lip, lower_lip)
            horizontal = self.distance(left_mouth, right_mouth)

            if horizontal <= 1:
                return 0.0

            return vertical / horizontal
        except Exception:
            return 0.0

    def calculate_head_pose_score(self, landmarks, w, h):
        try:
            nose = self.point(landmarks, 1, w, h)
            chin = self.point(landmarks, 152, w, h)
            left_face = self.point(landmarks, 234, w, h)
            right_face = self.point(landmarks, 454, w, h)

            vertical = abs(chin[1] - nose[1])
            horizontal = abs(right_face[0] - left_face[0])

            if horizontal <= 1:
                return 0.0

            return vertical / horizontal
        except Exception:
            return 0.0

    def get_face_center_and_area(self, face_box):
        x1, y1, x2, y2 = face_box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        area = max(1, (x2 - x1) * (y2 - y1))
        return cx, cy, area

    def estimate_confidence(self, face_box, frame_w, frame_h, visible_eye_count):
        if face_box is None:
            return 0.0

        x1, y1, x2, y2 = face_box

        face_area = max(1, (x2 - x1) * (y2 - y1))
        frame_area = frame_w * frame_h

        area_ratio = face_area / frame_area
        area_score = min(1.0, area_ratio * 8)

        eye_score = visible_eye_count / 2.0

        # Nếu không thấy mắt nhưng mặt vẫn rõ, cho phép vào chế độ kính râm
        if visible_eye_count == 0:
            eye_score = 0.15

        confidence = area_score * 0.55 + eye_score * 0.45
        return round(float(confidence), 2)

    def detect_leaning_forward(self, face_box):
        if face_box is None:
            return False

        _, cy, area = self.get_face_center_and_area(face_box)

        if self.prev_face_center_y is None or self.prev_face_area is None:
            self.prev_face_center_y = cy
            self.prev_face_area = area
            return False

        y_change = abs(cy - self.prev_face_center_y)
        area_change_ratio = abs(area - self.prev_face_area) / max(1, self.prev_face_area)

        self.prev_face_center_y = cy
        self.prev_face_area = area

        is_leaning = y_change > 45 or area_change_ratio > 0.28

        now = time.time()

        if is_leaning:
            if self.leaning_start is None:
                self.leaning_start = now
        else:
            self.leaning_start = None

        return is_leaning

    def is_in_lean_grace(self):
        if self.leaning_start is None:
            return False

        return time.time() - self.leaning_start < DrowsinessConfig.LEAN_GRACE_SECONDS

    def is_bad_camera_grace(self, confidence):
        now = time.time()

        if confidence < 0.35:
            if self.bad_camera_start is None:
                self.bad_camera_start = now

            return now - self.bad_camera_start < DrowsinessConfig.BAD_CAMERA_GRACE_SECONDS

        self.bad_camera_start = None
        return False

    def update_no_eye_mode(self, eye_visible, mar, head_pose_value, confidence, face_count):
        now = time.time()

        if face_count == 0:
            self.head_down_start = None
            self.no_eye_yawn_start = None
            self.no_eye_start = None

            return {
                "mode": "NO_FACE_MODE",
                "head_down_duration": 0.0,
                "fallback_yawn_duration": 0.0,
                "fallback_score": 0.0
            }

        if eye_visible:
            self.head_down_start = None
            self.no_eye_yawn_start = None
            self.no_eye_start = None

            return {
                "mode": "EYE_MODE",
                "head_down_duration": 0.0,
                "fallback_yawn_duration": 0.0,
                "fallback_score": 0.0
            }

        if confidence < DrowsinessConfig.NO_EYE_CONFIDENCE_MIN:
            self.head_down_start = None
            self.no_eye_yawn_start = None

            return {
                "mode": "CAMERA_BAD_MODE",
                "head_down_duration": 0.0,
                "fallback_yawn_duration": 0.0,
                "fallback_score": 0.0
            }

        if self.no_eye_start is None:
            self.no_eye_start = now

        if head_pose_value > DrowsinessConfig.HEAD_DOWN_THRESHOLD:
            if self.head_down_start is None:
                self.head_down_start = now
        else:
            self.head_down_start = None

        if mar > DrowsinessConfig.MAR_THRESHOLD:
            if self.no_eye_yawn_start is None:
                self.no_eye_yawn_start = now
        else:
            self.no_eye_yawn_start = None

        head_down_duration = 0.0
        fallback_yawn_duration = 0.0

        if self.head_down_start:
            head_down_duration = now - self.head_down_start

        if self.no_eye_yawn_start:
            fallback_yawn_duration = now - self.no_eye_yawn_start

        fallback_score = 0.0

        if head_down_duration > DrowsinessConfig.NO_EYE_HEAD_DOWN_SECONDS:
            fallback_score += 4.0

        if fallback_yawn_duration > DrowsinessConfig.NO_EYE_YAWN_SECONDS:
            fallback_score += 3.0

        if self.no_eye_start and now - self.no_eye_start > 3.0:
            fallback_score += 1.0

        return {
            "mode": "NO_EYE_MODE",
            "head_down_duration": round(head_down_duration, 2),
            "fallback_yawn_duration": round(fallback_yawn_duration, 2),
            "fallback_score": round(fallback_score, 2)
        }

    def update(self, face_count, selected_face_box, landmarks, frame_w, frame_h):
        now = time.time()

        ear = 0.0
        mar = 0.0
        head_pose = 0.0
        confidence = 0.0
        visible_eye_count = 0
        eye_visible = False
        eye_closed_duration = 0.0
        yawn_duration = 0.0
        is_leaning = False

        if selected_face_box is not None and landmarks is not None:
            ear, visible_eye_count, eye_visible = self.calculate_ear(
                landmarks,
                frame_w,
                frame_h
            )

            mar = self.calculate_mar(landmarks, frame_w, frame_h)
            head_pose = self.calculate_head_pose_score(landmarks, frame_w, frame_h)
            confidence = self.estimate_confidence(
                selected_face_box,
                frame_w,
                frame_h,
                visible_eye_count
            )

            is_leaning = self.detect_leaning_forward(selected_face_box)

        if face_count == 0 or selected_face_box is None:
            self.drowsy_score = max(0.0, self.drowsy_score - 0.3)
            self.eye_closed_start = None
            self.yawn_start = None

        threshold = (
            self.base_ear * 0.76
            if self.base_ear
            else DrowsinessConfig.EAR_DEFAULT_THRESHOLD
        )

        if eye_visible and ear > 0:
            if ear < threshold:
                if self.eye_closed_start is None:
                    self.eye_closed_start = now
            else:
                self.eye_closed_start = None

        if self.eye_closed_start:
            eye_closed_duration = now - self.eye_closed_start

        if mar > DrowsinessConfig.MAR_THRESHOLD:
            if self.yawn_start is None:
                self.yawn_start = now
        else:
            self.yawn_start = None

        if self.yawn_start:
            yawn_duration = now - self.yawn_start

        fallback = self.update_no_eye_mode(
            eye_visible=eye_visible,
            mar=mar,
            head_pose_value=head_pose,
            confidence=confidence,
            face_count=face_count
        )

        detection_mode = fallback["mode"]
        fallback_score = fallback["fallback_score"]
        head_down_duration = fallback["head_down_duration"]
        fallback_yawn_duration = fallback["fallback_yawn_duration"]

        bad_camera = detection_mode == "CAMERA_BAD_MODE" and face_count > 0
        bad_camera_grace = self.is_bad_camera_grace(confidence)
        leaning_grace = self.is_in_lean_grace()

        if not leaning_grace and not bad_camera_grace:
            if detection_mode == "EYE_MODE":
                if eye_closed_duration > DrowsinessConfig.EYE_CLOSED_SECONDS:
                    self.drowsy_score += 1.15

                if yawn_duration > DrowsinessConfig.YAWN_SECONDS:
                    self.drowsy_score += 0.65

                if head_pose > DrowsinessConfig.HEAD_DOWN_THRESHOLD:
                    self.drowsy_score += 0.75

            elif detection_mode == "NO_EYE_MODE":
                if fallback_score >= 4.0:
                    self.drowsy_score += 0.85

                if fallback_score >= 7.0:
                    self.drowsy_score += 1.2

            else:
                self.drowsy_score = max(0.0, self.drowsy_score - 0.35)

        else:
            self.drowsy_score = max(0.0, self.drowsy_score - 0.45)

        if detection_mode == "EYE_MODE":
            if (
                eye_closed_duration < 0.7
                and yawn_duration < 0.8
                and head_pose < 0.68
                and confidence >= 0.35
            ):
                self.drowsy_score -= 0.35

        elif detection_mode == "NO_EYE_MODE":
            if (
                fallback_score < 3.0
                and mar < DrowsinessConfig.MAR_THRESHOLD
                and head_pose < DrowsinessConfig.HEAD_DOWN_THRESHOLD
                and confidence >= DrowsinessConfig.NO_EYE_CONFIDENCE_MIN
            ):
                self.drowsy_score -= 0.3

        elif detection_mode == "NO_FACE_MODE":
            self.drowsy_score -= 0.25

        self.drowsy_score = max(0.0, min(self.drowsy_score, 10.0))

        if face_count == 0:
            state = "NO_FACE"
            is_drowsy = False

        elif selected_face_box is None:
            state = "NO_DRIVER_IN_ROI"
            is_drowsy = False

        elif bad_camera and not bad_camera_grace:
            state = "CAMERA_BAD"
            is_drowsy = False

        elif (
            detection_mode == "NO_EYE_MODE"
            and self.drowsy_score >= DrowsinessConfig.NO_EYE_DROWSY_SCORE_LIMIT
        ):
            state = "DROWSY_CONFIRMED"
            is_drowsy = True

        elif (
            detection_mode == "EYE_MODE"
            and self.drowsy_score >= DrowsinessConfig.NORMAL_DROWSY_LIMIT
        ):
            state = "DROWSY_CONFIRMED"
            is_drowsy = True

        elif self.drowsy_score >= DrowsinessConfig.WARNING_LIMIT:
            if detection_mode == "NO_EYE_MODE":
                state = "WARNING_SUNGLASSES_MODE"
            else:
                state = "WARNING_LEVEL_1"

            is_drowsy = False

        else:
            if detection_mode == "NO_EYE_MODE":
                state = "SUNGLASSES_MODE"
            else:
                state = "NORMAL"

            is_drowsy = False

        return {
            "ear": round(ear, 3),
            "mar": round(mar, 3),
            "head_pose": round(head_pose, 3),
            "confidence": confidence,
            "visible_eye_count": visible_eye_count,
            "eye_visible": eye_visible,
            "is_leaning": is_leaning,
            "threshold": round(threshold, 3),
            "drowsy_score": round(self.drowsy_score, 2),
            "state": state,
            "is_drowsy": is_drowsy,
            "detection_mode": detection_mode,
            "fallback_score": fallback_score,
            "head_down_duration": head_down_duration,
            "fallback_yawn_duration": fallback_yawn_duration,
            "can_alert": now - self.last_alert > DrowsinessConfig.ALERT_COOLDOWN
        }

    def mark_alerted(self):
        self.last_alert = time.time()