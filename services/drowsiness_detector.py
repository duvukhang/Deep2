# services/drowsiness_detector.py

import math
import time

import cv2
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
        self.face_baseline_center = None
        self.face_baseline_area = None

        self.head_down_start = None
        self.no_eye_yawn_start = None
        self.no_eye_start = None
        self.pose_buffer = []
        self.pose_baseline = None
        self.mask_score = 0.0
        self.nod_phase = "stable"
        self.nod_down_started_at = None
        self.last_posture_event_at = 0.0
        self.last_posture_score_at = 0.0
        self.nod_events = []
        self.lean_events = []
        self.previous_leaning = False

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
            return 0.0, 0, False, None, None

        avg_ear = float(np.mean(valid_ears))

        self.ear_buffer.append(avg_ear)

        if len(self.ear_buffer) > 10:
            self.ear_buffer.pop(0)

        smooth_ear = float(np.mean(self.ear_buffer))

        if self.base_ear is None and len(self.ear_buffer) >= 10:
            self.base_ear = smooth_ear

        return smooth_ear, len(valid_ears), True, left_ear, right_ear

    def are_both_eyes_closed(self, left_ear, right_ear, threshold):
        return (
            left_ear is not None
            and right_ear is not None
            and left_ear < threshold
            and right_ear < threshold
        )

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

    def normalize_angle(self, angle):
        angle = ((float(angle) + 180.0) % 360.0) - 180.0

        if angle > 90.0:
            angle = 180.0 - angle
        elif angle < -90.0:
            angle = -180.0 - angle

        return angle

    def calculate_head_pose_angles(self, landmarks, w, h):
        try:
            image_points = np.array([
                self.point(landmarks, 1, w, h),
                self.point(landmarks, 152, w, h),
                self.point(landmarks, 33, w, h),
                self.point(landmarks, 263, w, h),
                self.point(landmarks, 61, w, h),
                self.point(landmarks, 291, w, h)
            ], dtype=np.float64)

            model_points = np.array([
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (-28.9, -28.9, -24.1),
                (28.9, -28.9, -24.1)
            ], dtype=np.float64)

            focal_length = float(w)
            center = (w / 2.0, h / 2.0)
            camera_matrix = np.array([
                [focal_length, 0.0, center[0]],
                [0.0, focal_length, center[1]],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return None

            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            projection_matrix = np.hstack((rotation_matrix, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(
                projection_matrix
            )

            pitch, yaw, roll = euler_angles.flatten()

            return {
                "pitch": self.normalize_angle(pitch),
                "yaw": self.normalize_angle(yaw),
                "roll": self.normalize_angle(roll)
            }
        except Exception:
            return None

    def smooth_head_pose_angles(self, angles):
        self.pose_buffer.append(angles)

        if len(self.pose_buffer) > 6:
            self.pose_buffer.pop(0)

        return {
            key: float(np.mean([item[key] for item in self.pose_buffer]))
            for key in ["pitch", "yaw", "roll"]
        }

    def angle_risk(self, angle_delta, warn_degrees, drowsy_degrees):
        if drowsy_degrees <= warn_degrees:
            return 0.0

        value = abs(float(angle_delta))
        risk = (value - warn_degrees) / (drowsy_degrees - warn_degrees)
        return max(0.0, min(risk, 1.25))

    def calculate_head_pose_metrics(self, landmarks, w, h):
        legacy_ratio = self.calculate_head_pose_score(landmarks, w, h)
        angles = self.calculate_head_pose_angles(landmarks, w, h)

        default_metrics = {
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "head_roll": 0.0,
            "head_pitch_delta": 0.0,
            "head_yaw_delta": 0.0,
            "head_roll_delta": 0.0,
            "legacy_head_ratio": round(legacy_ratio, 3)
        }

        if angles is None:
            legacy_score = max(0.0, min((legacy_ratio - 0.55) / 0.35, 1.25))
            return legacy_score, default_metrics

        smooth_angles = self.smooth_head_pose_angles(angles)

        if self.pose_baseline is None:
            self.pose_baseline = smooth_angles.copy()

        deltas = {
            key: smooth_angles[key] - self.pose_baseline[key]
            for key in ["pitch", "yaw", "roll"]
        }

        pitch_score = self.angle_risk(
            deltas["pitch"],
            DrowsinessConfig.HEAD_PITCH_WARN_DEGREES,
            DrowsinessConfig.HEAD_PITCH_DROWSY_DEGREES
        )
        yaw_score = self.angle_risk(
            deltas["yaw"],
            DrowsinessConfig.HEAD_YAW_WARN_DEGREES,
            DrowsinessConfig.HEAD_YAW_DROWSY_DEGREES
        )
        roll_score = self.angle_risk(
            deltas["roll"],
            DrowsinessConfig.HEAD_ROLL_WARN_DEGREES,
            DrowsinessConfig.HEAD_ROLL_DROWSY_DEGREES
        )
        legacy_score = max(0.0, min((legacy_ratio - 0.55) / 0.35, 1.25))
        pose_score = max(pitch_score, yaw_score, roll_score, legacy_score)

        if pose_score < 0.20:
            alpha = DrowsinessConfig.HEAD_POSE_BASELINE_ALPHA
            for key in ["pitch", "yaw", "roll"]:
                self.pose_baseline[key] = (
                    (1.0 - alpha) * self.pose_baseline[key]
                    + alpha * smooth_angles[key]
                )

        metrics = {
            "head_pitch": round(smooth_angles["pitch"], 1),
            "head_yaw": round(smooth_angles["yaw"], 1),
            "head_roll": round(smooth_angles["roll"], 1),
            "head_pitch_delta": round(deltas["pitch"], 1),
            "head_yaw_delta": round(deltas["yaw"], 1),
            "head_roll_delta": round(deltas["roll"], 1),
            "legacy_head_ratio": round(legacy_ratio, 3)
        }

        return round(pose_score, 3), metrics

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

        if visible_eye_count == 0:
            eye_score = 0.15

        confidence = area_score * 0.55 + eye_score * 0.45
        return round(float(confidence), 2)

    def landmark_box(self, landmarks, indices, w, h, padding=0.25):
        points = [self.point(landmarks, idx, w, h) for idx in indices]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        pad_x = int((x2 - x1) * padding)
        pad_y = int((y2 - y1) * padding)

        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(w, x2 + pad_x),
            min(h, y2 + pad_y)
        )

    def region_stats(self, frame, box):
        if frame is None or box is None:
            return {
                "skin_ratio": 1.0,
                "dark_ratio": 0.0,
                "mean_gray": 128.0,
                "texture": 32.0
            }

        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return {
                "skin_ratio": 1.0,
                "dark_ratio": 0.0,
                "mean_gray": 128.0,
                "texture": 32.0
            }

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return {
                "skin_ratio": 1.0,
                "dark_ratio": 0.0,
                "mean_gray": 128.0,
                "texture": 32.0
            }

        b = crop[:, :, 0].astype(np.int16)
        g = crop[:, :, 1].astype(np.int16)
        r = crop[:, :, 2].astype(np.int16)

        max_rgb = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        gray = 0.114 * b + 0.587 * g + 0.299 * r

        rgb_skin = (
            (r > 80)
            & (g > 30)
            & (b > 15)
            & (r > g)
            & (r > b)
            & ((max_rgb - min_rgb) > 12)
        )
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)

        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        cr_channel = ycrcb[:, :, 1]
        cb_channel = ycrcb[:, :, 2]

        hsv_skin = (
            (((h_channel <= 25) | (h_channel >= 165)))
            & (s_channel >= 25)
            & (s_channel <= 190)
            & (v_channel >= 45)
        )
        ycrcb_skin = (
            (cr_channel >= 128)
            & (cr_channel <= 178)
            & (cb_channel >= 72)
            & (cb_channel <= 138)
        )
        skin = rgb_skin | (hsv_skin & ycrcb_skin)

        return {
            "skin_ratio": float(np.mean(skin)),
            "dark_ratio": float(np.mean(gray < 55)),
            "mean_gray": float(np.mean(gray)),
            "texture": float(np.std(gray))
        }

    def average_region_stats(self, stats_items):
        valid_stats = [item for item in stats_items if item is not None]

        if not valid_stats:
            return {
                "skin_ratio": 1.0,
                "dark_ratio": 0.0,
                "mean_gray": 128.0,
                "texture": 32.0
            }

        return {
            key: float(np.mean([item[key] for item in valid_stats]))
            for key in ["skin_ratio", "dark_ratio", "mean_gray", "texture"]
        }

    def is_mask_like_mouth_region(self, mouth_stats, reference_skin_ratio):
        if reference_skin_ratio < DrowsinessConfig.MASK_REFERENCE_SKIN_RATIO_MIN:
            return False

        mouth_skin_low = mouth_stats["skin_ratio"] < DrowsinessConfig.MOUTH_SKIN_RATIO_MIN
        relative_drop = (
            mouth_stats["skin_ratio"]
            <= reference_skin_ratio * DrowsinessConfig.MASK_SKIN_RELATIVE_MAX
        )
        not_shadow = mouth_stats["dark_ratio"] <= DrowsinessConfig.MASK_SHADOW_DARK_RATIO_MAX
        bright_enough = mouth_stats["mean_gray"] >= DrowsinessConfig.MASK_MOUTH_MEAN_GRAY_MIN

        return bool(mouth_skin_low and relative_drop and not_shadow and bright_enough)

    def update_mask_score(self, raw_mask_detected):
        if raw_mask_detected:
            self.mask_score = min(6.0, self.mask_score + 1.0)
        else:
            self.mask_score = max(0.0, self.mask_score - 0.85)

        return self.mask_score >= DrowsinessConfig.MASK_CONFIRM_SCORE

    def estimate_face_cover(
        self,
        landmarks,
        frame,
        w,
        h,
        eye_visible,
        visible_eye_count
    ):
        if landmarks is None:
            return {
                "mouth_visible": False,
                "sunglasses_detected": False,
                "mask_detected": False
            }

        eye_indices = [33, 133, 159, 145, 362, 263, 386, 374]
        mouth_indices = [0, 13, 14, 17, 61, 78, 164, 291, 308, 152]
        left_cheek_indices = [50, 101, 118, 187, 205]
        right_cheek_indices = [280, 330, 347, 411, 425]

        eye_box = self.landmark_box(landmarks, eye_indices, w, h, padding=0.45)
        mouth_box = self.landmark_box(landmarks, mouth_indices, w, h, padding=0.35)
        left_cheek_box = self.landmark_box(
            landmarks,
            left_cheek_indices,
            w,
            h,
            padding=0.55
        )
        right_cheek_box = self.landmark_box(
            landmarks,
            right_cheek_indices,
            w,
            h,
            padding=0.55
        )

        eye_stats = self.region_stats(frame, eye_box)
        mouth_stats = self.region_stats(frame, mouth_box)
        cheek_stats = self.average_region_stats([
            self.region_stats(frame, left_cheek_box),
            self.region_stats(frame, right_cheek_box)
        ])
        reference_skin_ratio = max(
            cheek_stats["skin_ratio"],
            mouth_stats["skin_ratio"]
        )

        raw_sunglasses_detected = (
            (
                not eye_visible
                and (
                    visible_eye_count == 0
                    or eye_stats["dark_ratio"] >= DrowsinessConfig.SUNGLASSES_DARK_RATIO_MIN
                )
            )
            or (
                eye_stats["dark_ratio"] >= DrowsinessConfig.SUNGLASSES_DARK_RATIO_MIN
                and reference_skin_ratio >= DrowsinessConfig.SUNGLASSES_REFERENCE_SKIN_RATIO_MIN
                and eye_stats["skin_ratio"]
                <= reference_skin_ratio * DrowsinessConfig.SUNGLASSES_SKIN_RELATIVE_MAX
            )
        )

        raw_mask_detected = (
            eye_visible
            and self.is_mask_like_mouth_region(
                mouth_stats,
                reference_skin_ratio
            )
        )
        mask_detected = self.update_mask_score(raw_mask_detected)

        mouth_visible = not mask_detected

        if not eye_visible and mouth_stats["skin_ratio"] < 0.10:
            mouth_visible = False

        return {
            "mouth_visible": mouth_visible,
            "sunglasses_detected": raw_sunglasses_detected,
            "mask_detected": mask_detected
        }

    def select_detection_mode(
        self,
        face_count,
        eye_visible,
        mouth_visible,
        sunglasses_detected,
        mask_detected,
        confidence
    ):
        if face_count == 0:
            return "NO_FACE_MODE"

        if confidence < DrowsinessConfig.NO_EYE_CONFIDENCE_MIN:
            if not eye_visible and not mouth_visible:
                return "CAMERA_BAD_MODE"

        if mask_detected and eye_visible:
            return "MASK_EYE_POSE_MODE"

        if sunglasses_detected and mouth_visible:
            return "SUNGLASSES_MOUTH_POSE_MODE"

        if eye_visible and mouth_visible:
            return "FULL_FACE_MODE"

        if eye_visible:
            return "EYE_POSE_MODE"

        if mouth_visible:
            return "MOUTH_POSE_MODE"

        return "CAMERA_BAD_MODE"

    def detect_leaning_forward(self, face_box):
        if face_box is None:
            return False

        x1, y1, x2, y2 = face_box
        cx, cy, area = self.get_face_center_and_area(face_box)
        face_w = max(1, x2 - x1)
        face_h = max(1, y2 - y1)

        if self.face_baseline_center is None or self.face_baseline_area is None:
            self.face_baseline_center = (cx, cy)
            self.face_baseline_area = area
            return False

        base_cx, base_cy = self.face_baseline_center

        y_change = cy - base_cy
        x_change = abs(cx - base_cx)
        area_change_ratio = (area - self.face_baseline_area) / max(
            1,
            self.face_baseline_area
        )

        self.prev_face_center_y = cy
        self.prev_face_area = area

        is_leaning = (
            y_change > max(35, face_h * DrowsinessConfig.LEAN_CENTER_SHIFT_RATIO)
            or x_change > max(45, face_w * DrowsinessConfig.LEAN_SIDE_SHIFT_RATIO)
            or area_change_ratio > DrowsinessConfig.LEAN_AREA_INCREASE_RATIO
        )
        now = time.time()

        if is_leaning:
            if self.leaning_start is None:
                self.leaning_start = now
        else:
            self.leaning_start = None
            alpha = DrowsinessConfig.LEAN_BASELINE_ALPHA
            self.face_baseline_center = (
                (1.0 - alpha) * base_cx + alpha * cx,
                (1.0 - alpha) * base_cy + alpha * cy
            )
            self.face_baseline_area = (
                (1.0 - alpha) * self.face_baseline_area + alpha * area
            )

        return is_leaning

    def is_in_lean_grace(self):
        if self.leaning_start is None:
            return False

        return time.time() - self.leaning_start < DrowsinessConfig.LEAN_GRACE_SECONDS

    def trim_posture_events(self, now):
        window = DrowsinessConfig.POSTURE_EVENT_WINDOW_SECONDS
        self.nod_events = [
            event_time for event_time in self.nod_events
            if now - event_time <= window
        ]
        self.lean_events = [
            event_time for event_time in self.lean_events
            if now - event_time <= window
        ]

    def detect_nod_cycle(self, now, head_pose, pose_metrics, selected_face_box):
        face_drop_ratio = 0.0

        if (
            selected_face_box is not None
            and self.face_baseline_center is not None
        ):
            _, cy, _ = self.get_face_center_and_area(selected_face_box)
            _, base_cy = self.face_baseline_center
            face_h = max(1, selected_face_box[3] - selected_face_box[1])
            face_drop_ratio = max(0.0, (cy - base_cy) / face_h)

        strong_pose_drop = (
            head_pose >= DrowsinessConfig.NOD_STRONG_DOWN_SCORE_THRESHOLD
        )
        strong_face_drop = (
            face_drop_ratio >= DrowsinessConfig.NOD_STRONG_FACE_DROP_RATIO
        )
        combined_pose_drop = (
            head_pose >= DrowsinessConfig.NOD_DOWN_SCORE_THRESHOLD
            and face_drop_ratio >= DrowsinessConfig.NOD_FACE_DROP_RATIO
        )
        downward_pose = strong_pose_drop or strong_face_drop or combined_pose_drop
        recovered_pose = (
            head_pose <= DrowsinessConfig.NOD_RECOVERY_SCORE_THRESHOLD
            and face_drop_ratio < DrowsinessConfig.NOD_FACE_DROP_RATIO * 0.55
        )

        posture_event = None

        if self.nod_phase == "stable":
            if downward_pose:
                self.nod_phase = "down"
                self.nod_down_started_at = now

        elif self.nod_phase == "down":
            duration = now - self.nod_down_started_at

            if recovered_pose:
                is_valid_duration = (
                    DrowsinessConfig.NOD_MIN_SECONDS
                    <= duration
                    <= DrowsinessConfig.NOD_MAX_SECONDS
                )
                is_after_cooldown = (
                    now - self.last_posture_event_at
                    >= DrowsinessConfig.POSTURE_EVENT_COOLDOWN_SECONDS
                )

                if is_valid_duration and is_after_cooldown:
                    self.nod_events.append(now)
                    self.last_posture_event_at = now
                    posture_event = "HEAD_NOD"

                self.nod_phase = "stable"
                self.nod_down_started_at = None

            elif duration > DrowsinessConfig.NOD_MAX_SECONDS:
                self.nod_phase = "stable"
                self.nod_down_started_at = None

        return posture_event

    def update_posture_events(
        self,
        now,
        is_leaning,
        head_pose,
        pose_metrics,
        selected_face_box
    ):
        self.trim_posture_events(now)
        posture_event = self.detect_nod_cycle(
            now,
            head_pose,
            pose_metrics,
            selected_face_box
        )

        is_after_cooldown = (
            now - self.last_posture_event_at
            >= DrowsinessConfig.POSTURE_EVENT_COOLDOWN_SECONDS
        )

        if is_leaning and not self.previous_leaning and is_after_cooldown:
            self.lean_events.append(now)
            self.last_posture_event_at = now

            if posture_event is None:
                posture_event = "LEAN_REPEAT"

        self.previous_leaning = is_leaning
        self.trim_posture_events(now)

        nod_count = len(self.nod_events)
        lean_count = len(self.lean_events)
        posture_score = 0.0

        if nod_count >= DrowsinessConfig.NOD_REPEAT_WARNING_COUNT:
            posture_score += 1.2

        if nod_count >= DrowsinessConfig.NOD_REPEAT_DROWSY_COUNT:
            posture_score += 1.4

        if lean_count >= DrowsinessConfig.LEAN_REPEAT_WARNING_COUNT:
            posture_score += 0.8

        if lean_count >= DrowsinessConfig.LEAN_REPEAT_DROWSY_COUNT:
            posture_score += 1.0

        if nod_count and lean_count:
            posture_score += 0.4

        if nod_count >= DrowsinessConfig.NOD_REPEAT_DROWSY_COUNT:
            posture_status = "HEAD_NOD_REPEAT"
        elif lean_count >= DrowsinessConfig.LEAN_REPEAT_DROWSY_COUNT:
            posture_status = "LEAN_REPEAT"
        elif nod_count >= DrowsinessConfig.NOD_REPEAT_WARNING_COUNT:
            posture_status = "HEAD_NOD_WARNING"
        elif lean_count >= DrowsinessConfig.LEAN_REPEAT_WARNING_COUNT:
            posture_status = "LEAN_WARNING"
        elif posture_event:
            posture_status = posture_event
        else:
            posture_status = "STABLE"

        return {
            "posture_event": posture_event,
            "posture_status": posture_status,
            "posture_score": round(posture_score, 2),
            "nod_count": nod_count,
            "lean_event_count": lean_count
        }

    def is_bad_camera_grace(self, confidence):
        now = time.time()

        if confidence < 0.35:
            if self.bad_camera_start is None:
                self.bad_camera_start = now

            return now - self.bad_camera_start < DrowsinessConfig.BAD_CAMERA_GRACE_SECONDS

        self.bad_camera_start = None
        return False

    def update_mode_timers(self, detection_mode, mar, head_pose_value):
        now = time.time()
        inactive_modes = {"NO_FACE_MODE", "CAMERA_BAD_MODE"}

        if detection_mode in inactive_modes:
            self.head_down_start = None
            self.no_eye_yawn_start = None
            self.no_eye_start = None

            return {
                "head_down_duration": 0.0,
                "fallback_yawn_duration": 0.0,
                "fallback_score": 0.0
            }

        mouth_modes = {
            "FULL_FACE_MODE",
            "SUNGLASSES_MOUTH_POSE_MODE",
            "MOUTH_POSE_MODE"
        }

        no_eye_modes = {
            "SUNGLASSES_MOUTH_POSE_MODE",
            "MOUTH_POSE_MODE"
        }

        if detection_mode in no_eye_modes:
            if self.no_eye_start is None:
                self.no_eye_start = now
        else:
            self.no_eye_start = None

        if head_pose_value > DrowsinessConfig.HEAD_DOWN_THRESHOLD:
            if self.head_down_start is None:
                self.head_down_start = now
        else:
            self.head_down_start = None

        if detection_mode in mouth_modes and mar > DrowsinessConfig.MAR_THRESHOLD:
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
            "head_down_duration": round(head_down_duration, 2),
            "fallback_yawn_duration": round(fallback_yawn_duration, 2),
            "fallback_score": round(fallback_score, 2)
        }

    def update(self, face_count, selected_face_box, landmarks, frame_w, frame_h, frame=None):
        now = time.time()

        ear = 0.0
        mar = 0.0
        head_pose = 0.0
        pose_metrics = {
            "head_pitch": 0.0,
            "head_yaw": 0.0,
            "head_roll": 0.0,
            "head_pitch_delta": 0.0,
            "head_yaw_delta": 0.0,
            "head_roll_delta": 0.0,
            "legacy_head_ratio": 0.0
        }
        confidence = 0.0
        visible_eye_count = 0
        eye_visible = False
        left_ear = None
        right_ear = None
        both_eyes_closed = False
        mouth_visible = False
        sunglasses_detected = False
        mask_detected = False
        eye_closed_duration = 0.0
        yawn_duration = 0.0
        is_leaning = False
        posture_metrics = {
            "posture_event": None,
            "posture_status": "STABLE",
            "posture_score": 0.0,
            "nod_count": 0,
            "lean_event_count": 0
        }

        if selected_face_box is not None and landmarks is not None:
            (
                ear,
                visible_eye_count,
                eye_visible,
                left_ear,
                right_ear
            ) = self.calculate_ear(landmarks, frame_w, frame_h)

            mar = self.calculate_mar(landmarks, frame_w, frame_h)
            head_pose, pose_metrics = self.calculate_head_pose_metrics(
                landmarks,
                frame_w,
                frame_h
            )
            confidence = self.estimate_confidence(
                selected_face_box,
                frame_w,
                frame_h,
                visible_eye_count
            )
            is_leaning = self.detect_leaning_forward(selected_face_box)
            cover = self.estimate_face_cover(
                landmarks,
                frame,
                frame_w,
                frame_h,
                eye_visible,
                visible_eye_count
            )
            mouth_visible = cover["mouth_visible"]
            sunglasses_detected = cover["sunglasses_detected"]
            mask_detected = cover["mask_detected"]

        if face_count == 0 or selected_face_box is None:
            self.drowsy_score = max(0.0, self.drowsy_score - 0.3)
            self.eye_closed_start = None
            self.yawn_start = None

        threshold = (
            self.base_ear * 0.76
            if self.base_ear
            else DrowsinessConfig.EAR_DEFAULT_THRESHOLD
        )
        both_eyes_closed = self.are_both_eyes_closed(
            left_ear,
            right_ear,
            threshold
        )

        if eye_visible and ear > 0:
            if both_eyes_closed:
                if self.eye_closed_start is None:
                    self.eye_closed_start = now
            else:
                self.eye_closed_start = None

        if self.eye_closed_start:
            eye_closed_duration = now - self.eye_closed_start

        if mouth_visible and mar > DrowsinessConfig.MAR_THRESHOLD:
            if self.yawn_start is None:
                self.yawn_start = now
        else:
            self.yawn_start = None

        if self.yawn_start:
            yawn_duration = now - self.yawn_start

        detection_mode = self.select_detection_mode(
            face_count=face_count,
            eye_visible=eye_visible,
            mouth_visible=mouth_visible,
            sunglasses_detected=sunglasses_detected,
            mask_detected=mask_detected,
            confidence=confidence
        )

        fallback = self.update_mode_timers(
            detection_mode=detection_mode,
            mar=mar,
            head_pose_value=head_pose
        )

        fallback_score = fallback["fallback_score"]
        head_down_duration = fallback["head_down_duration"]
        fallback_yawn_duration = fallback["fallback_yawn_duration"]

        bad_camera = detection_mode == "CAMERA_BAD_MODE" and face_count > 0
        bad_camera_grace = self.is_bad_camera_grace(confidence)
        posture_metrics = self.update_posture_events(
            now=now,
            is_leaning=is_leaning,
            head_pose=head_pose,
            pose_metrics=pose_metrics,
            selected_face_box=selected_face_box
        )
        posture_score = posture_metrics["posture_score"]
        repeated_posture = posture_score >= 1.2
        leaning_grace = self.is_in_lean_grace() and not repeated_posture

        full_face_modes = {"FULL_FACE_MODE"}
        eye_pose_modes = {"MASK_EYE_POSE_MODE", "EYE_POSE_MODE"}
        mouth_pose_modes = {"SUNGLASSES_MOUTH_POSE_MODE", "MOUTH_POSE_MODE"}

        if not leaning_grace and not bad_camera_grace:
            if detection_mode in full_face_modes:
                if eye_closed_duration > DrowsinessConfig.EYE_CLOSED_SECONDS:
                    self.drowsy_score += 1.15

                if yawn_duration > DrowsinessConfig.YAWN_SECONDS:
                    self.drowsy_score += 0.65

                if head_pose > DrowsinessConfig.HEAD_DOWN_THRESHOLD:
                    self.drowsy_score += 0.75

            elif detection_mode in eye_pose_modes:
                if eye_closed_duration > DrowsinessConfig.EYE_CLOSED_SECONDS:
                    self.drowsy_score += 1.2

                if head_down_duration > DrowsinessConfig.NO_EYE_HEAD_DOWN_SECONDS:
                    self.drowsy_score += 0.95

            elif detection_mode in mouth_pose_modes:
                if fallback_score >= 4.0:
                    self.drowsy_score += 0.9

                if fallback_score >= 7.0:
                    self.drowsy_score += 1.2

            else:
                self.drowsy_score = max(0.0, self.drowsy_score - 0.35)

            should_apply_posture_score = (
                posture_score > 0
                and (
                    posture_metrics["posture_event"] is not None
                    or now - self.last_posture_score_at >= 2.0
                )
            )

            if should_apply_posture_score:
                self.drowsy_score += posture_score
                self.last_posture_score_at = now

        else:
            should_apply_posture_score = (
                repeated_posture
                and not bad_camera_grace
                and (
                    posture_metrics["posture_event"] is not None
                    or now - self.last_posture_score_at >= 2.0
                )
            )

            if should_apply_posture_score:
                self.drowsy_score += posture_score * 0.65
                self.last_posture_score_at = now
            else:
                self.drowsy_score = max(0.0, self.drowsy_score - 0.45)

        if detection_mode in full_face_modes:
            if (
                eye_closed_duration < 0.7
                and yawn_duration < 0.8
                and head_pose < 0.68
                and confidence >= 0.35
            ):
                self.drowsy_score -= 0.35

        elif detection_mode in eye_pose_modes:
            if (
                eye_closed_duration < 0.7
                and head_pose < DrowsinessConfig.HEAD_DOWN_THRESHOLD
                and confidence >= DrowsinessConfig.NO_EYE_CONFIDENCE_MIN
            ):
                self.drowsy_score -= 0.3

        elif detection_mode in mouth_pose_modes:
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
            detection_mode in mouth_pose_modes
            and self.drowsy_score >= DrowsinessConfig.NO_EYE_DROWSY_SCORE_LIMIT
        ):
            state = "DROWSY_CONFIRMED"
            is_drowsy = True
        elif (
            detection_mode in full_face_modes.union(eye_pose_modes)
            and self.drowsy_score >= DrowsinessConfig.NORMAL_DROWSY_LIMIT
        ):
            state = "DROWSY_CONFIRMED"
            is_drowsy = True
        elif (
            repeated_posture
            and posture_metrics["nod_count"] >= DrowsinessConfig.NOD_REPEAT_DROWSY_COUNT
            and self.drowsy_score >= DrowsinessConfig.WARNING_LIMIT
        ):
            state = "DROWSY_CONFIRMED"
            is_drowsy = True
        elif self.drowsy_score >= DrowsinessConfig.WARNING_LIMIT:
            if posture_metrics["posture_status"] in {
                "HEAD_NOD_REPEAT",
                "HEAD_NOD_WARNING",
                "LEAN_REPEAT",
                "LEAN_WARNING"
            }:
                state = "WARNING_POSTURE_MODE"
            elif detection_mode == "SUNGLASSES_MOUTH_POSE_MODE":
                state = "WARNING_SUNGLASSES_MODE"
            elif detection_mode == "MASK_EYE_POSE_MODE":
                state = "WARNING_MASK_MODE"
            else:
                state = "WARNING_LEVEL_1"

            is_drowsy = False
        else:
            if detection_mode == "SUNGLASSES_MOUTH_POSE_MODE":
                state = "SUNGLASSES_MODE"
            elif detection_mode == "MASK_EYE_POSE_MODE":
                state = "MASK_MODE"
            else:
                state = "NORMAL"

            is_drowsy = False

        return {
            "ear": round(ear, 3),
            "mar": round(mar, 3),
            "head_pose": round(head_pose, 3),
            **pose_metrics,
            "confidence": confidence,
            "visible_eye_count": visible_eye_count,
            "eye_visible": eye_visible,
            "left_ear": round(left_ear, 3) if left_ear is not None else None,
            "right_ear": round(right_ear, 3) if right_ear is not None else None,
            "both_eyes_closed": both_eyes_closed,
            "mouth_visible": mouth_visible,
            "sunglasses_detected": sunglasses_detected,
            "mask_detected": mask_detected,
            "is_leaning": is_leaning,
            "threshold": round(threshold, 3),
            "drowsy_score": round(self.drowsy_score, 2),
            "state": state,
            "is_drowsy": is_drowsy,
            "detection_mode": detection_mode,
            **posture_metrics,
            "fallback_score": fallback_score,
            "head_down_duration": head_down_duration,
            "fallback_yawn_duration": fallback_yawn_duration,
            "can_alert": now - self.last_alert > DrowsinessConfig.ALERT_COOLDOWN
        }

    def mark_alerted(self):
        self.last_alert = time.time()
