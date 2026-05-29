# services/camera_lighting.py

import time

import cv2
import numpy as np


class CameraLightingController:
    def __init__(self):
        self.smoothed_mean = None
        self.last_capture_tune = 0.0

    def configure_capture(self, cap):
        try:
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)
        except Exception:
            pass

        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        except Exception:
            pass

    def apply_gamma(self, frame, gamma):
        gamma = max(0.35, min(float(gamma), 2.4))
        values = np.arange(256, dtype=np.float32) / 255.0
        table = np.clip((values ** gamma) * 255.0, 0, 255).astype(np.uint8)
        return cv2.LUT(frame, table)

    def apply_clahe(self, frame, clip_limit=2.0):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)
        merged = cv2.merge((enhanced_l, a_channel, b_channel))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def tune_capture(self, cap, lighting_mode):
        now = time.time()

        if now - self.last_capture_tune < 1.0:
            return

        self.last_capture_tune = now

        try:
            if lighting_mode == "LOW_LIGHT":
                cap.set(cv2.CAP_PROP_GAIN, 0.6)
            elif lighting_mode == "GLARE":
                cap.set(cv2.CAP_PROP_GAIN, 0.35)
        except Exception:
            pass

    def enhance(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_mean = float(gray.mean())

        if self.smoothed_mean is None:
            self.smoothed_mean = raw_mean
        else:
            self.smoothed_mean = 0.82 * self.smoothed_mean + 0.18 * raw_mean

        p5, p95 = np.percentile(gray, [5, 95])
        contrast = float(p95 - p5)
        dark_ratio = float(np.mean(gray < 45))
        glare_ratio = float(np.mean(gray > 235))
        lighting_mode = "NORMAL"
        enhanced = frame

        if self.smoothed_mean < 78 or dark_ratio > 0.24:
            lighting_mode = "LOW_LIGHT"
            gamma = 0.62 if self.smoothed_mean < 52 else 0.76
            enhanced = self.apply_gamma(frame, gamma)
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.08, beta=10)

            if contrast < 78:
                enhanced = self.apply_clahe(enhanced, clip_limit=2.4)

        elif self.smoothed_mean > 178 or glare_ratio > 0.08:
            lighting_mode = "GLARE"
            gamma = 1.32 if self.smoothed_mean < 210 else 1.55
            enhanced = self.apply_gamma(frame, gamma)
            enhanced = cv2.convertScaleAbs(enhanced, alpha=0.88, beta=-12)

        elif contrast < 46:
            lighting_mode = "LOW_CONTRAST"
            enhanced = self.apply_clahe(frame, clip_limit=1.8)

        info = {
            "lighting_mode": lighting_mode,
            "brightness": round(self.smoothed_mean, 1),
            "contrast": round(contrast, 1),
            "dark_ratio": round(dark_ratio, 3),
            "glare_ratio": round(glare_ratio, 3)
        }

        return enhanced, info
