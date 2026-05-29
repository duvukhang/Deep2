import os
import tempfile
import unittest

import numpy as np

from config import AppConfig
from services.alert_logger import AlertLogger
from services.camera_lighting import CameraLightingController
from services.drowsiness_detector import DrowsinessDetector
from services.osm_service import OSMService


class AlertLoggerTest(unittest.TestCase):
    def test_keeps_only_configured_history_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "alerts.json")
            logger = AlertLogger(log_path=log_path)

            for index in range(AppConfig.ALERT_HISTORY_LIMIT + 5):
                logger.add_alert({
                    "state": "WARNING_LEVEL_1",
                    "ear": 0.2,
                    "mar": 0.1,
                    "head_pose": 0.1,
                    "drowsy_score": index
                })

            logs = logger.load_logs()

            self.assertEqual(len(logs), AppConfig.ALERT_HISTORY_LIMIT)
            self.assertEqual(logs[-1]["drowsy_score"], AppConfig.ALERT_HISTORY_LIMIT + 4)


class DrowsinessDetectorTest(unittest.TestCase):
    def test_selects_expected_detection_modes(self):
        detector = DrowsinessDetector()

        self.assertEqual(
            detector.select_detection_mode(1, True, True, False, False, 0.8),
            "FULL_FACE_MODE"
        )
        self.assertEqual(
            detector.select_detection_mode(1, False, True, True, False, 0.8),
            "SUNGLASSES_MOUTH_POSE_MODE"
        )
        self.assertEqual(
            detector.select_detection_mode(1, True, False, False, True, 0.8),
            "MASK_EYE_POSE_MODE"
        )
        self.assertEqual(
            detector.select_detection_mode(1, False, False, False, False, 0.1),
            "CAMERA_BAD_MODE"
        )

    def test_pose_angle_risk_increases_after_warning_angle(self):
        detector = DrowsinessDetector()

        self.assertEqual(detector.angle_risk(8, 12, 28), 0.0)
        self.assertGreater(detector.angle_risk(24, 12, 28), 0.0)

    def test_mask_region_needs_reliable_skin_reference_and_not_shadow(self):
        detector = DrowsinessDetector()
        mouth_stats = {
            "skin_ratio": 0.04,
            "dark_ratio": 0.12,
            "mean_gray": 125.0,
            "texture": 18.0
        }
        shadow_stats = {
            "skin_ratio": 0.04,
            "dark_ratio": 0.72,
            "mean_gray": 55.0,
            "texture": 22.0
        }

        self.assertFalse(
            detector.is_mask_like_mouth_region(mouth_stats, reference_skin_ratio=0.06)
        )
        self.assertFalse(
            detector.is_mask_like_mouth_region(shadow_stats, reference_skin_ratio=0.35)
        )
        self.assertTrue(
            detector.is_mask_like_mouth_region(mouth_stats, reference_skin_ratio=0.35)
        )

    def test_mask_mode_requires_stable_frames(self):
        detector = DrowsinessDetector()

        self.assertFalse(detector.update_mask_score(True))
        self.assertFalse(detector.update_mask_score(True))
        self.assertTrue(detector.update_mask_score(True))
        self.assertFalse(detector.update_mask_score(False))

    def test_detects_repeated_head_nods(self):
        detector = DrowsinessDetector()

        detector.update_posture_events(10.0, False, 0.90, {}, None)
        first = detector.update_posture_events(10.45, False, 0.10, {}, None)
        detector.update_posture_events(12.0, False, 0.90, {}, None)
        second = detector.update_posture_events(12.45, False, 0.10, {}, None)
        detector.update_posture_events(14.0, False, 0.90, {}, None)
        third = detector.update_posture_events(14.45, False, 0.10, {}, None)
        detector.update_posture_events(16.0, False, 0.90, {}, None)
        fourth = detector.update_posture_events(16.45, False, 0.10, {}, None)

        self.assertEqual(first["posture_event"], "HEAD_NOD")
        self.assertEqual(second["posture_status"], "HEAD_NOD")
        self.assertEqual(second["posture_score"], 0.0)
        self.assertEqual(third["posture_status"], "HEAD_NOD_WARNING")
        self.assertEqual(third["nod_count"], 3)
        self.assertEqual(fourth["posture_status"], "HEAD_NOD_REPEAT")
        self.assertEqual(fourth["nod_count"], 4)
        self.assertGreater(third["posture_score"], second["posture_score"])

    def test_moderate_pose_without_face_drop_is_not_a_nod(self):
        detector = DrowsinessDetector()

        detector.update_posture_events(30.0, False, 0.74, {}, None)
        result = detector.update_posture_events(30.5, False, 0.10, {}, None)

        self.assertIsNone(result["posture_event"])
        self.assertEqual(result["nod_count"], 0)

    def test_detects_repeated_lean_events(self):
        detector = DrowsinessDetector()

        first = detector.update_posture_events(20.0, True, 0.10, {}, None)
        detector.update_posture_events(20.5, False, 0.10, {}, None)
        second = detector.update_posture_events(22.0, True, 0.10, {}, None)
        detector.update_posture_events(22.5, False, 0.10, {}, None)
        third = detector.update_posture_events(24.0, True, 0.10, {}, None)

        self.assertEqual(first["posture_event"], "LEAN_REPEAT")
        self.assertEqual(second["posture_status"], "LEAN_WARNING")
        self.assertEqual(third["posture_status"], "LEAN_REPEAT")
        self.assertEqual(third["lean_event_count"], 3)


class CameraLightingControllerTest(unittest.TestCase):
    def test_brightens_dark_frames(self):
        controller = CameraLightingController()
        frame = np.full((80, 80, 3), 28, dtype=np.uint8)

        enhanced, info = controller.enhance(frame)

        self.assertEqual(info["lighting_mode"], "LOW_LIGHT")
        self.assertGreater(float(enhanced.mean()), float(frame.mean()))

    def test_reduces_glare_frames(self):
        controller = CameraLightingController()
        frame = np.full((80, 80, 3), 245, dtype=np.uint8)

        enhanced, info = controller.enhance(frame)

        self.assertEqual(info["lighting_mode"], "GLARE")
        self.assertLess(float(enhanced.mean()), float(frame.mean()))


class OSMServiceTest(unittest.TestCase):
    def test_prioritizes_useful_stop_and_forward_direction(self):
        osm = OSMService()
        lat = 10.0
        lon = 106.0

        stops = [
            osm.normalize_stop(
                {"tags": {"amenity": "bench", "name": "Ghe gan"}, "lat": 10.002, "lon": 106.0},
                lat,
                lon,
                heading=0
            ),
            osm.normalize_stop(
                {"tags": {"amenity": "cafe", "name": "Cafe phia truoc"}, "lat": 10.05, "lon": 106.0},
                lat,
                lon,
                heading=0
            ),
            osm.normalize_stop(
                {"tags": {"amenity": "fuel", "name": "Cay xang phia sau"}, "lat": 9.98, "lon": 106.0},
                lat,
                lon,
                heading=0
            )
        ]

        ranked = osm.unique_and_sort([stop for stop in stops if stop is not None])

        self.assertEqual(ranked[0]["name"], "Cafe phia truoc")
        self.assertGreater(ranked[0]["rest_score"], ranked[-1]["rest_score"])


if __name__ == "__main__":
    unittest.main()
