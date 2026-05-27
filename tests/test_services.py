import os
import tempfile
import unittest

from config import AppConfig
from services.alert_logger import AlertLogger
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
