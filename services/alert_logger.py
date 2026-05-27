# services/alert_logger.py

import json
import os
from datetime import datetime
from threading import Lock

from config import AppConfig


class AlertLogger:
    def __init__(self, log_path="data/alert_logs.json"):
        self.log_path = log_path
        self.lock = Lock()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        if not os.path.exists(self.log_path):
            self.save_logs([])
        else:
            self.save_logs(self.load_logs()[-AppConfig.ALERT_HISTORY_LIMIT:])

    def load_logs(self):
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def save_logs(self, logs):
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)

    def trim_logs(self, logs):
        return logs[-AppConfig.ALERT_HISTORY_LIMIT:]

    def add_alert(self, result, location=None, stops=None):
        with self.lock:
            logs = self.load_logs()

            item = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "state": result.get("state"),
                "ear": result.get("ear"),
                "mar": result.get("mar"),
                "head_pose": result.get("head_pose"),
                "threshold": result.get("threshold"),
                "drowsy_score": result.get("drowsy_score"),
                "confidence": result.get("confidence"),
                "visible_eye_count": result.get("visible_eye_count"),
                "mouth_visible": result.get("mouth_visible"),
                "sunglasses_detected": result.get("sunglasses_detected"),
                "mask_detected": result.get("mask_detected"),
                "detection_mode": result.get("detection_mode"),
                "fallback_score": result.get("fallback_score"),
                "location": location,
                "stops": stops or []
            }

            logs.append(item)
            logs = self.trim_logs(logs)

            self.save_logs(logs)
            return item

    def update_latest_location(self, location=None, stops=None):
        with self.lock:
            logs = self.load_logs()

            if not logs:
                return None

            logs[-1]["location"] = location
            logs[-1]["stops"] = stops or []

            self.save_logs(self.trim_logs(logs))
            return logs[-1]

    def get_recent_logs(self, limit=AppConfig.ALERT_HISTORY_LIMIT):
        logs = self.load_logs()
        return logs[-limit:][::-1]

    def get_stats(self):
        logs = self.load_logs()

        total = len(logs)
        drowsy_count = len([
            item for item in logs
            if item.get("state") == "DROWSY_CONFIRMED"
        ])
        warning_count = len([
            item for item in logs
            if "WARNING" in str(item.get("state", ""))
        ])

        return {
            "total_alerts": total,
            "drowsy_count": drowsy_count,
            "warning_count": warning_count
        }
