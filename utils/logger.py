import csv
from datetime import datetime
import os

class DriveLogger:
    def __init__(self, path="logs/"):
        if not os.path.exists(path):
            os.makedirs(path)
        self.filename = f"{path}drive_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(self.filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "EAR", "Pose_Pitch", "Status", "Drowsy_Prob"])

    def log(self, ear, pose, status, prob):
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), ear, pose, status, prob])