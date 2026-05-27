# services/hybrid_predictor.py

import os
import torch
import numpy as np

from models.hybrid_model import DriverMonitoringSystem


class HybridPredictor:
    def __init__(self, weight_path="weights/hybrid_best.pth", seq_len=30):
        self.seq_len = seq_len
        self.weight_path = weight_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.spatial_buffer = []
        self.geo_buffer = []

        self.enabled = os.path.exists(weight_path)

        if self.enabled:
            self.model = DriverMonitoringSystem(feature_dim=256).to(self.device)
            self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
            self.model.eval()
            print("Đã tải Hybrid Transformer:", weight_path)
        else:
            print("Chưa có weights/hybrid_best.pth, AI mode sẽ tắt.")

    def update_features(self, spatial_feature, geo_feature):
        self.spatial_buffer.append(spatial_feature)
        self.geo_buffer.append(geo_feature)

        if len(self.spatial_buffer) > self.seq_len:
            self.spatial_buffer.pop(0)

        if len(self.geo_buffer) > self.seq_len:
            self.geo_buffer.pop(0)

    def is_ready(self):
        return self.enabled and len(self.spatial_buffer) >= self.seq_len

    def predict(self):
        if not self.is_ready():
            return {
                "ai_ready": False,
                "ai_probability": 0.0,
                "ai_label": 0
            }

        spatial = torch.tensor(
            np.array(self.spatial_buffer),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        geo = torch.tensor(
            np.array(self.geo_buffer),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(spatial, geo)
            prob = torch.softmax(output, dim=1)[0][1].item()
            label = 1 if prob >= 0.5 else 0

        return {
            "ai_ready": True,
            "ai_probability": round(prob, 4),
            "ai_label": label
        }