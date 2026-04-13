import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOFeatureExtractor(nn.Module):
    def __init__(self, model_path='yolov11n.pt'):
        super().__init__()
        # Load model YOLO và chỉ lấy phần backbone trích xuất đặc trưng
        yolo = YOLO(model_path).model.model
        self.backbone = nn.Sequential(*list(yolo.children())[:10]) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # x: [batch, 3, 224, 224]
        features = self.backbone(x)
        out = self.avgpool(features)
        return torch.flatten(out, 1) # Trả về vector 512D hoặc 1024D tùy version