import torch
import torch.nn as nn

from .attention_fusion import MultiBranchAttention
from .temporal_transformer import TemporalEncoder


class DriverMonitoringSystem(nn.Module):
    def __init__(self, feature_dim=256):
        super(DriverMonitoringSystem, self).__init__()

        # Nhánh 1: đặc trưng không gian từ YOLO/CNN
        self.spatial_fc = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Nhánh 2: đặc trưng hình học EAR, MAR, Head Pose...
        self.geo_fc = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Trộn 2 nhánh bằng attention
        self.fusion = MultiBranchAttention(
            feature_dim=feature_dim,
            num_heads=8
        )

        # Học chuỗi thời gian bằng Transformer
        self.temporal = TemporalEncoder(
            d_model=feature_dim,
            nhead=8,
            num_layers=2
        )

        # Phân loại: 0 tỉnh táo, 1 buồn ngủ
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, spatial_seq, geo_seq):
        s_feats = self.spatial_fc(spatial_seq)
        g_feats = self.geo_fc(geo_seq)

        fused_seq, attention_weights = self.fusion(s_feats, g_feats)

        temporal_out = self.temporal(fused_seq)

        # Lấy đặc trưng trung bình theo chuỗi thay vì chỉ frame cuối
        pooled = temporal_out.mean(dim=1)

        out = self.classifier(pooled)

        return out