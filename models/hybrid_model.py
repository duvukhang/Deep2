import torch
import torch.nn as nn

# Khang lưu ý: KHÔNG import chính file này ở đây nữa nhé!
# Tạm thời comment 2 dòng này lại nếu bạn chưa tạo file, để tránh lỗi ModuleNotFoundError
# from .attention_fusion import MultiBranchAttention
# from .temporal_transformer import TemporalEncoder

class DriverMonitoringSystem(nn.Module):
    def __init__(self, feature_dim=256):
        super(DriverMonitoringSystem, self).__init__()
        
        # Nhánh 1: Xử lý vector đặc trưng từ YOLO (Spatial)
        self.spatial_fc = nn.Sequential(
            nn.Linear(512, feature_dim), 
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Nhánh 2: Xử lý đặc trưng hình học (Geometric: EAR, MAR, Pose)
        self.geo_fc = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        # Phân loại đầu ra (0: Tỉnh táo, 1: Buồn ngủ)
        # Lưu ý: Khi nào bạn code xong file Attention và Transformer thì tích hợp vào sau
        self.classifier = nn.Linear(feature_dim * 2, 2)

    def forward(self, spatial_seq, geo_seq):
        batch_size, seq_len, _ = spatial_seq.size()
        
        s_feats = self.spatial_fc(spatial_seq) # [B, S, 256]
        g_feats = self.geo_fc(geo_seq)         # [B, S, 256]
        
        # Tạm thời nối (concat) 2 đặc trưng lại để chạy test trước
        fused_seq = torch.cat((s_feats, g_feats), dim=-1) # [B, S, 512]
        
        # Lấy trạng thái ở frame cuối cùng để dự đoán
        out = self.classifier(fused_seq[:, -1, :])
        return out