import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiBranchAttention(nn.Module):
    def __init__(self, feature_dim=256, num_heads=8):
        super().__init__()
        # Cross-attention để hòa trộn đặc trưng YOLO và đặc trưng Hình học
        self.multihead_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, spatial_feats, geo_feats):
        """
        spatial_feats: Đặc trưng từ YOLO [Batch, Seq, 256]
        geo_feats: Đặc trưng EAR/Pose [Batch, Seq, 256]
        """
        # Sử dụng geo_feats làm Query để tìm kiếm thông tin liên quan trong spatial_feats
        attn_output, weights = self.multihead_attn(query=geo_feats, key=spatial_feats, value=spatial_feats)
        
        # Residual connection và Layer Normalization
        out = self.norm(attn_output + geo_feats)
        return out, weights # Trả về weights để phục vụ việc giải thích mô hình (Explainable AI)