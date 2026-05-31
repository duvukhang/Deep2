import torch
import torch.nn as nn

from .attention_fusion import MultiBranchAttention
from .temporal_transformer import TemporalEncoder


class DriverMonitoringSystem(nn.Module):
    """Hybrid temporal classifier for driver drowsiness.

    Expected inputs:
        spatial_seq: [batch, seq_len, 512]
        geo_seq: [batch, seq_len, 6]

    geo order used by the training pipeline:
        [EAR, MAR, pitch, yaw, roll, camera_confidence]
    camera_confidence should be in [0, 1]. Low confidence softly reduces both
    the geometric branch and final certainty instead of letting noisy EAR/MAR
    dominate a sequence.
    """

    def __init__(
        self,
        spatial_dim=512,
        geo_dim=6,
        feature_dim=256,
        num_classes=2,
        dropout=0.30,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.spatial_norm = nn.LayerNorm(spatial_dim)
        self.geo_norm = nn.LayerNorm(geo_dim)

        self.spatial_fc = nn.Sequential(
            nn.Linear(spatial_dim, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
        )

        self.geo_fc = nn.Sequential(
            nn.Linear(geo_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout * 0.6),
            nn.Linear(128, feature_dim),
            nn.GELU(),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
        )

        self.geo_reliability = nn.Sequential(
            nn.Linear(geo_dim, 64),
            nn.GELU(),
            nn.Linear(64, feature_dim),
            nn.Sigmoid(),
        )

        self.branch_gate = nn.Sequential(
            nn.Linear(feature_dim * 2 + geo_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(feature_dim, 2),
        )

        self.fusion = MultiBranchAttention(feature_dim=feature_dim, num_heads=8)
        self.fusion_norm = nn.LayerNorm(feature_dim)

        self.temporal = TemporalEncoder(
            d_model=feature_dim,
            nhead=8,
            num_layers=2,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, spatial_seq, geo_seq, return_attention=False):
        if spatial_seq.dim() != 3 or geo_seq.dim() != 3:
            raise ValueError("spatial_seq va geo_seq phai co shape [batch, seq_len, dim].")

        spatial_input = self.spatial_norm(spatial_seq)
        geo_input = self.geo_norm(geo_seq)

        spatial_feats = self.spatial_fc(spatial_input)
        geo_feats = self.geo_fc(geo_input)

        camera_confidence = torch.clamp(geo_seq[..., 5:6], 0.0, 1.0)
        geo_reliability = self.geo_reliability(geo_input)

        # Low confidence means bad angle/lighting/occlusion. Do not fully remove
        # either branch; keep a floor so the temporal model can still learn from
        # repeated posture patterns.
        spatial_feats = spatial_feats * (0.45 + 0.55 * camera_confidence)
        geo_feats = geo_feats * (0.25 + 0.75 * geo_reliability)

        gate_logits = self.branch_gate(torch.cat([spatial_feats, geo_feats, geo_input], dim=-1))
        branch_weights = torch.softmax(gate_logits, dim=-1)
        gated_fusion = (
            branch_weights[..., 0:1] * spatial_feats
            + branch_weights[..., 1:2] * geo_feats
        )

        attention_fusion, attention_weights = self.fusion(spatial_feats, geo_feats)
        fused_seq = self.fusion_norm(gated_fusion + attention_fusion)

        temporal_out = self.temporal(fused_seq)
        pooled = temporal_out.mean(dim=1)
        logits = self.classifier(pooled)

        # Confidence-aware damping: on consistently bad camera sequences, avoid
        # overconfident logits while still allowing the model to choose a class.
        sequence_confidence = camera_confidence.mean(dim=1)
        logits = logits * (0.70 + 0.30 * sequence_confidence)

        if return_attention:
            return logits, {
                "attention": attention_weights,
                "branch_weights": branch_weights,
                "camera_confidence": camera_confidence,
            }
        return logits
