import logging

import torch
import torch.nn as nn
from torchvision import models


def _safe_weights(weights_enum, pretrained):
    if not pretrained:
        return None
    try:
        return weights_enum.DEFAULT
    except Exception:
        return None


class CNNFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone="mobilenet_v3_small",
        feature_dim=512,
        num_classes=4,
        dropout=0.30,
        pretrained=True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.backbone, in_features = self._build_backbone(backbone, pretrained)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def _build_backbone(self, backbone, pretrained):
        try:
            if backbone == "mobilenet_v3_small":
                weights = _safe_weights(models.MobileNet_V3_Small_Weights, pretrained)
                model = models.mobilenet_v3_small(weights=weights)
                return model.features, model.classifier[0].in_features

            if backbone == "mobilenet_v3_large":
                weights = _safe_weights(models.MobileNet_V3_Large_Weights, pretrained)
                model = models.mobilenet_v3_large(weights=weights)
                return model.features, model.classifier[0].in_features

            if backbone == "efficientnet_b0":
                weights = _safe_weights(models.EfficientNet_B0_Weights, pretrained)
                model = models.efficientnet_b0(weights=weights)
                return model.features, model.classifier[1].in_features

            if backbone == "resnet18":
                weights = _safe_weights(models.ResNet18_Weights, pretrained)
                model = models.resnet18(weights=weights)
                modules = list(model.children())[:-2]
                return nn.Sequential(*modules), model.fc.in_features
        except Exception as exc:
            if pretrained:
                logging.warning("Khong load duoc pretrained weights (%s). Dung random init.", exc)
                return self._build_backbone(backbone, pretrained=False)
            raise

        raise ValueError(
            "backbone phai la mobilenet_v3_small, mobilenet_v3_large, "
            "efficientnet_b0 hoac resnet18"
        )

    def forward(self, x, return_features=False):
        raw = self.backbone(x)
        pooled = self.pool(raw)
        features = self.projection(pooled)
        logits = self.classifier(features)
        if return_features:
            return features, logits
        return logits

    @torch.no_grad()
    def extract_features(self, x):
        self.eval()
        features, _ = self.forward(x, return_features=True)
        return features
