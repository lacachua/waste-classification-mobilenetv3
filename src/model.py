import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from src.config import CLASS_NAMES


def create_model(
    freeze_features: bool = True,
    pretrained: bool = True,
) -> nn.Module:
    """Tạo MobileNetV3-Small với đầu ra bằng số lớp của dataset."""
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)

    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, len(CLASS_NAMES))

    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    return model
