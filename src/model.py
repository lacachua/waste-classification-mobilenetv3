
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)

from src.config import CLASS_NAMES


def create_model(freeze_features: bool = True) -> nn.Module:
    """
    Tạo MobileNetV3-Small cho bài toán phân loại rác.
    """

    weights = MobileNet_V3_Small_Weights.DEFAULT

    model = mobilenet_v3_small(weights=weights)

    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    input_features = model.classifier[3].in_features
    number_of_classes = len(CLASS_NAMES)

    model.classifier[3] = nn.Linear(
        input_features,
        number_of_classes,
    )

    return model
