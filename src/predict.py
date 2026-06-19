
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import mobilenet_v3_small

from src.config import CLASS_NAMES, OUTPUT_DIR
from src.dataset import create_transforms


DEFAULT_MODEL_PATH = (
    OUTPUT_DIR / "best_finetuned_model.pth"
)


def load_trained_model(
    checkpoint_path=DEFAULT_MODEL_PATH,
    device=None,
):
    """
    Nạp MobileNetV3 đã fine-tune từ checkpoint.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model: {checkpoint_path}"
        )

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    class_names = checkpoint.get(
        "class_names",
        CLASS_NAMES,
    )

    # Chỉ cần tạo lại kiến trúc.
    # Không cần tải pretrained weights lần nữa.
    model = mobilenet_v3_small(weights=None)

    input_features = model.classifier[3].in_features

    model.classifier[3] = nn.Linear(
        input_features,
        len(class_names),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return (
        model,
        class_names,
        device,
        checkpoint,
    )


def predict_image(
    image_path,
    checkpoint_path=DEFAULT_MODEL_PATH,
    top_k=3,
):
    """
    Dự đoán top-k lớp của một ảnh.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ảnh: {image_path}"
        )

    model, class_names, device, checkpoint = (
        load_trained_model(
            checkpoint_path=checkpoint_path
        )
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    _, evaluation_transform = create_transforms()

    input_tensor = (
        evaluation_transform(image)
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():
        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )[0]

    top_k = min(
        top_k,
        len(class_names),
    )

    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=top_k,
    )

    results = []

    for probability, class_index in zip(
        top_probabilities.cpu().tolist(),
        top_indices.cpu().tolist(),
    ):
        results.append(
            {
                "class_name": class_names[class_index],
                "probability": probability,
            }
        )

    return image, results, checkpoint
