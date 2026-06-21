from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import mobilenet_v3_small

from src.config import CLASS_NAMES, FINE_TUNED_MODEL_PATH
from src.dataset import create_transforms


def load_trained_model(checkpoint_path=FINE_TUNED_MODEL_PATH, device=None):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    class_names = checkpoint.get("class_names", CLASS_NAMES)

    model = mobilenet_v3_small(weights=None)
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    _, evaluation_transform = create_transforms()
    return model, class_names, device, evaluation_transform, checkpoint


def predict_pil_image(
    image: Image.Image,
    model,
    class_names,
    device,
    evaluation_transform,
    top_k: int = 3,
):
    image = image.convert("RGB")
    input_tensor = evaluation_transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_k = min(top_k, len(class_names))
    top_probabilities, top_indices = torch.topk(probabilities, k=top_k)

    return [
        {
            "class_name": class_names[class_index],
            "probability": probability,
        }
        for probability, class_index in zip(
            top_probabilities.cpu().tolist(),
            top_indices.cpu().tolist(),
        )
    ]


def predict_image_path(image_path, loaded_model=None, top_k: int = 3):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    if loaded_model is None:
        loaded_model = load_trained_model()

    model, class_names, device, evaluation_transform, checkpoint = loaded_model
    image = Image.open(image_path).convert("RGB")
    results = predict_pil_image(
        image,
        model,
        class_names,
        device,
        evaluation_transform,
        top_k=top_k,
    )
    return image, results, checkpoint
