import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from src.config import (
    BASELINE_MODEL_PATH,
    CLASS_NAMES,
    FINE_TUNE_BACKBONE_LR,
    FINE_TUNE_CLASSIFIER_LR,
    FINE_TUNE_EPOCHS,
    FINE_TUNED_MODEL_PATH,
    OUTPUT_DIR,
    SEED,
    WEIGHT_DECAY,
)
from src.dataset import create_dataloaders
from src.model import create_model
from src.utils import seed_everything


def calculate_sqrt_class_weights(train_loader, device):
    """Class weight dạng căn bậc hai để cân bằng nhẹ hơn baseline."""
    dataframe = train_loader.dataset.dataframe
    class_counts = (
        dataframe["label"]
        .value_counts()
        .sort_index()
        .reindex(range(len(CLASS_NAMES)))
        .values
        .astype(np.float32)
    )
    original_weights = len(dataframe) / (len(CLASS_NAMES) * class_counts)
    sqrt_weights = np.sqrt(original_weights)
    weights_tensor = torch.tensor(sqrt_weights, dtype=torch.float32, device=device)

    print("Class weights dùng khi fine-tune:")
    for name, count, weight in zip(CLASS_NAMES, class_counts, sqrt_weights):
        print(f"- {name:10s}: {int(count):4d} ảnh | weight = {weight:.4f}")

    return weights_tensor


def load_baseline_for_fine_tuning(device):
    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy baseline: {BASELINE_MODEL_PATH}")

    checkpoint = torch.load(BASELINE_MODEL_PATH, map_location=device, weights_only=True)

    # pretrained=False vì checkpoint đã chứa toàn bộ trọng số cần thiết.
    model = create_model(freeze_features=True, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for block in model.features[-3:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    model = model.to(device)
    print("Đã nạp baseline từ epoch:", checkpoint["epoch"])
    return model


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()

    # Các block đầu vẫn đóng băng cả gradient lẫn BatchNorm statistics.
    for block in model.features[:-3]:
        block.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    progress_bar = tqdm(data_loader, desc="Fine-tuning", leave=False)

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        predictions = outputs.argmax(dim=1)
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{running_loss / total_samples:.4f}",
            accuracy=f"{running_correct / total_samples:.4f}",
        )

    return running_loss / total_samples, running_correct / total_samples


def validate(model, data_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0
    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (predictions == labels).sum().item()
            total_samples += batch_size
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return running_loss / total_samples, running_correct / total_samples, macro_f1


def save_checkpoint(model, epoch, validation_loss, validation_accuracy, validation_macro_f1):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "stage": "fine_tuning",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "validation_macro_f1": validation_macro_f1,
        "class_names": CLASS_NAMES,
    }
    torch.save(checkpoint, FINE_TUNED_MODEL_PATH)
    print("Đã lưu fine-tuned model tốt nhất:", FINE_TUNED_MODEL_PATH)


def fine_tune_model(data_dir, epochs: int = FINE_TUNE_EPOCHS):
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Thiết bị:", device)

    train_loader, val_loader, test_loader = create_dataloaders(data_dir)
    model = load_baseline_for_fine_tuning(device)

    class_weights = calculate_sqrt_class_weights(train_loader, device)
    train_criterion = nn.CrossEntropyLoss(weight=class_weights)
    validation_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {"params": model.features[-3:].parameters(), "lr": FINE_TUNE_BACKBONE_LR},
            {"params": model.classifier.parameters(), "lr": FINE_TUNE_CLASSIFIER_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
    )

    best_validation_macro_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, train_criterion, optimizer, device
        )
        validation_loss, validation_accuracy, validation_macro_f1 = validate(
            model, val_loader, validation_criterion, device
        )
        scheduler.step(validation_loss)
        backbone_lr = optimizer.param_groups[0]["lr"]
        classifier_lr = optimizer.param_groups[1]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "validation_macro_f1": validation_macro_f1,
                "backbone_lr": backbone_lr,
                "classifier_lr": classifier_lr,
            }
        )

        print(f"\nFine-tune epoch {epoch}/{epochs}")
        print(f"Train loss: {train_loss:.4f} | Train accuracy: {train_accuracy:.4f}")
        print(
            f"Validation loss: {validation_loss:.4f} | "
            f"Validation accuracy: {validation_accuracy:.4f} | "
            f"Macro F1: {validation_macro_f1:.4f}"
        )
        print(f"Backbone LR: {backbone_lr:.2e} | Classifier LR: {classifier_lr:.2e}")
        print(f"Thời gian: {time.time() - start_time:.1f} giây")

        if validation_macro_f1 > best_validation_macro_f1:
            best_validation_macro_f1 = validation_macro_f1
            save_checkpoint(
                model,
                epoch,
                validation_loss,
                validation_accuracy,
                validation_macro_f1,
            )

    # Nạp lại checkpoint tốt nhất thay vì giữ epoch cuối.
    checkpoint = torch.load(FINE_TUNED_MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\nFine-tuning hoàn tất.")
    print("Epoch tốt nhất:", checkpoint["epoch"])
    print("Validation Macro F1 tốt nhất:", f"{checkpoint['validation_macro_f1']:.4f}")

    return model, history, test_loader
