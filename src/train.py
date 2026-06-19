
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.config import (
    CLASS_NAMES,
    EPOCHS,
    LEARNING_RATE,
    OUTPUT_DIR,
    BEST_MODEL_PATH,
)
from src.dataset import create_dataloaders
from src.model import create_model


def calculate_class_weights(train_loader, device):
    """
    Tạo trọng số cho từng lớp để hạn chế ảnh hưởng
    của việc dataset bị mất cân bằng.
    """

    dataframe = train_loader.dataset.dataframe

    class_counts = (
        dataframe["label"]
        .value_counts()
        .sort_index()
        .reindex(range(len(CLASS_NAMES)))
    )

    number_of_samples = len(dataframe)
    number_of_classes = len(CLASS_NAMES)

    weights = (
        number_of_samples
        / (number_of_classes * class_counts.values)
    )

    weights_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )

    print("Class weights:")

    for class_name, count, weight in zip(
        CLASS_NAMES,
        class_counts.values,
        weights,
    ):
        print(
            f"- {class_name:10s}: "
            f"{count:4d} ảnh | weight = {weight:.4f}"
        )

    return weights_tensor


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        data_loader,
        desc="Training",
        leave=False,
    )

    for images, labels in progress_bar:
        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        # Xóa gradient của batch trước
        optimizer.zero_grad()

        # Dự đoán
        outputs = model(images)

        # Tính loss
        loss = criterion(outputs, labels)

        # Lan truyền ngược
        loss.backward()

        # Cập nhật trọng số
        optimizer.step()

        predictions = outputs.argmax(dim=1)

        batch_size = images.size(0)

        running_loss += loss.item() * batch_size
        running_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{running_loss / total_samples:.4f}",
            accuracy=(
                f"{running_correct / total_samples:.4f}"
            ),
        )

    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_correct / total_samples

    return epoch_loss, epoch_accuracy


def validate_one_epoch(
    model,
    data_loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        data_loader,
        desc="Validation",
        leave=False,
    )

    with torch.inference_mode():
        for images, labels in progress_bar:
            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            batch_size = images.size(0)

            running_loss += loss.item() * batch_size
            running_correct += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

            progress_bar.set_postfix(
                loss=(
                    f"{running_loss / total_samples:.4f}"
                ),
                accuracy=(
                    f"{running_correct / total_samples:.4f}"
                ),
            )

    epoch_loss = running_loss / total_samples
    epoch_accuracy = running_correct / total_samples

    return epoch_loss, epoch_accuracy


def save_checkpoint(
    model,
    epoch,
    validation_loss,
    validation_accuracy,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "class_names": CLASS_NAMES,
    }

    torch.save(
        checkpoint,
        BEST_MODEL_PATH,
    )

    print(
        "Đã lưu model tốt nhất:",
        BEST_MODEL_PATH,
    )


def train_model(
    data_dir,
    epochs=None,
):
    """
    Hàm chính dùng để train MobileNetV3.
    """

    if epochs is None:
        epochs = EPOCHS

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Thiết bị:", device)

    train_loader, val_loader, test_loader = (
        create_dataloaders(data_dir)
    )

    model = create_model(
        freeze_features=True
    ).to(device)

    class_weights = calculate_class_weights(
        train_loader,
        device,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    trainable_parameters = filter(
        lambda parameter: parameter.requires_grad,
        model.parameters(),
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )
    )

    best_validation_accuracy = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_accuracy = (
            train_one_epoch(
                model=model,
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )
        )

        validation_loss, validation_accuracy = (
            validate_one_epoch(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
            )
        )

        scheduler.step(validation_loss)

        elapsed_time = time.time() - start_time

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": (
                    validation_accuracy
                ),
                "learning_rate": (
                    current_learning_rate
                ),
            }
        )

        print(
            f"\nEpoch {epoch}/{epochs}"
        )

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.4f} | "
            f"Validation accuracy: "
            f"{validation_accuracy:.4f}"
        )

        print(
            f"Learning rate: "
            f"{current_learning_rate:.2e}"
        )

        print(
            f"Thời gian: {elapsed_time:.1f} giây"
        )

        if (
            validation_accuracy
            > best_validation_accuracy
        ):
            best_validation_accuracy = (
                validation_accuracy
            )

            save_checkpoint(
                model=model,
                epoch=epoch,
                validation_loss=validation_loss,
                validation_accuracy=(
                    validation_accuracy
                ),
            )

    print("\nTrain hoàn tất.")

    print(
        "Validation accuracy tốt nhất:",
        f"{best_validation_accuracy:.4f}",
    )

    return (
        model,
        history,
        test_loader,
    )
