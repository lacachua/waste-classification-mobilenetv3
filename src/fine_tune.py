
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm.auto import tqdm

from src.config import CLASS_NAMES, OUTPUT_DIR
from src.dataset import create_dataloaders
from src.model import create_model


BASELINE_MODEL_PATH = OUTPUT_DIR / "best_model.pth"
FINE_TUNED_MODEL_PATH = (
    OUTPUT_DIR / "best_finetuned_model.pth"
)


def calculate_sqrt_class_weights(
    train_loader,
    device,
):
    """
    Tính class weight bằng căn bậc hai.

    Cách này vẫn ưu tiên lớp ít ảnh nhưng nhẹ hơn
    class weight cũ, tránh model dự đoán quá nhiều
    ảnh thành lớp trash.
    """

    dataframe = train_loader.dataset.dataframe

    class_counts = (
        dataframe["label"]
        .value_counts()
        .sort_index()
        .reindex(range(len(CLASS_NAMES)))
        .values
        .astype(np.float32)
    )

    number_of_samples = len(dataframe)
    number_of_classes = len(CLASS_NAMES)

    original_weights = (
        number_of_samples
        / (number_of_classes * class_counts)
    )

    sqrt_weights = np.sqrt(original_weights)

    weights_tensor = torch.tensor(
        sqrt_weights,
        dtype=torch.float32,
        device=device,
    )

    print("Class weights dùng khi fine-tune:")

    for class_name, count, weight in zip(
        CLASS_NAMES,
        class_counts,
        sqrt_weights,
    ):
        print(
            f"- {class_name:10s}: "
            f"{int(count):4d} ảnh | "
            f"weight = {weight:.4f}"
        )

    return weights_tensor


def load_baseline_for_fine_tuning(device):
    """
    Nạp best_model.pth và mở khóa 3 block cuối.
    """

    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy baseline model: "
            f"{BASELINE_MODEL_PATH}"
        )

    checkpoint = torch.load(
        BASELINE_MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    # Tạo đúng kiến trúc MobileNetV3 gồm 6 lớp
    model = create_model(
        freeze_features=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Đóng băng lại toàn bộ phần features
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Mở khóa 3 block cuối của features
    for block in model.features[-3:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    # Classifier tiếp tục được train
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    model = model.to(device)

    print(
        "Đã nạp baseline từ epoch:",
        checkpoint["epoch"],
    )

    print(
        "Baseline validation accuracy:",
        f"{checkpoint['validation_accuracy']:.2%}",
    )

    return model


def train_one_fine_tune_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    # Giữ các block đầu ở chế độ eval.
    # Chỉ 3 block cuối và classifier thực sự thích nghi.
    for block in model.features[:-3]:
        block.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    progress_bar = tqdm(
        data_loader,
        desc="Fine-tuning",
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

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
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


def validate_fine_tuned_model(
    model,
    data_loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        for images, labels in data_loader:
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

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

    validation_loss = (
        running_loss / total_samples
    )

    validation_accuracy = (
        running_correct / total_samples
    )

    validation_macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return (
        validation_loss,
        validation_accuracy,
        validation_macro_f1,
    )


def save_fine_tuned_checkpoint(
    model,
    epoch,
    validation_loss,
    validation_accuracy,
    validation_macro_f1,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "stage": "fine_tuning",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "validation_loss": validation_loss,
        "validation_accuracy": validation_accuracy,
        "validation_macro_f1": validation_macro_f1,
        "class_names": CLASS_NAMES,
    }

    torch.save(
        checkpoint,
        FINE_TUNED_MODEL_PATH,
    )

    print(
        "Đã lưu fine-tuned model tốt nhất:",
        FINE_TUNED_MODEL_PATH,
    )


def fine_tune_model(
    data_dir,
    epochs=5,
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Thiết bị:", device)

    train_loader, val_loader, test_loader = (
        create_dataloaders(data_dir)
    )

    model = load_baseline_for_fine_tuning(
        device
    )

    class_weights = (
        calculate_sqrt_class_weights(
            train_loader,
            device,
        )
    )

    # Train có class weight
    train_criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # Validation dùng loss bình thường
    validation_criterion = (
        nn.CrossEntropyLoss()
    )

    # Learning rate rất nhỏ cho backbone
    # và lớn hơn một chút cho classifier.
    optimizer = torch.optim.AdamW(
        [
            {
                "params": (
                    model.features[-3:].parameters()
                ),
                "lr": 1e-5,
            },
            {
                "params": (
                    model.classifier.parameters()
                ),
                "lr": 1e-4,
            },
        ],
        weight_decay=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=1,
        )
    )

    best_validation_macro_f1 = -1.0
    history = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_accuracy = (
            train_one_fine_tune_epoch(
                model=model,
                data_loader=train_loader,
                criterion=train_criterion,
                optimizer=optimizer,
                device=device,
            )
        )

        (
            validation_loss,
            validation_accuracy,
            validation_macro_f1,
        ) = validate_fine_tuned_model(
            model=model,
            data_loader=val_loader,
            criterion=validation_criterion,
            device=device,
        )

        scheduler.step(validation_loss)

        elapsed_time = time.time() - start_time

        backbone_lr = (
            optimizer.param_groups[0]["lr"]
        )

        classifier_lr = (
            optimizer.param_groups[1]["lr"]
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": (
                    validation_loss
                ),
                "validation_accuracy": (
                    validation_accuracy
                ),
                "validation_macro_f1": (
                    validation_macro_f1
                ),
                "backbone_lr": backbone_lr,
                "classifier_lr": classifier_lr,
            }
        )

        print(f"\nFine-tune epoch {epoch}/{epochs}")

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
            f"Validation Macro F1: "
            f"{validation_macro_f1:.4f}"
        )

        print(
            f"Backbone LR: {backbone_lr:.2e} | "
            f"Classifier LR: {classifier_lr:.2e}"
        )

        print(
            f"Thời gian: {elapsed_time:.1f} giây"
        )

        # Chọn model theo Macro F1,
        # phù hợp hơn với dữ liệu mất cân bằng.
        if (
            validation_macro_f1
            > best_validation_macro_f1
        ):
            best_validation_macro_f1 = (
                validation_macro_f1
            )

            save_fine_tuned_checkpoint(
                model=model,
                epoch=epoch,
                validation_loss=validation_loss,
                validation_accuracy=(
                    validation_accuracy
                ),
                validation_macro_f1=(
                    validation_macro_f1
                ),
            )

    print("\nFine-tuning hoàn tất.")

    print(
        "Validation Macro F1 tốt nhất:",
        f"{best_validation_macro_f1:.4f}",
    )

    return model, history, test_loader
