
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    SEED,
    CLASS_NAMES,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def find_dataset_root(input_dir: str | Path) -> Path:
    """
    Tìm thư mục trực tiếp chứa đủ 6 thư mục lớp rác.
    Có thể truyền thẳng /kaggle/input.
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Không tồn tại đường dẫn: {input_dir}"
        )

    expected_classes = set(CLASS_NAMES)

    directories = [
        input_dir,
        *[
            path
            for path in input_dir.rglob("*")
            if path.is_dir()
        ],
    ]

    for directory in directories:
        child_names = {
            child.name.lower().strip()
            for child in directory.iterdir()
            if child.is_dir()
        }

        if expected_classes.issubset(child_names):
            return directory

    raise FileNotFoundError(
        "Không tìm thấy thư mục chứa đủ các lớp: "
        + ", ".join(CLASS_NAMES)
    )


def build_dataframe(dataset_root: str | Path) -> pd.DataFrame:
    """
    Tạo bảng gồm đường dẫn ảnh, tên lớp và nhãn số.
    """
    dataset_root = Path(dataset_root)

    child_directories = {
        child.name.lower().strip(): child
        for child in dataset_root.iterdir()
        if child.is_dir()
    }

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(CLASS_NAMES)
    }

    records = []

    for class_name in CLASS_NAMES:
        class_directory = child_directories.get(class_name)

        if class_directory is None:
            raise FileNotFoundError(
                f"Không tìm thấy thư mục lớp: {class_name}"
            )

        for image_path in class_directory.rglob("*"):
            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                records.append(
                    {
                        "image_path": str(image_path),
                        "class_name": class_name,
                        "label": class_to_index[class_name],
                    }
                )

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        raise RuntimeError(
            "Không tìm thấy ảnh hợp lệ trong dataset."
        )

    return dataframe


def split_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chia dữ liệu thành:
    - 70% train
    - 15% validation
    - 15% test
    """
    train_df, remaining_df = train_test_split(
        dataframe,
        test_size=0.30,
        random_state=SEED,
        stratify=dataframe["label"],
    )

    val_df, test_df = train_test_split(
        remaining_df,
        test_size=0.50,
        random_state=SEED,
        stratify=remaining_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class WasteDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform=None,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]

        image = Image.open(
            row["image_path"]
        ).convert("RGB")

        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def create_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.75, 1.0),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, evaluation_transform


def create_dataloaders(input_dir: str | Path):
    dataset_root = find_dataset_root(input_dir)
    dataframe = build_dataframe(dataset_root)

    train_df, val_df, test_df = split_dataframe(
        dataframe
    )

    train_transform, evaluation_transform = (
        create_transforms()
    )

    train_dataset = WasteDataset(
        train_df,
        transform=train_transform,
    )

    val_dataset = WasteDataset(
        val_df,
        transform=evaluation_transform,
    )

    test_dataset = WasteDataset(
        test_df,
        transform=evaluation_transform,
    )

    common_loader_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": NUM_WORKERS > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **common_loader_options,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_loader_options,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_options,
    )

    print("Dataset root:", dataset_root)
    print("Tổng số ảnh:", len(dataframe))
    print("Train:", len(train_dataset))
    print("Validation:", len(val_dataset))
    print("Test:", len(test_dataset))

    print("\nSố ảnh mỗi lớp:")
    print(
        dataframe["class_name"]
        .value_counts()
        .reindex(CLASS_NAMES)
    )

    return train_loader, val_loader, test_loader
