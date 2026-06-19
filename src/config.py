
from pathlib import Path

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
SEED = 42
NUM_WORKERS = 2

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
]

PROJECT_DIR = Path(
    "/kaggle/working/waste-classification-mobilenetv3"
)
OUTPUT_DIR = PROJECT_DIR / "outputs"
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pth"
