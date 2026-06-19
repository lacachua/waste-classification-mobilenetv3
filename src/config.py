# src/config.py

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

OUTPUT_DIR = Path("/kaggle/working/outputs")
BEST_MODEL_PATH = OUTPUT_DIR / "best_model.pth"