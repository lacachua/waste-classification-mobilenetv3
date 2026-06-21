from pathlib import Path

# Dữ liệu và khả năng lặp lại
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
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

# Baseline: chỉ train classifier
BASELINE_EPOCHS = 3
BASELINE_LEARNING_RATE = 1e-3

# Fine-tuning: mở 3 block cuối của backbone
FINE_TUNE_EPOCHS = 5
FINE_TUNE_BACKBONE_LR = 1e-5
FINE_TUNE_CLASSIFIER_LR = 1e-4

WEIGHT_DECAY = 1e-4

PROJECT_DIR = Path("/kaggle/working/waste-classification-mobilenetv3")
OUTPUT_DIR = PROJECT_DIR / "outputs"
BASELINE_MODEL_PATH = OUTPUT_DIR / "best_baseline_model.pth"
FINE_TUNED_MODEL_PATH = OUTPUT_DIR / "best_finetuned_model.pth"
