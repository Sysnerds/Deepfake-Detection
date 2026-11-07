from pathlib import Path


# Project root
PROJECT_DIR = Path(__file__).resolve().parent.parent


# Dataset location
DATASET_DIR = Path(r"C:\Users\mda16\Downloads\Dataset")

DATA_ROOT = DATASET_DIR / "real_vs_fake" / "real-vs-fake"


# Dataset splits
TRAIN_DIR = DATA_ROOT / "train"
VALID_DIR = DATA_ROOT / "valid"
TEST_DIR = DATA_ROOT / "test"


# CSV files
TRAIN_CSV = DATASET_DIR / "train.csv"
VALID_CSV = DATASET_DIR / "valid.csv"
TEST_CSV = DATASET_DIR / "test.csv"


# Project output directories
MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"


# Image settings
IMAGE_SIZE = 128