"""Configuration settings for the QC pipeline."""

from pathlib import Path

# HuggingFace model repository
HF_REPO_ID = "yale-cultural-heritage/yale-scanning-qc"

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw"
AUGMENTED_DIR = DATA_DIR / "augmented"
MODELS_DIR = PROJECT_ROOT / "models"
MANIFESTS_DIR = PROJECT_ROOT / "manifests"

# Download settings
DEFAULT_NUM_IMAGES = 20
MAX_RESOLUTION = 1500
DOWNLOAD_THREADS = 20
RANDOM_SEED = 1337  # =)
DEFAULT_MANIFEST_FILE = Path("/Users/wjm55/data/manifests.json")

# Augmentation settings
AUGMENTATION_MULTIPLIER = 5  # Number of augmented versions per original image
BLACK_BACKGROUND = True  # Place images on black background (scanner simulation)
BORDER_SIZE = 150  # Border size in pixels around the image
BORDER_VARIATION = 25  # Random variation in border position (+/- pixels)

# Model settings
YOLO_MODEL_SIZE = "n"  # n, s, m, l, x
EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 640

# QC scanning workflow defaults
DEFAULT_ENABLED_MODELS = ["focus", "dust", "obstruction"]
LAST_DIR_FILE = PROJECT_ROOT / ".last_directory"

# QC subdirectory names (created under watched directory)
QC_SUBDIR = "qc"
QC_CACHE_DIR = "cache"
QC_FAILED_TIFFS_DIR = "failed-tiffs"
QC_TRUE_POSITIVE_DIR = "true-positive"
QC_FALSE_POSITIVE_DIR = "false-positive"

# JPEG conversion settings (for TIFF -> JPEG cache)
JPEG_QUALITY = 85
JPEG_MAX_SIDE = 2500

# QC Categories
QC_CATEGORIES = {
    "focus": {
        "type": "classification",
        "classes": ["in_focus", "out_of_focus"],
        "model": "yolo26-focus.pt",
    },
    "obstruction": {
        "type": "detection",
        "classes": ["hand", "finger", "shadow", "object"],
        "model": "yolo26-obstruction.pt",
    },
    "orientation": {
        "type": "classification",
        "classes": ["correct", "rotate_90_cw", "rotate_180", "rotate_90_ccw"],
        "model": "yolo26-orientation.pt",
    },
    "dust": {
        "type": "detection",
        "classes": ["dust_speck", "scratch", "smudge", "fiber"],
        "model": "yolo26-dust.pt",
    },
    "skew": {
        "type": "algorithmic",
        "description": "Hough transform + perspective correction",
    },
}

# Detection visualization colors (BGR format for OpenCV)
DETECTION_COLORS = {
    # Obstruction colors
    "hand": (0, 100, 255),      # Orange
    "finger": (0, 165, 255),    # Light orange
    "shadow": (128, 128, 128),  # Gray
    "object": (0, 0, 255),      # Red
    # Dust colors
    "dust_speck": (255, 0, 255),  # Magenta
    "scratch": (255, 255, 0),     # Cyan
    "smudge": (0, 255, 255),      # Yellow
    "fiber": (255, 0, 0),         # Blue
}

# Create directories
for dir_path in [DATA_DIR, RAW_IMAGES_DIR, AUGMENTED_DIR, MODELS_DIR, MANIFESTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

