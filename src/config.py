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
DEFAULT_ENABLED_MODELS = ["focus", "obstruction"]
LAST_DIR_FILE = PROJECT_ROOT / ".last_directory"

# ---------------------------------------------------------------------------
# Batch / scheduled QC (database-backed workflow)
# ---------------------------------------------------------------------------
DB_PATH = DATA_DIR / "qc.db"
REGISTRY_DIR = MODELS_DIR / "registry"
DATASETS_DIR = DATA_DIR / "datasets"
TRAIN_RUNS_DIR = PROJECT_ROOT / "runs" / "feedback"
# Curated real-world examples dropped in by hand, kept as an escape hatch.
# Layout: <CURATED_DIR>/<role>/<label>/*.jpg — the folder names the class.
CURATED_DIR = DATA_DIR / "curated"

# Viewing copies of imported training images. Rows in `training_examples`
# own the label, so these are organised by provenance, not by class.
# Layout: <IMAGES_DIR>/<role>/<source>/<name>.jpg
IMAGES_DIR = DATA_DIR / "images"

# Scans an annotator kept from the staging folder (see src/preserve.py). One
# copy per image, shared by every role's example row, named by content hash
# so a re-pass of the same file lands on the same copy and a re-capture of
# the same filename does not.
KEPT_DIR = IMAGES_DIR / "kept"

# Longest side kept for a stored viewing copy. Archival TIFFs are far larger
# than anything the UI or the trainer needs (training runs at imgsz=640),
# so only this reduced JPEG is ever copied off the share.
IMPORT_MAX_SIDE = 1500
IMPORT_JPEG_QUALITY = 88

# ---------------------------------------------------------------------------
# Remote GPU box (DGX Spark) — offloads training and/or batch inference.
# Configured via .env; unset = everything runs locally.
# ---------------------------------------------------------------------------
import os as _os

SPARK_IP = (_os.environ.get("SPARK_IP") or "").strip().strip('"')
SPARK_PORT = (_os.environ.get("SPARK_PORT") or "8100").strip()
SPARK_SSH_PORT = int((_os.environ.get("SPARK_SSH_PORT") or "22").strip())
SPARK_USERNAME = (_os.environ.get("SPARK_USERNAME") or "").strip()
SPARK_PASSWORD = (_os.environ.get("SPARK_PASSWORD") or "").strip()
# Explicit URL override (direct reachability or a manually-run tunnel).
# When SPARK_USERNAME/PASSWORD are set, the app opens its own SSH tunnel
# instead and this is ignored.
TRAIN_SERVER_URL = (_os.environ.get("TRAIN_SERVER_URL") or "").strip().rstrip("/")
# Shared secret: TRAIN_SERVER_TOKEN preferred, falling back to SPARK_API_KEY.
TRAIN_SERVER_TOKEN = ((_os.environ.get("TRAIN_SERVER_TOKEN") or "").strip()
                      or (_os.environ.get("SPARK_API_KEY") or "").strip())
REMOTE_INFERENCE = (_os.environ.get("REMOTE_INFERENCE") or "").strip().lower() in ("1", "true", "yes", "on")

# Image types picked up by the batch pipeline
SCAN_EXTENSIONS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}

# Model roles managed by the registry
MODEL_ROLES = ("badscan", "pagetype")

# Binary bad-scan classifier (single simplified QC model).
# Every reviewer verdict becomes a labeled example; retraining always runs
# over the full accumulated set, so flags, missed flags, and corrections
# all feed the next model version.
BAD_SCAN_CLASSES = ["bad_scan", "no_bad_scan"]
BADSCAN_BASE_WEIGHTS = "yolo26n-cls.pt"  # yolo26 nano classification base

# Page-type classifier — built-in classes. Custom classes can be added in
# Settings (custom_page_types); they are classified by the LLM endpoint
# until a retrained local model has learned them.
PAGE_TYPES = ["blank_page", "handwriting", "photograph", "typed_text"]

PAGE_TYPE_DESCRIPTIONS = {
    "handwriting": "page is mostly handwritten text",
    "typed_text": "page is mostly printed or typewritten text",
    "photograph": "page is mostly a photograph or illustration",
    "blank_page": "page is blank or nearly blank",
}


def all_page_types(settings: dict) -> list[str]:
    """Built-in page types plus custom classes configured in Settings."""
    out = list(PAGE_TYPES)
    for raw in settings.get("custom_page_types", "").split(","):
        c = raw.strip().lower().replace(" ", "_")
        if c and c not in out:
            out.append(c)
    return out
PAGETYPE_BASE_WEIGHTS = "yolo11n-cls.pt"
PAGETYPE_SEED_WEIGHTS = MODELS_DIR / "yolo11n.pt"  # copied from gemini-yolo

# A page-type model may also be a region *detector* (e.g. the freeflow
# "typed" project: boxes around typed text, handwriting, photographs…). Its
# boxes are turned into one page type: the mapped class covering the most
# page area wins, and a page with no mapped box at or above the confidence
# floor is blank. Detector classes not listed here (table, stamp, passport…)
# are ignored. A model's own metrics["class_map"] overrides this default.
PAGETYPE_DETECT_CLASS_MAP = {
    "typed": "typed_text",
    "typed_text": "typed_text",
    "handwriting": "handwriting",
    "photograph": "photograph",
    "photo": "photograph",
    "blank": "blank_page",
    "blank_page": "blank_page",
}
PAGETYPE_DETECT_MIN_CONF = 0.25

# Page types that get sent to the transcription endpoint
TRANSCRIBE_PAGE_TYPES = ["handwriting"]

# Default settings written to the DB on first run (editable in the UI)
DEFAULT_SETTINGS = {
    "schedule_enabled": "0",
    "schedule_time": "02:00",           # HH:MM, local time, daily
    "transcription_backend": "off",     # off | sglang | vertex | kraken | nara
    "kraken_model_path": str(MODELS_DIR / "kraken-ppocrv6-medium" / "medium.safetensors"),
    "kraken_device": "mps",             # mps | cpu | auto
    "nara_model_id": "wjbmattingly/nara-qwen-3.5-0.8b-ep1",
    # Blank = run the VLM locally. "spark:<port>" = vLLM on the DGX through
    # the SSH tunnel. A full http(s) URL is used as-is.
    "nara_server_url": "",
    # Pages transcribed at once against a server backend (vLLM/SGLang/
    # Vertex batch concurrent requests). Local engines always run one page
    # at a time regardless of this value.
    "transcribe_concurrency": "6",
    "sglang_base_url": "http://localhost:30000",
    "sglang_model": "default",
    "sglang_api_key": "",
    "vertex_project": "cultural-heritage-gemini",
    "vertex_location": "global",
    "vertex_model": "gemini-3.1-flash-lite",
    "transcribe_page_types": ",".join(TRANSCRIBE_PAGE_TYPES),
    "custom_page_types": "",
    "retrain_epochs": "50",
    "retrain_auto_activate": "1",
    # Dataset composition. Every bad scan and every past model mistake is
    # always included; clean scans are sampled, spread evenly across
    # projects, to this multiple of the positive count.
    "neg_sample_ratio": "1.5",
    "neg_sample_seed": "",          # blank = a fresh sample per build
    # The watched folder is a staging area whose files move on within days.
    # Only scans the annotator says the model got wrong are copied into the
    # app's own store and kept as training data. Set to 1 to also keep the
    # ones where annotator and model agreed.
    "preserve_agreements": "0",
}

# QC subdirectory names (created under watched directory)
QC_SUBDIR = "qc"
QC_CACHE_DIR = "cache"
QC_FAILED_TIFFS_DIR = "failed-tiffs"
QC_TRUE_POSITIVE_DIR = "true-positive"
QC_FALSE_POSITIVE_DIR = "false-positive"
QC_FALSE_NEGATIVE_DIR = "false-negative"

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
for dir_path in [DATA_DIR, RAW_IMAGES_DIR, AUGMENTED_DIR, MODELS_DIR, MANIFESTS_DIR,
                 REGISTRY_DIR, DATASETS_DIR, IMAGES_DIR, KEPT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

