"""State management for the SensorMCP server (fixed data paths, EN comments)."""

import os
from typing import Any, Dict
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────
# 0. Environment variables
# ──────────────────────────────────────────────────────────────
load_dotenv()  # read variables from .env, if present

# ──────────────────────────────────────────────────────────────
# 1. Fixed directory layout
# ──────────────────────────────────────────────────────────────
# Project root = directory containing this file
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_DIR           = os.path.join(BASE_DIR, "data")
RAW_IMAGES_DIR     = os.path.join(DATA_DIR, "raw_images")       # original/unlabeled images
LABELED_IMAGES_DIR = os.path.join(DATA_DIR, "labeled_images")   # auto-labeled datasets
MODELS_DIR         = os.path.join(DATA_DIR, "models")           # trained weight files

# Ensure the directories exist
for d in (DATA_DIR, RAW_IMAGES_DIR, LABELED_IMAGES_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 2. Central mutable state
# ──────────────────────────────────────────────────────────────
STATE: Dict[str, Any] = {
    "base_model": None,
    "target_model": None,
    "ontology": None,
    "input_folder": RAW_IMAGES_DIR,
    "output_folder": LABELED_IMAGES_DIR,
    "labeled_dataset": LABELED_IMAGES_DIR,
    "model_output_folder": MODELS_DIR,
    "trained_model_path": None,
    "last_predictions": None,
    "unsplash_api_key": os.environ.get("UNSPLASH_API_KEY", ""),
}

# ──────────────────────────────────────────────────────────────
# 3. Constants
# ──────────────────────────────────────────────────────────────
SUPPORTED_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
SUPPORTED_BASE_MODELS  = ["grounded_sam"]
SUPPORTED_TARGET_MODELS = [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"
]
MAX_IMAGES_PER_PAGE = 30
MAX_WAIT_TIME = 1
SHORT_WAIT_TIME = 0.2
HTTP_OK = 200

__all__ = [
    "STATE",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "SUPPORTED_BASE_MODELS",
    "SUPPORTED_TARGET_MODELS",
    "RAW_IMAGES_DIR",
    "LABELED_IMAGES_DIR",
    "MODELS_DIR",
]
