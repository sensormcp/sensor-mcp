"""Main MCP server implementation for Autodistill (fixed paths, EN comments)."""

import glob
import json
import os
import shutil
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount

from image_utils import (
    download_images,
    get_image_files,
    search_unsplash_images,
    validate_unsplash_params,
)
from models import (
    create_ontology,
    initialize_base_model,
    initialize_target_model,
    train_target_model,
)
from state import (
    STATE,
    SUPPORTED_BASE_MODELS,
    SUPPORTED_TARGET_MODELS,
    RAW_IMAGES_DIR,
    LABELED_IMAGES_DIR,
    MODELS_DIR,
)

# ──────────────────────────────────────────────────────────────
# 0. Environment
# ──────────────────────────────────────────────────────────────
load_dotenv()

# ──────────────────────────────────────────────────────────────
# 1. MCP server
# ──────────────────────────────────────────────────────────────
mcp = FastMCP("autodistill-server")

# ──────────────────────────────────────────────────────────────
# 2. Utility: list supported models + fixed paths
# ──────────────────────────────────────────────────────────────
@mcp.tool()
async def list_available_models() -> str:
    """Return supported base/target models and directory layout."""
    data = {
        "detection": {
            "base": SUPPORTED_BASE_MODELS,
            "target": SUPPORTED_TARGET_MODELS,
        },
        "paths": {
            "raw_images_dir": RAW_IMAGES_DIR,
            "labeled_images_dir": LABELED_IMAGES_DIR,
            "models_dir": MODELS_DIR,
        },
    }
    return json.dumps(data, indent=2)

# ──────────────────────────────────────────────────────────────
# 3. Import existing images into RAW_IMAGES_DIR
# ──────────────────────────────────────────────────────────────
@mcp.tool()
async def import_images_from_folder(folder_path: str) -> str:
    """Copy user-provided images into the fixed raw-images directory."""
    if not os.path.isdir(folder_path):
        return f"Error: '{folder_path}' is not a directory."

    files = get_image_files(folder_path)
    if not files:
        return f"Error: no image files found in '{folder_path}'."

    for f in files:
        try:
            shutil.copy2(f, RAW_IMAGES_DIR)
        except Exception as e:
            return f"Error copying '{f}': {e}"

    STATE["input_folder"] = RAW_IMAGES_DIR
    return f"Imported {len(files)} images into '{RAW_IMAGES_DIR}'."

# ──────────────────────────────────────────────────────────────
# 4. Ontology / model selection
# ──────────────────────────────────────────────────────────────
@mcp.tool()
async def define_ontology(objects_list: str) -> str:
    ok, msg = create_ontology(objects_list)
    return msg

@mcp.tool()
async def set_base_model(model_name: str) -> str:
    ok, msg = initialize_base_model(model_name)
    return msg

@mcp.tool()
async def set_target_model(model_name: str) -> str:
    ok, msg = initialize_target_model(model_name)
    return msg

# ──────────────────────────────────────────────────────────────
# 5. Auto-label images
# ──────────────────────────────────────────────────────────────
@mcp.tool()
async def label_images() -> str:
    """Run the base model over RAW_IMAGES_DIR and save labels."""
    if STATE["base_model"] is None:
        return "Error: base model not set."

    images = get_image_files(RAW_IMAGES_DIR)
    if not images:
        return f"Error: no images found in '{RAW_IMAGES_DIR}'."

    try:
        os.makedirs(LABELED_IMAGES_DIR, exist_ok=True)
        STATE["base_model"].label(
            input_folder=RAW_IMAGES_DIR,
            output_folder=LABELED_IMAGES_DIR,
        )
        data_yaml = os.path.join(LABELED_IMAGES_DIR, "data.yaml")
        if os.path.exists(data_yaml):
            return f"Labeled data saved to '{LABELED_IMAGES_DIR}'."
        return "Labeling finished, but 'data.yaml' not found."
    except Exception as e:
        return f"Labeling error: {e}"

# ──────────────────────────────────────────────────────────────
# 6. Train target model
# ──────────────────────────────────────────────────────────────
@mcp.tool()
async def train_model(epochs: int = 50, device: int = 0) -> str:
    """Fine-tune the target model and report the weight file location."""
    if STATE["target_model"] is None:
        return "Error: target model not set."

    if not os.path.exists(LABELED_IMAGES_DIR):
        return f"Error: labeled dataset '{LABELED_IMAGES_DIR}' is missing."

    ok, msg = train_target_model(
        epochs=epochs,
        device=device,
        output_dir=MODELS_DIR,  # ensure weights are written here
    )

    # Pick the newest .pt file as the trained model
    pt_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
    latest_pt: Optional[str] = max(pt_files, key=os.path.getmtime) if pt_files else None
    if latest_pt:
        STATE["trained_model_path"] = latest_pt
        msg += f"\nTrained model saved at: {latest_pt}"
    else:
        msg += f"\nWarning: no .pt file found in '{MODELS_DIR}'."

    return msg

# ──────────────────────────────────────────────────────────────
# 7. Fetch Unsplash images into RAW_IMAGES_DIR
# ──────────────────────────────────────────────────────────────
@mcp.tool()
async def fetch_unsplash_images(query: str, max_images: int = 20) -> str:
    """Download Unsplash images matching *query* into RAW_IMAGES_DIR."""
    if not query.strip():
        return "Error: query cannot be empty."
    if max_images <= 0:
        return "Error: max_images must be > 0."

    # Validate API key / directory
    access_key = STATE["unsplash_api_key"]
    ok, err_or_dir = validate_unsplash_params(access_key, RAW_IMAGES_DIR)
    if not ok:
        return err_or_dir  # error string

    subdir = os.path.join(RAW_IMAGES_DIR, f"{query.replace(' ', '_')}_images")
    os.makedirs(subdir, exist_ok=True)

    success, photos = search_unsplash_images(query, max_images, access_key)
    if not success:
        return photos  # error message from helper

    downloaded = download_images(photos, subdir)
    if downloaded == 0:
        return "Failed to download images."

    STATE["input_folder"] = RAW_IMAGES_DIR
    return f"Downloaded {downloaded} images to '{subdir}'."

# ──────────────────────────────────────────────────────────────
# 8. Starlette application factory
# ──────────────────────────────────────────────────────────────
def create_app():
    return Starlette(routes=[Mount("/", app=mcp.sse_app())])

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6277)
