"""SAM3 Model Management Utilities — local model detection, downloading, path management."""

import logging
import os

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

SAM3_MODELS_DIR = "sam3"


def get_sam3_models_path(models_dir: str = "models") -> str:
    """Get the path to SAM3 models directory, creating it if needed."""
    sam3_path = os.path.join(models_dir, SAM3_MODELS_DIR)
    os.makedirs(sam3_path, exist_ok=True)
    return sam3_path


def get_available_models(models_dir: str = "models") -> list[str]:
    """Get list of available SAM3 model checkpoints under models_dir/sam3."""
    sam3_path = get_sam3_models_path(models_dir)
    extensions = [".pt", ".pth", ".safetensors", ".bin"]

    models = ["auto (download from HuggingFace)"]

    if os.path.exists(sam3_path):
        for file in os.listdir(sam3_path):
            if any(file.endswith(ext) for ext in extensions):
                models.append(file)

    return models


def get_model_path(model_name: str, models_dir: str = "models") -> str | None:
    """Get full path to model checkpoint. Returns None for 'auto'."""
    if model_name in ("auto (download from HuggingFace)", "auto"):
        return None

    sam3_path = get_sam3_models_path(models_dir)
    model_path = os.path.join(sam3_path, model_name)

    if os.path.exists(model_path):
        return model_path

    return None


def download_sam3_model(
    hf_repo: str = "facebook/sam3",
    models_dir: str = "models",
) -> str:
    """Download the main SAM3 checkpoint (sam3.pt) into <models_dir>/sam3/.

    Returns the directory path containing sam3.pt.
    """
    sam3_dir = get_sam3_models_path(models_dir)
    os.makedirs(sam3_dir, exist_ok=True)

    local_ckpt_path = os.path.join(sam3_dir, "sam3.pt")
    if os.path.isfile(local_ckpt_path):
        logger.info("Using existing checkpoint: %s", local_ckpt_path)
        return sam3_dir

    logger.info("Downloading sam3.pt from %s to %s ...", hf_repo, local_ckpt_path)

    downloaded_path = hf_hub_download(
        repo_id=hf_repo,
        filename="sam3.pt",
        revision="main",
        local_dir=sam3_dir,
        local_dir_use_symlinks=False,
    )

    if downloaded_path != local_ckpt_path:
        import shutil

        shutil.move(downloaded_path, local_ckpt_path)

    logger.info("Model downloaded successfully to: %s", local_ckpt_path)
    return sam3_dir
