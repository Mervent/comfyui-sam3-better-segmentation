"""Native Florence2 model loading — no vendored code, no trust_remote_code.

Requires transformers >= 4.56.0 for native Florence2 support.
Pure logic — zero ComfyUI imports.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

FLORENCE2_MODELS: dict[str, str] = {
    "Florence-2-base": "florence-community/Florence-2-base",
    "Florence-2-large": "florence-community/Florence-2-large",
    "Florence-2-base-ft": "florence-community/Florence-2-base-ft",
    "Florence-2-large-ft": "florence-community/Florence-2-large-ft",
}

PRECISION_MAP: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def download_florence2(
    repo_id: str,
    models_dir: str,
) -> str:
    """Download a Florence2 model from HuggingFace Hub.

    Parameters
    ----------
    repo_id:
        HuggingFace repository ID (e.g. ``florence-community/Florence-2-large``).
    models_dir:
        Root models directory (e.g. ``folder_paths.models_dir``).

    Returns
    -------
    str
        Local directory path containing the downloaded model.
    """
    from huggingface_hub import snapshot_download

    model_name = repo_id.rsplit("/", 1)[-1]
    fl2_dir = os.path.join(models_dir, "florence2")
    os.makedirs(fl2_dir, exist_ok=True)
    local_path = os.path.join(fl2_dir, model_name)

    if not os.path.exists(local_path):
        logger.info("Downloading %s to %s", repo_id, local_path)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_path,
            local_dir_use_symlinks=False,
        )
    else:
        logger.info("Using cached model at %s", local_path)

    return local_path


def load_florence2_model(
    model_path: str,
    *,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str = "cpu",
    attn_implementation: str = "sdpa",
) -> dict[str, Any]:
    """Load Florence2 using native transformers (>= 4.56.0).

    Returns an FL2MODEL-compatible dict with ``use_cache=True``.

    Parameters
    ----------
    model_path:
        Local directory containing the model files.
    dtype:
        Compute dtype (fp16, bf16, or fp32).
    device:
        Torch device to load the model onto.
    attn_implementation:
        Attention backend — ``"sdpa"``, ``"eager"``, or ``"flash_attention_2"``.

    Returns
    -------
    dict[str, Any]
        FL2MODEL dict: ``{"model", "processor", "dtype", "use_cache"}``.
    """
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    logger.info(
        "Loading Florence2 from %s (dtype=%s, attn=%s)",
        model_path,
        dtype,
        attn_implementation,
    )

    model = (
        Florence2ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        )
        .to(device)
        .eval()
    )

    processor = AutoProcessor.from_pretrained(model_path)

    return {
        "model": model,
        "processor": processor,
        "dtype": dtype,
        "use_cache": True,
    }
