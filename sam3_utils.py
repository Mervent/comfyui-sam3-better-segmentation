"""
SAM3 Utility Functions — tensor/image conversions and model device management.
"""

import gc
import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def comfy_image_to_pil(image):
    """Convert ComfyUI image tensor [B, H, W, C] range [0,1] to PIL Image."""
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image[0]
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)
    return image


def pil_to_comfy_image(pil_image):
    """Convert PIL Image to ComfyUI image tensor [1, H, W, C] range [0,1]."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    img_np = np.array(pil_image).astype(np.float32) / 255.0
    return torch.from_numpy(img_np).unsqueeze(0)


def masks_to_comfy_mask(masks):
    """Convert SAM3 masks to ComfyUI mask format [N, H, W] float32 on CPU."""
    if isinstance(masks, np.ndarray):
        masks = torch.from_numpy(masks).float()
    elif isinstance(masks, torch.Tensor):
        masks = masks.float()
    else:
        return masks

    if masks.max() > 1.0:
        masks = masks / 255.0
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)
    return masks.cpu()


def visualize_masks_on_image(image, masks, boxes=None, scores=None, alpha=0.5):
    """Create visualization of masks overlaid on image."""
    image = comfy_image_to_pil(image)
    img_np = np.array(image).astype(np.float32) / 255.0

    if isinstance(masks, torch.Tensor):
        masks_np = masks.cpu().numpy()
    else:
        masks_np = masks

    rng = np.random.RandomState(42)
    overlay = img_np.copy()

    for i, mask in enumerate(masks_np):
        while mask.ndim > 2:
            mask = mask.squeeze(0)

        if mask.shape != img_np.shape[:2]:
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize(
                (img_np.shape[1], img_np.shape[0]), Image.NEAREST
            )
            mask = np.array(mask_pil).astype(np.float32) / 255.0

        color = rng.rand(3)
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask > 0.5,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c],
            )

    result = Image.fromarray((overlay * 255).astype(np.uint8))

    if boxes is not None:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(result)

        if isinstance(boxes, torch.Tensor):
            boxes_np = boxes.cpu().numpy()
        else:
            boxes_np = boxes

        for i, box in enumerate(boxes_np):
            x0, y0, x1, y1 = box
            box_rng = np.random.RandomState(42 + i)
            color_int = tuple((box_rng.rand(3) * 255).astype(int).tolist())
            draw.rectangle([x0, y0, x1, y1], outline=color_int, width=3)

            if scores is not None:
                score = (
                    scores[i]
                    if isinstance(scores, (list, np.ndarray))
                    else scores[i].item()
                )
                draw.text((x0, y0 - 15), f"{score:.2f}", fill=color_int)

    return result


def tensor_to_list(tensor):
    """Convert torch tensor to python list."""
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().tolist()
    return tensor


def ensure_model_on_device(sam3_model, target_device=None):
    """Ensure model is on the target device before inference."""
    model = sam3_model["model"]
    processor = sam3_model["processor"]

    if target_device is None:
        target_device = sam3_model["original_device"]

    current_device = next(model.parameters()).device
    if str(current_device) != target_device:
        logger.info(f"Moving model from {current_device} to {target_device}")
        model.to(target_device)
        processor.device = target_device
        sam3_model["device"] = target_device


def offload_model_if_needed(sam3_model):
    """Offload model to CPU if use_gpu_cache is False."""
    if not sam3_model.get("use_gpu_cache", True):
        model = sam3_model["model"]
        processor = sam3_model["processor"]
        current_device = next(model.parameters()).device

        if "cuda" in str(current_device):
            logger.info("Offloading model to CPU to free VRAM")
            model.to("cpu")
            processor.device = "cpu"
            sam3_model["device"] = "cpu"
            torch.cuda.empty_cache()
            gc.collect()


def run_sam3_inference(
    sam3_model: dict,
    pil_image,
    confidence_threshold: float,
    text_prompt: str,
    box_prompts: list,
    box_labels: list,
    point_prompts: list,
    point_labels: list,
    mask_prompt=None,
) -> tuple:
    """Run SAM3 inference: model setup, prompt injection, return raw (masks, boxes, scores)."""
    ensure_model_on_device(sam3_model)
    processor = sam3_model["processor"]
    logger.info("Running segmentation")
    logger.info(f"Confidence threshold: {confidence_threshold}")
    logger.info(f"Image size: {pil_image.size}")

    processor.set_confidence_threshold(confidence_threshold)
    state = processor.set_image(pil_image)

    if text_prompt and text_prompt.strip():
        logger.info(f"Using text_prompt='{text_prompt.strip()}'")
        state = processor.set_text_prompt(text_prompt.strip(), state)

    logger.info(f"total box prompts={len(box_prompts)}")
    if box_prompts:
        state = processor.add_multiple_box_prompts(box_prompts, box_labels, state)

    logger.info(f"total point prompts={len(point_prompts)}")
    if point_prompts:
        state = processor.add_point_prompt(point_prompts, point_labels, state)

    if mask_prompt is not None:
        if not isinstance(mask_prompt, torch.Tensor):
            mask_prompt = torch.from_numpy(mask_prompt)
        mask_prompt = mask_prompt.to(sam3_model["device"])
        logger.info("Adding external mask_prompt")
        state = processor.add_mask_prompt(mask_prompt, state)

    masks = state.get("masks")
    boxes = state.get("boxes")
    scores = state.get("scores")

    total_scores = len(scores) if scores is not None else 0
    logger.debug(f"RAW PREDICTIONS: total {total_scores}")
    if boxes is not None:
        logger.debug(f"Output boxes shape: {boxes.shape}")

    return masks, boxes, scores
