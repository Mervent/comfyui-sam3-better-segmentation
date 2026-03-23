from collections.abc import Callable
from typing import Any

import torch


def empty_masks(
    height: int, width: int, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Return a single all-zero mask tensor of shape (1, H, W)."""
    return torch.zeros(1, height, width, device=device)


def empty_segs(height: int, width: int) -> tuple[tuple[int, int], list]:
    """Return an empty SEGS structure: ((H, W), [])."""
    return ((height, width), [])


def empty_segmentation_result(
    height: int,
    width: int,
    vis_image_fn: Callable[..., Any],
    pil_image: Any,
    device: str | torch.device = "cpu",
) -> tuple:
    """Return an 8-tuple of empty outputs matching SAM3BSSegmentation's RETURN_TYPES."""
    empty_mask = empty_masks(height, width, device=device)
    segs = empty_segs(height, width)
    vis_image = vis_image_fn(pil_image)
    if isinstance(vis_image, torch.Tensor):
        vis_image = vis_image.to(device)
    return (
        empty_mask,
        vis_image,
        "[]",
        "[]",
        segs,
        empty_mask,
        segs,
        segs,
    )
