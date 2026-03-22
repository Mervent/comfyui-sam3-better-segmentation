import torch
from PIL import Image

from lib.florence2_captioner import (
    TASK_LIST,
    TASK_PROMPTS,
    bbox_crops_to_tensor,
    build_caption,
    crop_image_region,
    hash_seed,
)


# --- hash_seed ---


def test_hash_seed_deterministic() -> None:
    """Same input always produces the same hash."""
    assert hash_seed(42) == hash_seed(42)


def test_hash_seed_range() -> None:
    """Output fits within 32-bit unsigned integer range."""
    result = hash_seed(99999)

    assert 0 <= result < 2**32


def test_hash_seed_different_inputs() -> None:
    """Different seeds produce different hashes."""
    assert hash_seed(1) != hash_seed(2)


# --- crop_image_region ---


def test_crop_image_region_3d_tensor() -> None:
    """[H, W, C] tensor is cropped to region and returned as PIL RGB."""
    image = torch.rand(20, 30, 3)
    region = (5, 5, 15, 15)

    result = crop_image_region(image, region)

    assert isinstance(result, Image.Image)
    assert result.mode == "RGB"
    assert result.size == (10, 10)


def test_crop_image_region_4d_tensor() -> None:
    """[B, H, W, C] tensor takes batch[0] and crops correctly."""
    image = torch.rand(2, 20, 30, 3)
    region = (0, 0, 10, 8)

    result = crop_image_region(image, region)

    assert isinstance(result, Image.Image)
    assert result.size == (10, 8)


def test_crop_image_region_pixel_values() -> None:
    """Cropped pixels match the source region scaled to 0-255."""
    image = torch.zeros(10, 10, 3)
    image[2:5, 3:7, :] = 1.0
    region = (3, 2, 7, 5)

    result = crop_image_region(image, region)

    assert result.size == (4, 3)
    # All pixels in crop region were 1.0 -> 255
    pixels = list(result.getdata())
    assert all(p == (255, 255, 255) for p in pixels)


def test_crop_image_region_from_list_coords() -> None:
    """Coords converted from list (crop_region format) work identically."""
    image = torch.rand(20, 30, 3)
    crop_region_list = [2, 3, 18, 15]

    result = crop_image_region(image, tuple(crop_region_list))

    assert isinstance(result, Image.Image)
    assert result.size == (16, 12)


# --- bbox_crops_to_tensor ---


def test_bbox_crops_to_tensor_single() -> None:
    """Single image produces [1, H, W, 3] tensor."""
    img = Image.new("RGB", (10, 8), color=(128, 128, 128))

    result = bbox_crops_to_tensor([img])

    assert result.shape == (1, 8, 10, 3)
    assert result.dtype == torch.float32


def test_bbox_crops_to_tensor_variable_sizes() -> None:
    """Different-sized crops are padded to max dimensions."""
    small = Image.new("RGB", (5, 5), color=(255, 0, 0))
    large = Image.new("RGB", (10, 8), color=(0, 255, 0))

    result = bbox_crops_to_tensor([small, large])

    assert result.shape == (2, 8, 10, 3)


def test_bbox_crops_to_tensor_empty() -> None:
    """Empty list returns [1, 64, 64, 3] zeros fallback."""
    result = bbox_crops_to_tensor([])

    assert result.shape == (1, 64, 64, 3)
    assert torch.count_nonzero(result).item() == 0


# --- build_caption ---


def test_build_caption_prepend() -> None:
    """Prepend mode puts user text before generated."""
    result = build_caption(
        generated="a fluffy cat",
        user_prompt="photo of",
        prompt_mode="prepend",
    )

    assert result == "photo of, a fluffy cat"


def test_build_caption_append() -> None:
    """Append mode puts generated text before user text."""
    result = build_caption(
        generated="a fluffy cat",
        user_prompt="high quality",
        prompt_mode="append",
    )

    assert result == "a fluffy cat, high quality"


def test_build_caption_generated_only() -> None:
    """Generated-only mode ignores user text."""
    result = build_caption(
        generated="a fluffy cat",
        user_prompt="should be ignored",
        prompt_mode="generated only",
    )

    assert result == "a fluffy cat"


def test_build_caption_user_only() -> None:
    """User-only mode ignores generated text."""
    result = build_caption(
        generated="should be ignored",
        user_prompt="my prompt",
        prompt_mode="user only",
    )

    assert result == "my prompt"


def test_build_caption_empty_user_falls_back() -> None:
    """Empty user prompt in any mode falls back to generated."""
    result = build_caption(
        generated="a fluffy cat",
        user_prompt="",
        prompt_mode="prepend",
    )

    assert result == "a fluffy cat"


# --- constants ---


def test_task_prompts_keys_match_task_list() -> None:
    """TASK_LIST is derived from TASK_PROMPTS keys."""
    assert TASK_LIST == list(TASK_PROMPTS.keys())
