import numpy as np
import torch

from lib.masktosegs import (
    SEG,
    make_2d_mask,
    make_crop_region,
    mask_to_segs,
    normalize_region,
)


# --- SEG namedtuple ---


def test_seg_fields_accessible() -> None:
    """All 7 SEG fields are accessible by name."""
    seg = SEG(
        cropped_image=None,
        cropped_mask=np.ones((10, 10)),
        confidence=0.9,
        crop_region=(0, 0, 10, 10),
        bbox=(2, 2, 8, 8),
        label="cat_0",
        control_net_wrapper=None,
    )

    assert seg.cropped_image is None
    assert seg.cropped_mask is not None
    assert seg.confidence == 0.9
    assert seg.crop_region == (0, 0, 10, 10)
    assert seg.bbox == (2, 2, 8, 8)
    assert seg.label == "cat_0"
    assert seg.control_net_wrapper is None


def test_seg_control_net_wrapper_defaults_none() -> None:
    """control_net_wrapper defaults to None when omitted."""
    seg = SEG(
        cropped_image=None,
        cropped_mask=np.ones((5, 5)),
        confidence=1.0,
        crop_region=(0, 0, 5, 5),
        bbox=(0, 0, 5, 5),
        label="test",
    )

    assert seg.control_net_wrapper is None


# --- make_2d_mask ---


def test_make_2d_mask_from_3d_tensor() -> None:
    """[1, H, W] torch tensor becomes [H, W] numpy array."""
    mask = torch.rand(1, 20, 30)

    result = make_2d_mask(mask)

    assert isinstance(result, np.ndarray)
    assert result.shape == (20, 30)


def test_make_2d_mask_from_2d_array() -> None:
    """[H, W] numpy array passes through unchanged."""
    mask = np.random.rand(20, 30).astype(np.float32)

    result = make_2d_mask(mask)

    assert result.shape == (20, 30)
    np.testing.assert_array_equal(result, mask)


def test_make_2d_mask_from_4d_tensor() -> None:
    """[1, 1, H, W] tensor squeezes down to [H, W]."""
    mask = torch.rand(1, 1, 15, 25)

    result = make_2d_mask(mask)

    assert result.shape == (15, 25)


# --- normalize_region ---


def test_normalize_region_clamps_negative_start() -> None:
    """Negative start is clamped to 0, end clamped to min(limit, size)."""
    start, end = normalize_region(limit=100, startp=-10, size=50)

    assert start == 0
    assert end == 50


def test_normalize_region_clamps_overflow() -> None:
    """Start + size exceeding limit is pushed back."""
    start, end = normalize_region(limit=100, startp=80, size=50)

    assert start == 50
    assert end == 100


def test_normalize_region_valid_passthrough() -> None:
    """Valid coordinates within limits pass through unchanged."""
    start, end = normalize_region(limit=100, startp=20, size=30)

    assert start == 20
    assert end == 50


# --- make_crop_region ---


def test_make_crop_region_with_factor() -> None:
    """crop_factor > 1 expands the crop region beyond the bbox."""
    bbox = (40, 40, 60, 60)

    region = make_crop_region(w=100, h=100, bbox=bbox, crop_factor=2.0)

    x1, y1, x2, y2 = region
    # Expanded region should be larger than the 20x20 bbox
    assert (x2 - x1) > 20
    assert (y2 - y1) > 20
    # But clamped within image bounds
    assert x1 >= 0 and y1 >= 0
    assert x2 <= 100 and y2 <= 100


def test_make_crop_region_tight() -> None:
    """crop_factor=1 still adds minimum padding (72px per axis)."""
    bbox = (10, 10, 90, 90)

    region = make_crop_region(w=200, h=200, bbox=bbox, crop_factor=1.0)

    x1, y1, x2, y2 = region
    # bbox is 80x80, minimum expansion is bbox + 72 = 152
    assert (x2 - x1) >= 80
    assert (y2 - y1) >= 80


# --- mask_to_segs ---


def test_mask_to_segs_combined_single_seg() -> None:
    """combined=True with one mask region produces exactly one SEG."""
    mask = np.zeros((1, 50, 50), dtype=np.float32)
    mask[0, 10:30, 10:30] = 1.0

    shape_info, segs = mask_to_segs(
        mask=mask,
        combined=True,
        crop_factor=1.5,
        bbox_fill=False,
    )

    assert shape_info == (50, 50)
    assert len(segs) == 1
    assert segs[0].label == "A"
    assert segs[0].confidence == 1.0
    assert segs[0].cropped_mask is not None


def test_mask_to_segs_empty_mask() -> None:
    """All-zero mask returns empty seg list."""
    mask = np.zeros((1, 50, 50), dtype=np.float32)

    shape_info, segs = mask_to_segs(
        mask=mask,
        combined=True,
        crop_factor=1.0,
        bbox_fill=False,
    )

    assert segs == []


def test_mask_to_segs_none_mask() -> None:
    """None mask returns (0, 0) shape and empty list."""
    shape_info, segs = mask_to_segs(
        mask=None,
        combined=True,
        crop_factor=1.0,
        bbox_fill=False,
    )

    assert shape_info == (0, 0)
    assert segs == []


def test_mask_to_segs_custom_label() -> None:
    """Custom label is stored in resulting SEG."""
    mask = np.zeros((1, 50, 50), dtype=np.float32)
    mask[0, 5:20, 5:20] = 1.0

    _, segs = mask_to_segs(
        mask=mask,
        combined=True,
        crop_factor=1.5,
        bbox_fill=False,
        label="person_0",
    )

    assert len(segs) == 1
    assert segs[0].label == "person_0"


def test_mask_to_segs_bbox_matches_nonzero() -> None:
    """SEG bbox matches the nonzero region of the mask."""
    mask = np.zeros((1, 100, 100), dtype=np.float32)
    mask[0, 20:40, 30:60] = 1.0

    _, segs = mask_to_segs(
        mask=mask,
        combined=True,
        crop_factor=1.5,
        bbox_fill=False,
    )

    assert len(segs) == 1
    x1, y1, x2, y2 = segs[0].bbox
    assert x1 == 30
    assert y1 == 20
    assert x2 == 59  # np.max of nonzero cols
    assert y2 == 39  # np.max of nonzero rows


def test_mask_to_segs_2d_input() -> None:
    """2D mask [H, W] is auto-expanded and processed correctly."""
    mask = np.zeros((50, 50), dtype=np.float32)
    mask[10:30, 10:30] = 1.0

    shape_info, segs = mask_to_segs(
        mask=mask,
        combined=True,
        crop_factor=1.5,
        bbox_fill=False,
    )

    assert len(segs) == 1
    assert shape_info == (50, 50)
