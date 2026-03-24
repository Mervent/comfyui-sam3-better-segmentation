import re
from typing import Any

# Mode → set of pipeline keys to extract
_MODE_KEYS: dict[str, set[str]] = {
    "all": {"positive_boxes", "negative_boxes", "positive_points", "negative_points"},
    "boxes_only": {"positive_boxes", "negative_boxes"},
    "points_only": {"positive_points", "negative_points"},
    "positive_only": {"positive_boxes", "positive_points"},
    "negative_only": {"negative_boxes", "negative_points"},
}


def valid_block(block: Any, key: str) -> bool:
    """Check if block is a dict containing a truthy value for key."""
    return isinstance(block, dict) and key in block and bool(block[key])


def extract_by_mode(
    pipeline_data: dict | None,
    mode: str,
) -> tuple[Any, Any, Any, Any]:
    """Extract prompt blocks from pipeline dict filtered by mode."""
    positive_boxes = None
    negative_boxes = None
    positive_points = None
    negative_points = None

    if pipeline_data is None or mode == "disabled":
        return positive_boxes, negative_boxes, positive_points, negative_points

    if not isinstance(pipeline_data, dict):
        raise ValueError(
            f"sam3_selectors_pipe must be a dictionary, got {type(pipeline_data)}"
        )

    keys_to_extract = _MODE_KEYS.get(mode, set())
    if "positive_boxes" in keys_to_extract:
        positive_boxes = pipeline_data.get("positive_boxes", None)
    if "negative_boxes" in keys_to_extract:
        negative_boxes = pipeline_data.get("negative_boxes", None)
    if "positive_points" in keys_to_extract:
        positive_points = pipeline_data.get("positive_points", None)
    if "negative_points" in keys_to_extract:
        negative_points = pipeline_data.get("negative_points", None)

    return positive_boxes, negative_boxes, positive_points, negative_points


def aggregate_prompts(
    positive_block: dict | None,
    negative_block: dict | None,
    prompt_type: str,
) -> tuple[list, list]:
    """Merge positive and negative prompt blocks into flat lists of prompts and labels."""
    if prompt_type not in ("boxes", "points"):
        raise ValueError(f"Unsupported prompt_type: {prompt_type}")

    all_prompts: list = []
    all_labels: list = []

    if valid_block(positive_block, prompt_type):
        all_prompts.extend(positive_block[prompt_type])
        all_labels.extend(
            positive_block.get("labels", [1] * len(positive_block[prompt_type]))
        )

    if valid_block(negative_block, prompt_type):
        all_prompts.extend(negative_block[prompt_type])
        all_labels.extend(
            negative_block.get("labels", [0] * len(negative_block[prompt_type]))
        )

    return all_prompts, all_labels


def split_text_prompts(text_prompt: str) -> list[str]:
    """Split a text prompt by uppercase ``OR`` separator into sub-prompts.

    Only the exact uppercase token ``OR`` surrounded by whitespace is treated
    as a separator.  Lowercase ``or`` is preserved as part of the prompt text.

    Returns a list of stripped, non-empty sub-prompts.  An empty or
    whitespace-only input returns an empty list.
    """
    if not text_prompt or not text_prompt.strip():
        return []

    parts = re.split(r"\bOR\b", text_prompt)
    return [p.strip() for p in parts if p.strip()]
