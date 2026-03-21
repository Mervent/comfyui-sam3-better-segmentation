import torch

from lib.conditioning_wrapper import ConditioningOverrideWrapper


# --- __init__ ---


def test_init_stores_fields() -> None:
    """Constructor stores conditioning, mode, and original_wrapper."""
    cond = [[torch.randn(1, 4, 64), {}]]

    wrapper = ConditioningOverrideWrapper(
        conditioning=cond,
        mode="replace",
        original_wrapper=None,
    )

    assert wrapper.conditioning is cond
    assert wrapper.mode == "replace"
    assert wrapper.original_wrapper is None


def test_init_extracts_control_image() -> None:
    """Picks up control_image attribute from original wrapper."""

    class FakeWrapper:
        control_image = "fake_image"

    wrapper = ConditioningOverrideWrapper(
        conditioning=[[torch.zeros(1, 4, 64), {}]],
        original_wrapper=FakeWrapper(),
    )

    assert wrapper.control_image == "fake_image"


def test_init_no_original_wrapper_control_image_none() -> None:
    """control_image is None when no original wrapper provided."""
    wrapper = ConditioningOverrideWrapper(
        conditioning=[[torch.zeros(1, 4, 64), {}]],
    )

    assert wrapper.control_image is None


# --- apply ---


def test_apply_replace_mode() -> None:
    """Replace mode returns stored conditioning as positive."""
    stored_cond = [[torch.randn(1, 4, 64), {"pooled_output": torch.randn(1, 64)}]]
    neg = [[torch.randn(1, 4, 64), {}]]

    wrapper = ConditioningOverrideWrapper(conditioning=stored_cond, mode="replace")
    pos_out, neg_out, extras = wrapper.apply(
        positive=[[torch.zeros(1, 4, 64), {}]],
        negative=neg,
        image=None,
    )

    assert pos_out is stored_cond
    assert neg_out is neg
    assert extras == []


def test_apply_concat_mode() -> None:
    """Concat mode concatenates stored conditioning onto base positive dim=1."""
    base_cond = [[torch.zeros(1, 4, 64), {"key": "val"}]]
    added_cond = [[torch.ones(1, 2, 64), {}]]

    wrapper = ConditioningOverrideWrapper(conditioning=added_cond, mode="concat")
    pos_out, neg_out, extras = wrapper.apply(
        positive=base_cond,
        negative=[[torch.zeros(1, 4, 64), {}]],
        image=None,
    )

    # dim=1 should be 4 + 2 = 6
    assert pos_out[0][0].shape[1] == 6
    assert extras == []


def test_apply_chains_original_wrapper() -> None:
    """Delegates to original_wrapper.apply() when present."""
    call_log = []

    class FakeWrapper:
        control_image = None

        def apply(self, positive, negative, image, noise_mask=None):
            call_log.append("chained")
            return positive, negative, ["extra"]

    stored_cond = [[torch.randn(1, 4, 64), {}]]
    wrapper = ConditioningOverrideWrapper(
        conditioning=stored_cond,
        mode="replace",
        original_wrapper=FakeWrapper(),
    )
    _, _, extras = wrapper.apply(
        positive=[[torch.zeros(1, 4, 64), {}]],
        negative=[[torch.zeros(1, 4, 64), {}]],
        image=None,
    )

    assert call_log == ["chained"]
    assert extras == ["extra"]


# --- _concat_conditioning ---


def test_concat_conditioning_shape() -> None:
    """Concatenated tensor has base + added tokens on dim 1."""
    base = [[torch.zeros(1, 3, 64), {"key": "val"}]]
    added = [[torch.ones(1, 5, 64), {}]]

    wrapper = ConditioningOverrideWrapper(conditioning=added)
    result = wrapper._concat_conditioning(base=base, added=added)

    assert result[0][0].shape == (1, 8, 64)
    # Metadata dict is copied
    assert result[0][1] == {"key": "val"}


# --- doit_ipadapter ---


def test_doit_ipadapter_no_original() -> None:
    """Without original wrapper, returns model unchanged and empty list."""
    wrapper = ConditioningOverrideWrapper(
        conditioning=[[torch.zeros(1, 4, 64), {}]],
    )
    model_in = object()

    model_out, extras = wrapper.doit_ipadapter(model_in)

    assert model_out is model_in
    assert extras == []


def test_doit_ipadapter_delegates() -> None:
    """With original wrapper, delegates to its doit_ipadapter."""

    class FakeWrapper:
        control_image = None

        def doit_ipadapter(self, model):
            return "modified_model", ["adapter"]

    wrapper = ConditioningOverrideWrapper(
        conditioning=[[torch.zeros(1, 4, 64), {}]],
        original_wrapper=FakeWrapper(),
    )

    model_out, extras = wrapper.doit_ipadapter("any_model")

    assert model_out == "modified_model"
    assert extras == ["adapter"]
