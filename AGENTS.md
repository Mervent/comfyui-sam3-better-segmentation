# AGENTS.md — comfyui-sam3-better-segmentation

## What This Project Is

A ComfyUI custom-node extension integrating Meta's SAM3 for image segmentation. Provides 3 nodes: model loading, text/point/box segmentation, and Florence2-powered SEGS captioning. Outputs Impact Pack SEGS format.

## Architecture

```
comfyui-sam3-better-segmentation/
├── nodes.py              # ComfyUI node classes (thin adapters → lib/)
├── lib/                  # Pure logic — zero ComfyUI imports
│   ├── geometry.py       # IoU, point-in-box, box/point denormalization, UnionFind
│   ├── mask_ops.py       # normalize, fill_holes, dilate_erode, apply_per_mask, build_combined_mask
│   ├── mask_filters.py   # filter_by_size, filter_by_density, filter_by_instances, limit_detections
│   ├── segs_builder.py   # build_detection_segs, build_combined_segs, build_overlapping_segs
│   ├── prompt_handler.py # extract_by_mode, aggregate_prompts, valid_block
│   ├── masktosegs.py     # mask_to_segs, make_2d_mask (Impact Pack SEGS conversion)
│   ├── types.py          # empty_masks, empty_segs, empty_segmentation_result
│   ├── conditioning_wrapper.py # ConditioningOverrideWrapper for per-SEG conditioning
│   └── florence2_captioner.py  # Florence2 captioning helpers
├── sam3_utils.py          # Tensor/image conversions, model device management
├── model_manager.py       # Model download/path resolution
├── sam3_lib/              # Vendored Meta SAM3 — DO NOT MODIFY
├── tests/                 # pytest suite — unit tests for lib/ modules
├── install.py             # SAM3 installation script
└── pyproject.toml         # Package metadata and dependencies
```

### The Golden Rule

`lib/` has **zero ComfyUI imports**. Pure Python + torch + numpy + cv2.
`nodes.py` is a **thin adapter** — INPUT_TYPES + delegation to `lib/`.

## Commands

```bash
make venv              # create .venv with Python 3.10 via uv
make test              # run full pytest suite
make lint              # ruff check lib/ tests/ nodes.py
make format            # ruff format lib/ tests/ nodes.py

# Run a single test file
uv run --no-project pytest tests/test_geometry.py -v

# Run a single test function
uv run --no-project pytest tests/test_geometry.py::test_iou_identical_boxes -v

# Run tests matching a keyword
uv run --no-project pytest tests/ -v -k "filter_by_size"
```

Tests cover `lib/` modules only — pure logic, no SAM3 model needed. No ruff config in pyproject.toml; ruff uses defaults.

## Code Style

### Python Version & Types

Target **Python 3.10+**. Use modern union syntax and lowercase generics:

```python
# YES
x: torch.Tensor | None = None
def foo() -> tuple[int, int]: ...
_Triplet: TypeAlias = tuple[...]

# NO — do not use these
x: Optional[torch.Tensor] = None    # old-style Optional
def foo() -> Tuple[int, int]: ...    # old-style Tuple
type _Triplet = tuple[...]           # 3.12-only syntax
```

### Imports

- `lib/` modules use **relative imports**: `from .module import ...`
- `nodes.py` imports from `lib/` via relative: `from .lib import mask_filters`
- `nodes.py` imports from root-level modules without dot: `from sam3_utils import ...`
- `tests/` import `lib/` as a package: `from lib.geometry import compute_iou`
- `conftest.py` prepends the project root to `sys.path` so `lib` is importable
- No unused imports. No wildcard imports.

### Function Signatures

- **Use all parameters.** No dead params in signatures.
- **Consistent return types.** Same shape on all code paths.
- **No-op means no-op.** Identity operations return input unchanged.

### Function Calls — Keyword Arguments

Use kwargs when passing more than 2 parameters:

```python
# 2 params — positional is fine
compute_iou(box_a, box_b)

# 3+ params — use kwargs
masks, boxes, scores = filter_by_size(
    masks=masks,
    boxes=boxes,
    scores=scores,
    min_size=32,
)
```

### Null Handling

One pattern — early return guard, then proceed:

```python
if masks is None or len(masks) == 0:
    return shape_info, []

masks_cpu = masks.detach().cpu()
```

### Error Handling

- Use `raise ValueError(...)` with descriptive messages for invalid inputs
- No empty `except` blocks
- Filter functions return `(None, None, None)` when all items filtered out

### Naming

- Functions: `snake_case` — verb-first (`filter_by_size`, `build_combined_mask`)
- Private helpers: `_prefixed` (`_index`)
- Type aliases: `_PrefixedPascalCase` (`_Triplet: TypeAlias = tuple[...]`)
- Classes: `PascalCase`, ComfyUI nodes prefixed with `SAM3BS` (`SAM3BSSegmentation`)
- Constants: `UPPER_SNAKE` (`_MODEL_CACHE`, `TASK_LIST`)
- Test functions: `test_<unit>_<scenario>` (`test_filter_by_size_removes_small`)

## Key Types

- **Masks**: `torch.Tensor` `[N, H, W]` float32 0-1. 4D `[N, 1, H, W]` normalized via `mask_ops.normalize_masks()`.
- **SEGS**: `tuple[tuple[int, int], list[SEG]]` — `((height, width), seg_list)`.
- **Boxes**: `torch.Tensor` `[N, 4]` as `[x1, y1, x2, y2]` pixel coords.
- **Scores**: `torch.Tensor` `[N]` confidence values.
- **Filter triplet**: `(masks | None, boxes | None, scores | None)` — aliased as `_Triplet`.

## Test Conventions

- All test functions annotated `-> None`
- Docstring on every test: one sentence stating what is verified
- AAA pattern (Arrange / Act / Assert) with blank-line separation
- Shared fixtures in `tests/conftest.py` with return type hints
- Fixtures: `square_mask_np`, `mask_with_hole_np`, `masks_3d`, `masks_4d`, `sample_boxes`, `sample_scores`

## Data Flow (single image segmentation)

```
nodes.py SAM3BSSegmentation.segment()
  ├─ prompt_handler.extract_by_mode()     → route pipeline prompts by mode
  ├─ SAM3 inference (sam3_utils)          → raw masks, boxes, scores
  ├─ mask_filters.filter_by_size()        → remove small masks
  ├─ mask_filters.filter_by_density()     → remove sparse masks
  ├─ mask_filters.filter_by_instances()   → keep only prompt-matched detections
  ├─ mask_filters.limit_detections()      → top-k by score
  ├─ mask_ops.apply_per_mask(fill_holes)  → morphology
  ├─ mask_ops.apply_per_mask(dilate_erode)→ morphology
  ├─ mask_ops.build_combined_mask()       → OR-merge all masks
  └─ segs_builder.build_*_segs()          → 4 SEGS output variants
```

## Do NOT

- Add ComfyUI imports to `lib/` — the golden rule
- Modify anything in `sam3_lib/` — vendored upstream
- Use `as any`-style type suppression or `# type: ignore` without justification
- Return empty tensors from filters — return `None` triplet instead
- Add dead parameters to function signatures
- Use `Optional[X]` or `Tuple[X]` — use `X | None` and `tuple[x]`
