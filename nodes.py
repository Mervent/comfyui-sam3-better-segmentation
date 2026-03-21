"""
ComfyUI SAM3 Nodes, unified model loader for both image and video using official Meta sam3_lib.
All class names and functions prefixed with SAM3BS for uniqueness.
"""

import json
import os

import torch

from sam3_utils import (
    comfy_image_to_pil,
    ensure_model_on_device,
    masks_to_comfy_mask,
    offload_model_if_needed,
    pil_to_comfy_image,
    tensor_to_list,
    visualize_masks_on_image,
)

# Impact-Pack style MASK -> SEGS helper (your file in same folder)
from .lib import mask_filters, mask_ops, prompt_handler
from .lib import types as lib_types
from .lib.conditioning_wrapper import ConditioningOverrideWrapper
from .lib.florence2_captioner import (
    TASK_LIST,
    bbox_crops_to_tensor,
    build_caption,
    caption_image,
    crop_seg_bbox,
)
from .lib.masktosegs import SEG
from .lib.segs_builder import (
    build_combined_segs,
    build_detection_segs,
    build_overlapping_segs,
)
from .model_manager import download_sam3_model, get_available_models, get_model_path
from .sam3_lib.model.sam3_image_processor import Sam3Processor
from .sam3_lib.model_builder import build_sam3_image_model

_MODEL_CACHE = {}


class SAM3BSModelLoaderAndDownloader:
    """
    Advanced SAM3 model loader that:
    - Can use the official API (auto configuration)
    - Can auto-download a local checkpoint if missing
    - Can load a specific local checkpoint under models/sam3
    Returns a SAM3_MODEL dict: {model, processor, device, original_device}.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # List known local models from model_manager
        # get_available_models() returns ["auto (download from HuggingFace)", <files...>]
        available = get_available_models()
        # Present clearer choices in UI
        model_sources = (
            [
                "auto (API to cache)",  # build default model (no fixed ckpt path)
                "local (auto-download)",  # download sam3.pt into models/sam3 if missing
            ]
            + available[1:]
        )  # additional discovered checkpoint files

        return {
            "required": {
                "model_source": (model_sources, {"default": "local (auto-download)"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            },
        }

    RETURN_TYPES = ("SAM3_MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "load_model"
    CATEGORY = "SAM3BS"

    def load_model(self, model_source: str, device: str):
        hf_repo = "facebook/sam3"

        """
        Build and return a SAM3_MODEL dict:
          {model, processor, device, original_device}
        """
        # Resolve checkpoint path if needed
        checkpoint_path = None

        if model_source == "auto (API to cache)":
            # Let builder construct its default weights / config
            print("[SAM3BSModelLoaderAdvanced] Using API/default SAM3 image model.")
            checkpoint_path = None

        elif model_source == "local (auto-download)":
            # Download only sam3.pt into models/sam3
            sam3_dir = download_sam3_model(hf_repo)  # returns models/sam3
            checkpoint_path = os.path.join(sam3_dir, "sam3.pt")
            if not os.path.isfile(checkpoint_path):
                raise RuntimeError(
                    f"[SAM3BSModelLoaderAdvanced] Downloaded model file not found at: {checkpoint_path}"
                )
            print(
                f"[SAM3BSModelLoaderAdvanced] Using downloaded local checkpoint: {checkpoint_path}"
            )

        else:
            # Specific local checkpoint chosen from list under models/sam3
            checkpoint_path = get_model_path(model_source)
            if not checkpoint_path or not os.path.isfile(checkpoint_path):
                raise RuntimeError(
                    f"[SAM3BSModelLoaderAdvanced] Local model file not found: {model_source} -> {checkpoint_path}"
                )
            print(
                f"[SAM3BSModelLoaderAdvanced] Using selected local checkpoint: {checkpoint_path}"
            )

        # --- Build SAM3 image model + processor, mirroring SAM3BSLoadModel ---

        if checkpoint_path:
            sam3_model = build_sam3_image_model(checkpoint_path=checkpoint_path)
        else:
            sam3_model = build_sam3_image_model()

        processor = Sam3Processor(sam3_model)

        sam3_model.to(device)
        sam3_model.processor = processor
        sam3_model.eval()

        model_dict = {
            "model": sam3_model,
            "processor": processor,
            "device": device,
            "original_device": device,
        }

        print("[SAM3BSModelLoaderAdvanced] SAM3 model ready on device:", device)
        return (model_dict,)


class SAM3BSSegmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam3_model": (
                    "SAM3_MODEL",
                    {"tooltip": "SAM3 model loaded from LoadSAM3Model node"},
                ),
                "image": (
                    "IMAGE",
                    {"tooltip": "Input image to perform segmentation on"},
                ),
                "confidence_threshold": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "slider",
                        "tooltip": "Minimum confidence score to keep detections. Lower threshold (0.2) works better with SAM3's presence scoring",
                    },
                ),
                "pipeline_mode": (
                    [
                        "all",
                        "boxes_only",
                        "points_only",
                        "positive_only",
                        "negative_only",
                        "disabled",
                    ],
                    {
                        "default": "all",
                        "tooltip": "Which prompts from pipeline to use.",
                    },
                ),
                "detect_all": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Detect All",
                        "label_off": "Limit Detections to max_detection",
                        "tooltip": "When enabled, detects all objects. When disabled, uses max_detections value.",
                    },
                ),
                "max_detections": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Maximum detections when detect_all is disabled.",
                    },
                ),
                "instances": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "No Instances",
                        "label_off": "All Instances",
                        "tooltip": (
                            "When ON: keep only detections whose boxes overlap a positive box or contain a positive point.\n"
                            "When OFF: return all SAM3 detections including instances."
                        ),
                    },
                ),
                "crop_factor": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 4.0,
                        "step": 0.1,
                        "tooltip": "Crop factor used when building combined SEGS (Impact Pack style). 1.0 = tight bbox.",
                    },
                ),
                "min_size": (
                    "INT",
                    {
                        "default": 100,
                        "min": 1,
                        "max": 500,
                        "step": 1,
                        "display": "slider",
                        "tooltip": "Minimum segment size in pixels as a square side. 1=1x1, 200=200x200; smaller masks are discarded.",
                    },
                ),
                "min_density": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                        "tooltip": "Minimum mask density (foreground pixels / bbox area). 0=disabled, 0.1=at least 10% of bbox must be filled. Filters sparse/feathery masks.",
                    },
                ),
                "fill_holes": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Fill Holes",
                        "label_off": "Keep Holes",
                        "tooltip": "When enabled, fills holes inside each mask (solid segments).",
                    },
                ),
                "dilation": (
                    "INT",
                    {
                        "default": 0,
                        "min": -512,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Expand (positive) or shrink (negative) the mask boundary. 0 = no change.",
                    },
                ),
            },
            "optional": {
                "text_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "placeholder": "e.g., 'cat', 'person in red', 'car'",
                        "tooltip": "Text to guide segmentation (optional).",
                    },
                ),
                "sam3_selectors_pipe": (
                    "SAM3_PROMPT_PIPELINE",
                    {"tooltip": "Unified pipeline containing boxes/points)."},
                ),
                "mask_prompt": (
                    "MASK",
                    {"tooltip": "Optional mask to refine the segmentation."},
                ),
            },
        }

    RETURN_TYPES = (
        "MASK",
        "IMAGE",
        "STRING",
        "STRING",
        "SEGS",
        "MASK",
        "SEGS",
        "SEGS",
    )
    RETURN_NAMES = (
        "masks",
        "visualization",
        "boxes",
        "scores",
        "segs",
        "combined_mask",
        "combined_segs",
        "overlapping_segs",
    )
    FUNCTION = "segment"
    CATEGORY = "SAM3BS"

    def segment(
        self,
        sam3_model,
        image,
        confidence_threshold=0.2,
        detect_all=True,
        pipeline_mode="all",
        instances=False,
        crop_factor=1.5,
        min_size=32,
        min_density=0.0,
        fill_holes=False,
        dilation=0,
        text_prompt="",
        sam3_selectors_pipe=None,
        mask_prompt=None,
        max_detections=10,
    ):
        actual_max_detections = -1 if detect_all else max_detections

        positive_boxes, negative_boxes, positive_points, negative_points = (
            prompt_handler.extract_by_mode(
                sam3_selectors_pipe,
                pipeline_mode,
            )
        )
        print(
            f"[SAM3] pipeline_mode='{pipeline_mode}', instances={instances} | "
            f"pos_boxes={prompt_handler.valid_block(positive_boxes, 'boxes')}, "
            f"neg_boxes={prompt_handler.valid_block(negative_boxes, 'boxes')}, "
            f"pos_points={prompt_handler.valid_block(positive_points, 'points')}, "
            f"neg_points={prompt_handler.valid_block(negative_points, 'points')}"
        )

        ensure_model_on_device(sam3_model)
        processor = sam3_model["processor"]
        print("[SAM3] Running segmentation")
        print(f"[SAM3] Confidence threshold: {confidence_threshold}")

        pil_image = comfy_image_to_pil(image)
        print(f"[SAM3] Image size: {pil_image.size}")

        _, height, width, _ = image.shape
        processor.set_confidence_threshold(confidence_threshold)
        state = processor.set_image(pil_image)

        if text_prompt and text_prompt.strip():
            print(f"[SAM3] Using text_prompt='{text_prompt.strip()}'")
            state = processor.set_text_prompt(text_prompt.strip(), state)

        all_boxes, all_box_labels = prompt_handler.aggregate_prompts(
            positive_boxes, negative_boxes, "boxes"
        )
        print(f"[SAM3] total box prompts={len(all_boxes)}")
        if all_boxes:
            state = processor.add_multiple_box_prompts(all_boxes, all_box_labels, state)

        all_points, all_point_labels = prompt_handler.aggregate_prompts(
            positive_points, negative_points, "points"
        )
        print(f"[SAM3] total point prompts={len(all_points)}")
        if all_points:
            state = processor.add_point_prompt(all_points, all_point_labels, state)

        if mask_prompt is not None:
            if not isinstance(mask_prompt, torch.Tensor):
                mask_prompt = torch.from_numpy(mask_prompt)
            mask_prompt = mask_prompt.to(sam3_model["device"])
            print("[SAM3] Adding external mask_prompt")
            state = processor.add_mask_prompt(mask_prompt, state)

        masks = state.get("masks", None)
        boxes = state.get("boxes", None)
        scores = state.get("scores", None)

        total_scores = len(scores) if scores is not None else 0
        print(f"[SAM3 DEBUG] RAW PREDICTIONS: total {total_scores}")
        if boxes is not None:
            print(f"[SAM3 DEBUG] Output boxes shape: {boxes.shape}")

        device_before = masks.device if masks is not None else "cpu"

        before_count = len(masks) if masks is not None else 0
        masks, boxes, scores = mask_filters.filter_by_size(
            masks, boxes, scores, min_size
        )
        if masks is None:
            print(f"[SAM3] All {before_count} detections removed by min_size filter")
            offload_model_if_needed(sam3_model)
            return lib_types.empty_segmentation_result(
                height, width, pil_to_comfy_image, pil_image, device=device_before
            )
        if len(masks) < before_count:
            print(
                f"[SAM3] min_size={min_size}px removed {before_count - len(masks)} masks, keeping {len(masks)}"
            )

        before_count = len(masks)
        masks, boxes, scores = mask_filters.filter_by_density(
            masks, boxes, scores, min_density
        )
        if masks is None:
            print(f"[SAM3] All {before_count} detections removed by min_density filter")
            offload_model_if_needed(sam3_model)
            return lib_types.empty_segmentation_result(
                height, width, pil_to_comfy_image, pil_image, device=device_before
            )
        if len(masks) < before_count:
            print(
                f"[SAM3] min_density={min_density:.2f} removed {before_count - len(masks)} sparse masks, keeping {len(masks)}"
            )

        if masks is None or len(masks) == 0:
            print(f"[SAM3] No detections found at threshold {confidence_threshold}")
            offload_model_if_needed(sam3_model)
            return lib_types.empty_segmentation_result(
                height, width, pil_to_comfy_image, pil_image, device=device_before
            )

        if instances and boxes is not None:
            print(
                "[SAM3] Instances filter: keep only detections overlapping positive boxes / containing positive points"
            )
            before_instances = len(boxes)
            print(
                f"[SAM3] Instances filter: total detections before filter={before_instances}"
            )
            boxes_device = boxes.device
            masks, boxes, scores = mask_filters.filter_by_instances(
                masks,
                boxes,
                scores,
                positive_boxes,
                positive_points,
                width,
                height,
                iou_threshold=0.1,
            )
            if masks is None:
                print(
                    "[SAM3] Instances filter removed all detections; returning empty result"
                )
                offload_model_if_needed(sam3_model)
                return lib_types.empty_segmentation_result(
                    height, width, pil_to_comfy_image, pil_image, device=boxes_device
                )
            print(
                f"[SAM3] Instances filter kept {len(masks)} of {before_instances} detections"
            )

        masks, boxes, scores = mask_filters.limit_detections(
            masks, boxes, scores, actual_max_detections
        )

        if fill_holes:
            masks = mask_ops.apply_per_mask(masks, mask_ops.fill_holes)

        if dilation != 0:
            masks = mask_ops.apply_per_mask(
                masks, lambda m: mask_ops.dilate_erode(m, dilation)
            )

        comfy_masks = masks_to_comfy_mask(masks)
        combined_tensor = mask_ops.build_combined_mask(masks)
        combined_mask = masks_to_comfy_mask(combined_tensor)
        vis_image = visualize_masks_on_image(pil_image, masks, boxes, scores, alpha=0.5)
        vis_tensor = pil_to_comfy_image(vis_image)

        def tensor_to_list_safe(t):
            if t is None:
                return []
            return tensor_to_list(t)

        boxes_json = json.dumps(tensor_to_list_safe(boxes), indent=2)
        scores_json = json.dumps(tensor_to_list_safe(scores), indent=2)

        segs = build_detection_segs(masks, text_prompt, width, height, crop_factor)
        combined_segs = build_combined_segs(
            combined_tensor, text_prompt, width, height, crop_factor
        )
        overlapping_segs = build_overlapping_segs(
            masks, text_prompt, width, height, crop_factor
        )
        combined_count = (
            len(combined_segs[1])
            if isinstance(combined_segs, tuple) and len(combined_segs) > 1
            else 0
        )
        overlapping_count = (
            len(overlapping_segs[1])
            if isinstance(overlapping_segs, tuple) and len(overlapping_segs) > 1
            else 0
        )

        print(
            f"[SAM3] Segmentation complete. {len(comfy_masks)} masks, {len(segs[1])} SEGS, "
            f"combined_segs has {combined_count} elements, "
            f"overlapping_segs has {overlapping_count} groups."
        )

        offload_model_if_needed(sam3_model)
        return (
            comfy_masks,
            vis_tensor,
            boxes_json,
            scores_json,
            segs,
            combined_mask,
            combined_segs,
            overlapping_segs,
        )


class SAM3BSFlorence2SEGSCaptioner:
    """Caption each SEG with Florence2, CLIP-encode, and bake conditioning in.

    Replaces the multi-node chain:
        SEGSPreview → Florence2Run → StringListToString → AttachConditioning

    For each SEG the node crops the original image to the SEG's region,
    masks out non-segment pixels (so Florence2 sees only the object),
    generates a caption, optionally combines it with a user prompt,
    CLIP-encodes the result, and attaches it as per-SEG conditioning
    via ``control_net_wrapper``.

    Requires kijai's ComfyUI-Florence2 extension for the FL2MODEL type.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "segs": ("SEGS",),
                "florence2_model": ("FL2MODEL",),
                "clip": ("CLIP",),
                "task": (TASK_LIST, {"default": "caption"}),
                "prompt_mode": (
                    ["prepend", "append", "generated only", "user only"],
                    {"default": "prepend"},
                ),
                "conditioning_mode": (["replace", "concat"],),
            },
            "optional": {
                "text_input": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "num_beams": ("INT", {"default": 3, "min": 1, "max": 64}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("SEGS", "STRING", "IMAGE")
    RETURN_NAMES = ("segs", "captions", "bbox_preview")
    FUNCTION = "doit"
    CATEGORY = "SAM3BS"

    def doit(
        self,
        image,
        segs,
        florence2_model,
        clip,
        task,
        prompt_mode,
        conditioning_mode,
        text_input="",
        keep_model_loaded=False,
        max_new_tokens=1024,
        num_beams=3,
        do_sample=True,
        seed=None,
    ):
        import comfy.model_management as mm

        shape, seg_list = segs

        if not seg_list:
            empty_img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return ((shape, []), "", empty_img)

        # --- Device management (matches kijai's Florence2Run pattern) ---
        fl2_model = florence2_model["model"]
        processor = florence2_model["processor"]
        dtype = florence2_model["dtype"]
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        fl2_model.to(device)

        new_segs: list = []
        captions: list[str] = []
        bbox_crops: list = []

        try:
            for seg in seg_list:
                # --- Crop original image to tight bbox ---
                pil_crop = crop_seg_bbox(image=image, bbox=seg.bbox)
                bbox_crops.append(pil_crop)

                # --- Florence2 captioning ---
                generated = caption_image(
                    model=fl2_model,
                    processor=processor,
                    dtype=dtype,
                    pil_image=pil_crop,
                    task=task,
                    device=device,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    seed=seed,
                )

                # --- Combine with user prompt ---
                final_prompt = build_caption(
                    generated=generated,
                    user_prompt=text_input,
                    prompt_mode=prompt_mode,
                )
                captions.append(final_prompt)

                # --- CLIP encode → conditioning ---
                tokens = clip.tokenize(final_prompt)
                cond, pooled = clip.encode_from_tokens(
                    tokens,
                    return_pooled=True,
                )
                conditioning = [[cond, {"pooled_output": pooled}]]

                # --- Wrap and attach to SEG ---
                wrapper = ConditioningOverrideWrapper(
                    conditioning=conditioning,
                    mode=conditioning_mode,
                    original_wrapper=seg.control_net_wrapper,
                )

                new_segs.append(
                    SEG(
                        seg.cropped_image,
                        seg.cropped_mask,
                        seg.confidence,
                        seg.crop_region,
                        seg.bbox,
                        seg.label,
                        wrapper,
                    )
                )
        finally:
            # --- Offload Florence2 model ---
            if not keep_model_loaded:
                fl2_model.to(offload_device)
                mm.soft_empty_cache()

        all_captions = "\n".join(captions)
        preview_tensor = bbox_crops_to_tensor(bbox_crops)
        return ((shape, new_segs), all_captions, preview_tensor)


NODE_CLASS_MAPPINGS = {
    "SAM3BSModelLoader": SAM3BSModelLoaderAndDownloader,
    "SAM3BSSegmentation": SAM3BSSegmentation,
    "SAM3BSFlorence2SEGSCaptioner": SAM3BSFlorence2SEGSCaptioner,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM3BSModelLoader": "SAM3BS Model Loader",
    "SAM3BSSegmentation": "SAM3BS Segmentation",
    "SAM3BSFlorence2SEGSCaptioner": "SAM3BS Florence2 SEGS Captioner",
}
