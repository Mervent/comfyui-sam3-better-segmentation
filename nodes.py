"""
ComfyUI SAM3 Nodes, unified model loader for both image and video using official Meta sam3_lib.
All class names and functions prefixed with SAM3BS for uniqueness.
"""

import json
import logging
import os

import torch

import folder_paths

from .lib import mask_filters, mask_ops, prompt_handler
from .lib import types as lib_types
from .lib.conditioning_wrapper import ConditioningOverrideWrapper
from .lib.florence2_captioner import (
    TASK_LIST,
    batch_caption_images,
    bbox_crops_to_tensor,
    build_caption,
    crop_image_region,
)
from .lib.masktosegs import SEG
from .lib.model_manager import download_sam3_model, get_available_models, get_model_path
from .lib.sam3_utils import (
    comfy_image_to_pil,
    masks_to_comfy_mask,
    offload_model_if_needed,
    pil_to_comfy_image,
    run_sam3_inference,
    tensor_to_list,
    visualize_masks_on_image,
)
from .lib.segs_builder import (
    build_combined_segs,
    build_detection_segs,
    build_overlapping_segs,
)
from .sam3_lib.model.sam3_image_processor import Sam3Processor
from .sam3_lib.model_builder import build_sam3_image_model

logger = logging.getLogger(__name__)


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
        available = get_available_models(models_dir=folder_paths.models_dir)
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
        """Build and return a SAM3_MODEL dict: {model, processor, device, original_device}."""
        hf_repo = "facebook/sam3"
        # Resolve checkpoint path if needed
        checkpoint_path = None

        if model_source == "auto (API to cache)":
            # Let builder construct its default weights / config
            logger.info("Using API/default SAM3 image model.")
            checkpoint_path = None

        elif model_source == "local (auto-download)":
            # Download only sam3.pt into models/sam3
            sam3_dir = download_sam3_model(hf_repo, models_dir=folder_paths.models_dir)
            checkpoint_path = os.path.join(sam3_dir, "sam3.pt")
            if not os.path.isfile(checkpoint_path):
                raise RuntimeError(
                    f"[SAM3BSModelLoaderAndDownloader] Downloaded model file not found at: {checkpoint_path}"
                )
            logger.info(f"Using downloaded local checkpoint: {checkpoint_path}")

        else:
            # Specific local checkpoint chosen from list under models/sam3
            checkpoint_path = get_model_path(
                model_source, models_dir=folder_paths.models_dir
            )
            if not checkpoint_path or not os.path.isfile(checkpoint_path):
                raise RuntimeError(
                    f"[SAM3BSModelLoaderAndDownloader] Local model file not found: {model_source} -> {checkpoint_path}"
                )
            logger.info(f"Using selected local checkpoint: {checkpoint_path}")

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

        logger.info(f"SAM3 model ready on device: {device}")
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
                        "default": 0.2,
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
                        "default": 32,
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
        max_detections=50,
    ):
        actual_max_detections = -1 if detect_all else max_detections

        positive_boxes, negative_boxes, positive_points, negative_points = (
            prompt_handler.extract_by_mode(
                sam3_selectors_pipe,
                pipeline_mode,
            )
        )
        logger.info(
            f"pipeline_mode='{pipeline_mode}', instances={instances} | "
            f"pos_boxes={prompt_handler.valid_block(positive_boxes, 'boxes')}, "
            f"neg_boxes={prompt_handler.valid_block(negative_boxes, 'boxes')}, "
            f"pos_points={prompt_handler.valid_block(positive_points, 'points')}, "
            f"neg_points={prompt_handler.valid_block(negative_points, 'points')}"
        )

        all_boxes, all_box_labels = prompt_handler.aggregate_prompts(
            positive_boxes, negative_boxes, "boxes"
        )
        all_points, all_point_labels = prompt_handler.aggregate_prompts(
            positive_points, negative_points, "points"
        )

        pil_image = comfy_image_to_pil(image)
        _, height, width, _ = image.shape

        masks, boxes, scores = run_sam3_inference(
            sam3_model=sam3_model,
            pil_image=pil_image,
            confidence_threshold=confidence_threshold,
            text_prompt=text_prompt,
            box_prompts=all_boxes,
            box_labels=all_box_labels,
            point_prompts=all_points,
            point_labels=all_point_labels,
            mask_prompt=mask_prompt,
        )

        device_before = masks.device if masks is not None else "cpu"

        masks, boxes, scores = mask_filters.run_filter_pipeline(
            masks,
            boxes,
            scores,
            steps=[
                (mask_filters.filter_by_size, (min_size,)),
                (mask_filters.filter_by_density, (min_density,)),
            ],
        )
        if masks is None:
            offload_model_if_needed(sam3_model)
            return lib_types.empty_segmentation_result(
                height, width, pil_to_comfy_image, pil_image, device=device_before
            )

        if instances and boxes is not None:
            logger.info(
                "Instances filter: keep only detections overlapping positive boxes / containing positive points"
            )
            before_instances = len(boxes)
            logger.info(
                f"Instances filter: total detections before filter={before_instances}"
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
                logger.warning(
                    "Instances filter removed all detections; returning empty result"
                )
                offload_model_if_needed(sam3_model)
                return lib_types.empty_segmentation_result(
                    height, width, pil_to_comfy_image, pil_image, device=boxes_device
                )
            logger.info(
                f"Instances filter kept {len(masks)} of {before_instances} detections"
            )

        masks, boxes, scores = mask_filters.limit_detections(
            masks, boxes, scores, actual_max_detections
        )

        if masks is None or masks.numel() == 0:
            offload_model_if_needed(sam3_model)
            return lib_types.empty_segmentation_result(
                height, width, pil_to_comfy_image, pil_image, device=device_before
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

        logger.info(
            f"Segmentation complete. {len(comfy_masks)} masks, {len(segs[1])} SEGS, "
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

    For each SEG the node crops the original image to either the tight
    bounding box or expanded crop region (controlled by ``crop_source``),
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
                "crop_source": (
                    ["bbox", "crop_region"],
                    {"default": "bbox"},
                ),
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
        crop_source="bbox",
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

        # --- Phase 1: Crop all SEGs (CPU, fast) ---
        pil_crops: list = []
        for seg in seg_list:
            region = (
                tuple(seg.crop_region) if crop_source == "crop_region" else seg.bbox
            )
            pil_crops.append(crop_image_region(image=image, region=region))

        # --- Phase 2: Batch Florence2 inference (GPU, chunked) ---
        print(f"[DEBUG doit] {len(pil_crops)} crops, calling batch_caption_images")
        try:
            captions_raw = batch_caption_images(
                model=fl2_model,
                processor=processor,
                dtype=dtype,
                pil_images=pil_crops,
                task=task,
                device=device,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=do_sample,
                seed=seed,
            )
        finally:
            if not keep_model_loaded:
                fl2_model.to(offload_device)
                mm.soft_empty_cache()

        print(
            f"[DEBUG doit] captions_raw type={type(captions_raw).__name__}, len={len(captions_raw)}"
        )
        for idx, cr in enumerate(captions_raw):
            print(
                f"[DEBUG doit] captions_raw[{idx}] type={type(cr).__name__}, value={cr!r}"
            )

        # --- Phase 3: Build captions + CLIP encode per-SEG ---
        new_segs: list = []
        captions: list[str] = []

        for i, seg in enumerate(seg_list):
            print(f"[DEBUG doit] seg[{i}] generated input: {captions_raw[i]!r}")
            final_prompt = build_caption(
                generated=captions_raw[i],
                user_prompt=text_input,
                prompt_mode=prompt_mode,
            )
            print(f"[DEBUG doit] seg[{i}] final_prompt: {final_prompt!r}")
            captions.append(final_prompt)

            tokens = clip.tokenize(final_prompt)
            cond, pooled = clip.encode_from_tokens(
                tokens,
                return_pooled=True,
            )
            conditioning = [[cond, {"pooled_output": pooled}]]

            wrapper = ConditioningOverrideWrapper(
                conditioning=conditioning,
                mode=conditioning_mode,
                original_wrapper=seg.control_net_wrapper,
            )

            new_segs.append(
                SEG(
                    cropped_image=seg.cropped_image,
                    cropped_mask=seg.cropped_mask,
                    confidence=seg.confidence,
                    crop_region=seg.crop_region,
                    bbox=seg.bbox,
                    label=seg.label,
                    control_net_wrapper=wrapper,
                )
            )

        all_captions = "\n".join(captions)
        preview_tensor = bbox_crops_to_tensor(pil_crops)
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
