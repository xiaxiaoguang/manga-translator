from typing import Optional, List, Dict
import numpy as np
import cv2
import os
from ..config import SAM, SAMConfig
from .common import CommonSAM, SAM2Inpainter, OfflineSAM  # You will create these below
from ..utils import TextBlock, Quadrilateral
from ..utils.bubble import is_ignore
from .text_mask_utils import complete_mask_fill, complete_mask

SAM_MODELS = {
    SAM.sam2_1: SAM2Inpainter,
}

sam_cache: Dict[SAM, CommonSAM] = {}


def get_sam_model(key: SAM, *args, **kwargs) -> CommonSAM:
    if key not in SAM_MODELS:
        # Default to a "None" or fallback if not found
        key = SAM.sam2_1
    if not sam_cache.get(key):
        model_class = SAM_MODELS[key]
        sam_cache[key] = model_class(*args, **kwargs)
    return sam_cache[key]


async def prepare_sam(sam_key: SAM, device: str = "cpu"):
    model = get_sam_model(sam_key)
    await model.load(device)

async def dispatch(
    sam_key: SAM,
    text_regions: List[TextBlock],
    raw_image: np.ndarray,
    raw_mask: np.ndarray,
    method: str = "fit_text",
    dilation_offset: int = 0,
    ignore_bubble: int = 0,
    verbose: bool = False,
    kernel_size: int = 3,
    device: str = "cpu",
    config: Optional[SAMConfig] = None,
) -> np.ndarray:

    # os.makedirs("debug_output", exist_ok=True)
    # cv2.imwrite("debug_output/01_raw_mask_input.png", raw_mask)
    # cv2.imwrite("debug_output/01_raw_image.png", raw_image)

    model = get_sam_model(sam_key)
    if isinstance(model, SAM2Inpainter): # Ensure correct class check
        await model.load(device)
    config = config or SAMConfig()

    # --- Step 1: Precise Text Mask Generation ---
    scale_factor = max(min((raw_mask.shape[0] - raw_image.shape[0] / 3) / raw_mask.shape[0], 1), 0.5)
    img_resized = cv2.resize(raw_image, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)))
    mask_resized = cv2.resize(raw_mask, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)))
    mask_resized[mask_resized > 0] = 255

    textlines = []
    for region in text_regions:
        for l in region.lines:
            q = Quadrilateral(l * scale_factor, "", 0)
            textlines.append(q)

    # This creates the 'initial' text-only mask
    refined_text_mask = (
        complete_mask(img_resized, mask_resized, textlines, dilation_offset=dilation_offset, kernel_size=kernel_size)
        if method == "fit_text"
        else complete_mask_fill([txtln.aabb.xywh for txtln in textlines])
    )

    if refined_text_mask is None:
        refined_text_mask = np.zeros(raw_image.shape[:2], dtype=np.uint8)
    else:
        refined_text_mask = cv2.resize(refined_text_mask, (raw_image.shape[1], raw_image.shape[0]))
        refined_text_mask[refined_text_mask > 0] = 255
    canvas_prompts = raw_image.copy()
    canvas_results = raw_image.copy()
    mask_overlay = np.zeros_like(raw_image)
    # --- Step 2: SAM Refinement (Ratio-based Expansion) ---
    if sam_key != SAM.none and text_regions:
        img_h, img_w = raw_image.shape[:2]
        # Ratio of expansion (0.5 means add 50% of width/height to each side)
        expand_ratio = 0
        prompt_boxes = []

        for region in text_regions:
            x1, y1, x2, y2 = region.xyxy
            w = x2 - x1
            h = y2 - y1

            # Calculate expansion amounts
            dw = w * expand_ratio
            tmp = dw
            dh = h * expand_ratio
            dw = dh
            dh = tmp
            
            # Apply expansion while staying within image boundaries
            px1 = max(0, x1 - dw)
            py1 = max(0, y1 - dh)
            px2 = min(img_w, x2 + dw)
            py2 = min(img_h, y2 + dh)
            
            prompt_boxes.append([float(px1), float(py1), float(px2), float(py2)])            
            # Draw the padded box on the prompt canvas (Red)
            cv2.rectangle(canvas_prompts, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 2)

        # Run Prediction
        sam_masks = await model.predict(raw_image, prompt_boxes, config, verbose)
        bubble_combined_mask = np.zeros_like(refined_text_mask)

        for i, m in enumerate(sam_masks):
            region = text_regions[i]
            
            # Logic check: OSB vs Bubble
            is_light_bg = np.mean(region.bg_colors) > 200
            is_likely_osb = not is_light_bg or abs(region.angle) > 5.0
            region.is_osb = is_likely_osb

            if is_likely_osb:
                osb_mask = np.zeros_like(refined_text_mask)
                cv2.fillPoly(osb_mask, [region.min_rect.astype(np.int32)], 255)
                m = cv2.dilate(osb_mask, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
            
            region.bubble_mask = m
            bubble_combined_mask = cv2.bitwise_or(bubble_combined_mask, m)
            
            # Add to global green overlay
            mask_overlay[m > 0] = [0, 255, 0]

        # Blend the green masks onto the results canvas
        canvas_results = cv2.addWeighted(canvas_results, 0.7, mask_overlay, 0.3, 0)

        # Create the side-by-side full image visualization
        # Resize for easier viewing if the manga page is massive (optional)
        global_comparison = np.hstack([canvas_prompts, canvas_results])
        
        # Save the full page debug image
        # cv2.imwrite("debug_output/00_GLOBAL_SAM_REFINE.jpg", global_comparison)

        # Merge results
        final_mask = cv2.bitwise_or(refined_text_mask, bubble_combined_mask)
        
    # --- Step 3: Cleanup / Ignore logic ---
    if 1 <= ignore_bubble <= 50:
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            temp_mask = np.zeros_like(final_mask)
            cv2.rectangle(temp_mask, (x, y), (x+w, y+h), 255, -1)
            if is_ignore(cv2.bitwise_and(raw_image, raw_image, mask=temp_mask), ignore_bubble):
                cv2.drawContours(final_mask, [cnt], -1, 0, -1)

    # cv2.imwrite("debug_output/02_final_refined_mask.png", final_mask)
    return final_mask