import os
import cv2
import numpy as np
from typing import List
from shapely import affinity
from shapely.geometry import Polygon
from tqdm import tqdm

# from .ballon_extractor import extract_ballon_region
from . import text_render
from .text_render_eng import render_textblock_list_eng
from .text_render_pillow_eng import render_textblock_list_eng as render_textblock_list_eng_pillow
from ..utils import (
    BASE_PATH,
    TextBlock,
    color_difference,
    get_logger,
    rotate_polygons,
)

logger = get_logger('render')

def parse_font_paths(path: str, default: List[str] = None) -> List[str]:
    if path:
        parsed = path.split(',')
        parsed = list(filter(lambda p: os.path.isfile(p), parsed))
    else:
        parsed = default or []
    return parsed

def fg_bg_compare(fg, bg):
    fg_avg = np.mean(fg)
    if color_difference(fg, bg) < 30:
        bg = (255, 255, 255) if fg_avg <= 127 else (0, 0, 0)
    return fg, bg

def count_text_length(text: str) -> float:
    """Calculate text length, treating っッぁぃぅぇぉ as 0.5 characters"""
    half_width_chars = 'っッぁぃぅぇぉ'  
    length = 0.0
    for char in text.strip():
        if char in half_width_chars:
            length += 0.5
        else:
            length += 1.0
    return length

def get_optimal_font_size(region: 'TextBlock', min_fs: int, max_fs: int) -> int:
    """
    Binary search to find the largest font size that fits within the region's unrotated size.
    """
    low = min_fs
    high = max_fs
    best_size = min_fs
    
    # Pre-fetch dimensions to avoid re-calculating inside loop
    box_w, box_h = region.unrotated_size
    
    while low <= high:
        mid = (low + high) // 2
        if mid <= 0: # Safety check
            low = mid + 1
            continue
            
        fits = False
        
        if region.horizontal:
            # Check if total height of lines <= box height
            lines, _ = text_render.calc_horizontal(
                mid, 
                region.translation, 
                max_width=box_w, 
                max_height=box_h, 
                language=getattr(region, "target_lang", "en_US")
            )
            # Estimate total height (approximate 1.2x line spacing if not provided by renderer)
            # If calc_horizontal respects max_height, it might truncate. 
            # We assume it returns all lines that fit width-wise.
            total_text_h = len(lines) * (mid * 1.05) # Using tight spacing for check
            
            if total_text_h <= box_h and len(lines) > 0:
                fits = True
                
        else: # Vertical
            # Check if total width of columns <= box width
            lines, _ = text_render.calc_vertical(
                mid, 
                region.translation, 
                max_height=box_h
            )
            total_text_w = len(lines) * (mid * 1.05)
            if total_text_w <= box_w and len(lines) > 0:
                fits = True

        if fits:
            best_size = mid
            low = mid + 1 # Try larger
        else:
            high = mid - 1 # Too big, try smaller
            
    return best_size

def resize_regions_to_font_size(img: np.ndarray, text_regions: List['TextBlock'], font_size_fixed: int, font_size_offset: int, font_size_minimum: int):  
    """
    Adjust text region size to accommodate font size and translated text length.
    
    Args:  
        img: Input image
        text_regions: List of text regions to process
        font_size_fixed: Fixed font size (overrides other font parameters)
        font_size_offset: Font size offset
        font_size_minimum: Minimum font size (-1 for auto-calculation)

    Returns:  
        List of adjusted text region bounding boxes
    """    
    
    # Define minimum font size
    if font_size_minimum == -1:  
        font_size_minimum = round((img.shape[0] + img.shape[1]) / 400)  
    font_size_minimum = max(1, font_size_minimum)  

    dst_points_list = []  
    for region in text_regions: 
    
        # Store and validate original font size
        original_region_font_size = region.font_size  
        if original_region_font_size <= 0:  
            original_region_font_size = font_size_minimum

        # --- STEP 1: Determine Target Font Size ---
        if font_size_fixed is not None:
            # User forced a specific size
            target_font_size = font_size_fixed
        elif font_size_offset != 0:
            # User forced an offset relative to detected size
            target_font_size = original_region_font_size + font_size_offset
        else:
            # AUTO MODE: Binary Search for Max Size
            # Search range: from minimum up to 1.5x the smallest box dimension (sanity cap)
            box_min_dim = min(region.unrotated_size)
            search_max = int(max(box_min_dim * 1.5, font_size_minimum + 10))
            
            target_font_size = get_optimal_font_size(region, font_size_minimum, search_max) + 10

        # Enforce minimums
        target_font_size = max(target_font_size, font_size_minimum, 1)  

        # --- STEP 2: Logic to Expand Box if Font is Too Small or Doesn't Fit ---
        # Even if we found an "optimal" size, it might be clamped to font_size_minimum.
        # If target_font_size (now clamped) is still too big for the box, we expand the box.
        
        single_axis_expanded = False
        dst_points = None
        
        if region.horizontal: 
            used_rows = len(region.texts)
            
            # Use TARGET font size for calculation, not the old region.font_size
            line_text_list, _ = text_render.calc_horizontal(
                target_font_size,
                region.translation,
                max_width=region.unrotated_size[0],
                max_height=region.unrotated_size[1],
                language=getattr(region, "target_lang", "en_US")
            )
            needed_rows = len(line_text_list)
            
            # Check if expansion is needed (either by row count or if binary search hit the floor)
            if needed_rows > used_rows:
                scale_x = ((needed_rows - used_rows) / used_rows) * 1 + 1
                try:  
                    poly = Polygon(region.unrotated_min_rect[0])
                    minx, miny, maxx, maxy = poly.bounds
                    poly = affinity.scale(poly, xfact=scale_x, yfact=1.0, origin=(minx, miny))        
                
                    pts = np.array(poly.exterior.coords[:4])  
                    dst_points = rotate_polygons(  
                        region.center, pts.reshape(1, -1), -region.angle,  
                        to_int=False  
                    ).reshape(-1, 4, 2)  
                    dst_points = dst_points.astype(np.int64)
                    single_axis_expanded = True
                except Exception as e:  
                    pass
                    
        if region.vertical:
            used_cols = len(region.texts)
            
            line_text_list, _ = text_render.calc_vertical(
                target_font_size, 
                region.translation, 
                max_height=region.unrotated_size[1],
            )
            needed_cols = len(line_text_list)
            
            if needed_cols > used_cols:
                scale_x = ((needed_cols - used_cols) / used_cols) * 1 + 1
                try:  
                    poly = Polygon(region.unrotated_min_rect[0])
                    minx, miny, maxx, maxy = poly.bounds
                    poly = affinity.scale(poly, xfact=1.0, yfact=scale_x, origin=(minx, miny))                    
                    
                    pts = np.array(poly.exterior.coords[:4])  
                    dst_points = rotate_polygons(  
                        region.center, pts.reshape(1, -1), -region.angle,  
                        to_int=False  
                    ).reshape(-1, 4, 2)  
                    dst_points = dst_points.astype(np.int64)
                    single_axis_expanded = True
                except Exception as e:  
                    pass

        # If single-axis expansion failed, use general scaling
        if not single_axis_expanded:
            # Calculate scaling factor based on text length ratio
            orig_text = getattr(region, "text_raw", region.text)
            char_count_orig = count_text_length(orig_text)
            char_count_trans = count_text_length(region.translation.strip())     
            length_ratio = 1.0

            if char_count_orig > 0 and char_count_trans > char_count_orig:  
                increase_percentage = (char_count_trans - char_count_orig) / char_count_orig
                font_increase_ratio = 1 + (increase_percentage * 0.2)
                font_increase_ratio = min(1.5, max(1.0, font_increase_ratio))
                
                # We already calculated target_font_size via binary search (max possible).
                # If we are here, it means we are expanding strictly due to length ratio.
                # However, usually binary search handles "fitting", so this is a fallback for density.
                
                # If we used binary search, target_font_size is already maximized for the box.
                # Increasing it further requires box expansion.
                if font_size_fixed is None:
                     target_font_size = int(target_font_size * font_increase_ratio)

                target_scale = max(1, min(1 + increase_percentage * 0.2, 2))  
            else:  
                target_scale = 1              

            # Calculate final scaling factor
            # Ensure we don't scale down if we already set a target size
            font_size_scale = (((target_font_size - original_region_font_size) / original_region_font_size) * 0.4 + 1) if original_region_font_size > 0 else 1.0  
            final_scale = max(font_size_scale, target_scale)
            final_scale = max(1, min(final_scale, 1.1))  

            # Scale bounding box if needed
            if final_scale > 1.001:  
                try:  
                    poly = Polygon(region.unrotated_min_rect[0])  
                     # Scale from the center  
                    poly = affinity.scale(poly, xfact=final_scale, yfact=final_scale, origin='center')  
                    scaled_unrotated_points = np.array(poly.exterior.coords[:4])  

                    dst_points = rotate_polygons(region.center, scaled_unrotated_points.reshape(1, -1), -region.angle, to_int=False).reshape(-1, 4, 2)  
                    dst_points = dst_points.astype(np.int64)  
                    dst_points = dst_points.reshape((-1, 4, 2))  

                except Exception as e:  
                    dst_points = region.min_rect
            else:
                dst_points = region.min_rect

        # Store results and update font size
        dst_points_list.append(dst_points)  
        region.font_size = int(target_font_size)

    return dst_points_list

async def dispatch(
    img: np.ndarray,
    text_regions: List[TextBlock],
    font_path: str = '',
    font_size_fixed: int = None,
    font_size_offset: int = 0,
    font_size_minimum: int = 0,
    hyphenate: bool = True,
    render_mask: np.ndarray = None,
    line_spacing: int = None,
    disable_font_border: bool = False
    ) -> np.ndarray:
    
    text_render.set_font(font_path)
    text_regions = list(filter(lambda region: region.translation, text_regions))

    # Resize regions that are too small
    dst_points_list = resize_regions_to_font_size(img, text_regions, font_size_fixed, font_size_offset, font_size_minimum)
    
    # Render text
    for region, dst_points in tqdm(zip(text_regions, dst_points_list), '[render]', total=len(text_regions)):
        if render_mask is not None:
            cv2.fillConvexPoly(render_mask, dst_points.astype(np.int32), 1)
        
        img = render(img, region, dst_points, hyphenate, line_spacing, disable_font_border)

        # # --- VISUALIZATION (RED BOXES) ---
        # if region.lines is not None:
        #      for line in region.lines:
        #         pts = line.reshape((-1, 1, 2)).astype(np.int32)
        #         cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        # # ---------------------------------

    return img

def render(
    img,
    region: TextBlock,
    dst_points,
    hyphenate,
    line_spacing,
    disable_font_border
):
    fg, bg = region.get_font_colors()
    fg, bg = fg_bg_compare(fg, bg)

    if disable_font_border:
        bg = None

    middle_pts = (dst_points[:, [1, 2, 3, 0]] + dst_points) / 2
    norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
    norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
    r_orig = np.mean(norm_h / norm_v)

    forced_direction = region._direction if hasattr(region, "_direction") else region.direction
    if forced_direction != "auto":
        if forced_direction in ["horizontal", "h"]:
            render_horizontally = True
        elif forced_direction in ["vertical", "v"]:
            render_horizontally = False
        else:
            render_horizontally = region.horizontal
    else:
        render_horizontally = region.horizontal

    # Safely generate the text canvas block
    if render_horizontally:
        temp_box = text_render.put_text_horizontal(
            region.font_size, region.get_translation_for_rendering(),
            round(norm_h[0]), round(norm_v[0]), region.alignment,
            region.direction == 'hl', fg, bg, region.target_lang,
            hyphenate, line_spacing,
        )
    else:
        temp_box = text_render.put_text_vertical(
            region.font_size, region.get_translation_for_rendering(),
            round(norm_v[0]), region.alignment, fg, bg, line_spacing,
        )
        
    if temp_box is None or temp_box.size == 0:
        return img

    h, w, _ = temp_box.shape
    r_temp = w / h
    box = None  
    
    if render_horizontally:  
        if r_temp > r_orig:   
            h_ext = int((w / r_orig - h) // 2) if r_orig > 0 else 0  
            if h_ext >= 0:  
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)  
                box[h_ext:h_ext+h, 0:w] = temp_box  
            else:  
                box = temp_box.copy()  
        else:   
            w_ext = int((h * r_orig - w) // 2)  
            if w_ext >= 0:  
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)  
                box[0:h, 0:w] = temp_box  
            else:  
                box = temp_box.copy()  
    else:  
        if r_temp > r_orig:   
            h_ext = int(w / (2 * r_orig) - h / 2) if r_orig > 0 else 0   
            if h_ext >= 0:  
                box = np.zeros((h + h_ext * 2, w, 4), dtype=np.uint8)  
                box[0:h, 0:w] = temp_box  
            else:  
                box = temp_box.copy()   
        else:   
            w_ext = int((h * r_orig - w) / 2)  
            if w_ext >= 0:  
                box = np.zeros((h, w + w_ext * 2, 4), dtype=np.uint8)  
                box[0:h, w_ext:w_ext+w] = temp_box  
            else:  
                box = temp_box.copy()   

    # Setup source rendering points
    src_points = np.array([[0, 0], [box.shape[1], 0], [box.shape[1], box.shape[0]], [0, box.shape[0]]]).astype(np.float32)

    # --- ENHANCED FORCE CLAMPING LOGIC ---
    # 1. Get the boundary of where the text wants to go
    img_h, img_w = img.shape[:2]
    x, y, w_b, h_b = cv2.boundingRect(dst_points.astype(np.int32))
    # 2. Check if text box falls outside or threatens to exceed canvas limits
    if x < 0 or y < 0 or (x + w_b) > img_w or (y + h_b) > img_h:
        # Calculate corrective directional offsets
        offset_x = 0
        offset_y = 0
        if x < 0: offset_x = -x
        if y < 0: offset_y = -y
        if (x + w_b) > img_w: offset_x = img_w - (x + w_b)
        if (y + h_b) > img_h: offset_y = img_h - (y + h_b)
        
        # Modify the coordinates safely using the explicit (1, 4, 2) shape indices
        dst_points[0, :, 0] += offset_x
        dst_points[0, :, 1] += offset_y
        
        # Absolute safety fallback clamp to prevent decimal overflow anomalies
        dst_points[0, :, 0] = np.clip(dst_points[0, :, 0], 0, img_w)
        dst_points[0, :, 1] = np.clip(dst_points[0, :, 1], 0, img_h)
        
    # Run standard warp perspective
    M, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    rgba_region = cv2.warpPerspective(box, M, (img_w, img_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    
    # Recalculate safe destination dimensions within exact bounds
    x, y, w_b, h_b = cv2.boundingRect(dst_points.astype(np.int32))
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_w, x + w_b), min(img_h, y + h_b)

    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return img
    
    canvas_region = rgba_region[y1:y2, x1:x2, :3]
    mask_region = rgba_region[y1:y2, x1:x2, 3:4].astype(np.float32) / 255.0
    
    img[y1:y2, x1:x2] = np.clip(
        (img[y1:y2, x1:x2].astype(np.float32) * (1 - mask_region) + canvas_region.astype(np.float32) * mask_region), 
        0, 255
    ).astype(np.uint8)
    
    return img

async def dispatch_eng_render(img_canvas: np.ndarray, original_img: np.ndarray, text_regions: List[TextBlock], font_path: str = '', line_spacing: int = 0, disable_font_border: bool = False) -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if not font_path:
        font_path = os.path.join(BASE_PATH, 'fonts/comic shanns 2.ttf')
    text_render.set_font(font_path)

    return render_textblock_list_eng(img_canvas, text_regions, line_spacing=line_spacing, size_tol=1.2, original_img=original_img, downscale_constraint=0.8,disable_font_border=disable_font_border)

async def dispatch_eng_render_pillow(img_canvas: np.ndarray, original_img: np.ndarray, text_regions: List[TextBlock], font_path: str = '', line_spacing: int = 0, disable_font_border: bool = False) -> np.ndarray:
    if len(text_regions) == 0:
        return img_canvas

    if not font_path:
        font_path = os.path.join(BASE_PATH, 'fonts/NotoSansMonoCJK-VF.ttf.ttc')
    text_render.set_font(font_path)

    return render_textblock_list_eng_pillow(font_path, img_canvas, text_regions, original_img=original_img, downscale_constraint=0.95)