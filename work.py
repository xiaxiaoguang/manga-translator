import os
import time
import torch
import gc
import numpy as np
import configparser
import shutil
import argparse
import re
import random
from PIL import Image, ImageFilter, ImageChops, ImageOps, ImageEnhance, ImageDraw
from pywinauto import Application, Desktop
from scipy.ndimage import label, binary_dilation, center_of_mass 

from diffusers import DiffusionPipeline
from transformers import Sam2Processor, Sam2Model
from utils.inpainter import DecensorInpainter,DecensorInpainter2
from utils.inpainter2 import DecensorInpainterX
from utils.tools import *

proxy = "http://127.0.0.1:7897" 
os.environ["HTTP_PROXY"] = proxy
os.environ["HTTPS_PROXY"] = proxy
os.environ["HF_HUB_PROXY"] = proxy
# --- CLASS 1: MASK MANAGER (Segmentation, HAI Detection, SAM2 Refinement) ---

class DecensorMaskManager:
    def __init__(self, args):
        self.args = args
        self.hai_dir = os.path.dirname(args.hai_path)
        self.hai_mask_folder = os.path.join(self.hai_dir, "decensor_input")
        self.sam_processor = None
        self.sam_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def preprocess_for_detection(self, img):
        """
        Applies a black-level adjustment to turn dark gray bars into pure black
        for better detection by HAI.
        """
        if self.args.black_level > 0:
            # Create a lookup table for the levels adjustment
            # Pixels <= black_level become 0
            # Pixels > black_level are stretched to fill 0-255
            lookup = []
            for i in range(256):
                if i < self.args.black_level:
                    lookup.append(0)
                else:
                    val = int(255 * (i - self.args.black_level) / (255 - self.args.black_level))
                    lookup.append(min(255, val))
            
            # Apply to image
            if img.mode == 'RGB':
                r, g, b = img.split()
                r = r.point(lookup)
                g = g.point(lookup)
                b = b.point(lookup)
                img = Image.merge('RGB', (r, g, b))
            else:
                img = img.point(lookup)
            
            # Optional: Slight contrast boost after levels
            img = ImageEnhance.Contrast(img).enhance(1.1)
        return img

    def load_sam2(self):
        if self.sam_model is None:
            print(f"正在从本地路径加载 SAM2.1 模型: {SAM2_MODEL_PATH}")
            self.sam_processor = Sam2Processor.from_pretrained(SAM2_MODEL_PATH)
            self.sam_model = Sam2Model.from_pretrained(SAM2_MODEL_PATH).to(self.device)
            print(f"SAM2.1 加载完成，运行设备: {self.device}")

    def unload_sam2(self):
        self.sam_model = None
        self.sam_processor = None
        cleanup_gpu()

    def update_hai_config(self, input_dir):
        config_path = os.path.join(self.hai_dir, "hconfig.ini")
        if os.path.exists(config_path):
            config = configparser.ConfigParser()
            config.read(config_path)
            if 'Paths' not in config: config['Paths'] = {}
            config['Paths']['input'] = input_dir
            with open(config_path, 'w') as f: config.write(f)

    def segment_images_to_temp(self):
        print(f"模式: Segmentation - 正在准备带固定偏置缓冲的 HAI 检测块 (Tile Size: {LOGICAL_TILE_W}x{LOGICAL_TILE_H})...")
        if os.path.exists(self.args.temp_tiles): shutil.rmtree(self.args.temp_tiles)
        os.makedirs(self.args.temp_tiles)
        
        files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        files.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])
        
        for i,f in enumerate(files):
            img = smart_resize(Image.open(os.path.join(self.args.input, f)).convert("RGB"))
            
            # Apply darkening preprocessing here
            if self.args.black_level > 0:
                img = self.preprocess_for_detection(img)

            w, h = img.size
            tile_configs = get_tiles_with_padding((w, h))
            
            for config in tile_configs:
                p_box, l_box = config['padded'], config['logical']
                tile = extract_padded_tile(img, p_box)
                tile_name = f"{i}_T_{l_box[0]}_{l_box[1]}_{l_box[2]}_{l_box[3]}_P_{p_box[0]}_{p_box[1]}_{p_box[2]}_{p_box[3]}.png"
                tile.save(os.path.join(self.args.temp_tiles, tile_name))

    def segment_focused_from_coarse(self):
        print("模式: Segmentation (Focused) - 基于初步检测结果生成重点切片...")
        if os.path.exists(self.args.temp_tiles): shutil.rmtree(self.args.temp_tiles)
        os.makedirs(self.args.temp_tiles)

        files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        files.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])

        for idx, orig_file in enumerate(files):
            orig_path = os.path.join(self.args.input, orig_file)
            img = smart_resize(Image.open(orig_path).convert("RGB"))
            
            if self.args.black_level > 0:
                img = self.preprocess_for_detection(img)
                
            w, h = img.size
            
            full_mask = Image.new("L", (w, h), 0)
            if os.path.exists(self.hai_mask_folder):
                for m_file in os.listdir(self.hai_mask_folder):
                    if m_file.startswith(f"{idx}_T_") and m_file.endswith(".png"):
                        parts = m_file.replace(".png", "").split("_")
                        try:
                            lx1, ly1, lx2, ly2 = map(int, parts[2:6])
                            marked_tile = Image.open(os.path.join(self.hai_mask_folder, m_file))
                            tile_mask = extract_mask_from_green(marked_tile)
                            offset = CONTEXT_PADDING // 2
                            logical_mask_part = tile_mask.crop((offset, offset, offset + (lx2 - lx1), offset + (ly2 - ly1)))
                            full_mask.paste(logical_mask_part, (lx1, ly1))
                        except: continue

            if not full_mask.getbbox(): continue

            mask_arr = np.array(full_mask)
            labeled_array, num_features = label(mask_arr > 128)
            print(f"  [{orig_file}] 发现 {num_features} 个潜在区域，生成聚焦切片...")
            
            # Coverage mask to prevent duplicate tiles for dense bars
            coverage_mask = np.zeros((h, w), dtype=bool)

            for i in range(1, num_features + 1):
                # Check if this island is already covered by previous tiles
                this_island_mask = (labeled_array == i)
                if np.all(coverage_mask[this_island_mask]):
                    continue

                cy, cx = center_of_mass(mask_arr, labeled_array, i)
                cx, cy = int(cx), int(cy)
                
                focus_size = int(h//4)
                half_size = focus_size // 2
                lx1 = max(0, cx - half_size)
                ly1 = max(0, cy - half_size)
                lx2 = min(w, lx1 + focus_size)
                ly2 = min(h, ly1 + focus_size)
                
                if (lx2 - lx1) < focus_size: lx1 = max(0, lx2 - focus_size)
                if (ly2 - ly1) < focus_size: ly1 = max(0, ly2 - focus_size)
                
                # Mark this tile area as covered
                coverage_mask[ly1:ly2, lx1:lx2] = True

                px1 = lx1 - CONTEXT_PADDING
                py1 = ly1 - CONTEXT_PADDING
                px2 = lx2 + CONTEXT_PADDING
                py2 = ly2 + CONTEXT_PADDING
                
                tile = extract_padded_tile(img, (px1, py1, px2, py2))
                tile_name = f"{idx}_T_{lx1}_{ly1}_{lx2}_{ly2}_P_{px1}_{py1}_{px2}_{py2}_focus{i}.png"
                tile.save(os.path.join(self.args.temp_tiles, tile_name))

    def run_hai_detection(self, mosaic_index, clear_existing=True):
        print(f"模式: HAI - 启动自动化检测 (Index: {mosaic_index})...")
        if clear_existing:
            if os.path.exists(self.hai_mask_folder): shutil.rmtree(self.hai_mask_folder)
            os.makedirs(self.hai_mask_folder)
        elif not os.path.exists(self.hai_mask_folder):
             os.makedirs(self.hai_mask_folder)

        try:
            self.update_hai_config(self.args.temp_tiles)
            app = Application(backend="uia").start(self.args.hai_path, wait_for_idle=False)
            time.sleep(15)
            desktop = Desktop(backend="uia")
            main_window = desktop.window(title_re=".*hentAI.*")
            main_window.wait('exists', timeout=20)
            main_window.set_focus()
            btns = main_window.descendants(control_type="Button")
            if len(btns) > mosaic_index:
                btns[mosaic_index].click_input()
                time.sleep(3)
            det_win = desktop.window(title_re=".*Detection.*")
            det_win.wait('exists', timeout=20)
            det_win.set_focus()
            f_btns = [b for b in det_win.descendants(control_type="Button") if b.window_text() not in ["最小化", "最大化", "关闭"]]
            if f_btns: f_btns[1].click_input()
            
            for _ in range(3600):
                try:
                    success = desktop.window(title_re=".*Success.*")
                    if success.exists(): success.close(); break
                except: pass
                time.sleep(1)
            if det_win.exists(): det_win.close()
            app.kill() 
        except Exception as e: print(f"GUI 自动化异常: {e}")

    def merge_detection_results(self):
        if not os.path.exists(self.hai_mask_folder): return
        print("模式: Merge - 正在合并所有检测切片为单一全图掩码...")
        
        input_files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        input_files.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])
        
        for idx, orig_file in enumerate(input_files):
            orig_path = os.path.join(self.args.input, orig_file)
            original_img = smart_resize(Image.open(orig_path).convert("RGB"))
            w, h = original_img.size
            
            full_rough_mask = Image.new("L", (w, h), 0)
            tiles_to_delete = []
            
            for m_file in os.listdir(self.hai_mask_folder):
                if m_file.startswith(f"{idx}_T_") and m_file.endswith(".png"):
                    tiles_to_delete.append(os.path.join(self.hai_mask_folder, m_file))
                    parts = m_file.replace(".png", "").split("_")
                    try:
                        lx1, ly1, lx2, ly2 = map(int, parts[2:6])
                        marked_tile = Image.open(os.path.join(self.hai_mask_folder, m_file))
                        tile_mask = extract_mask_from_green(marked_tile)
                        offset = CONTEXT_PADDING // 2
                        logical_mask_part = tile_mask.crop((offset, offset, offset + (lx2 - lx1), offset + (ly2 - ly1)))
                        full_rough_mask.paste(logical_mask_part, (lx1, ly1))
                    except: continue
            
            if not tiles_to_delete: continue

            p_box = (-CONTEXT_PADDING, -CONTEXT_PADDING, w + CONTEXT_PADDING, h + CONTEXT_PADDING)
            base_canvas = extract_padded_tile(original_img, p_box)
            
            green_canvas = Image.new("RGB", base_canvas.size, (0, 255, 0))
            final_mask_canvas = Image.new("L", base_canvas.size, 0)
            final_mask_canvas.paste(full_rough_mask, (CONTEXT_PADDING // 2, CONTEXT_PADDING // 2))
            
            merged_tile = Image.composite(green_canvas, base_canvas, final_mask_canvas)
            
            for p in tiles_to_delete:
                try: os.remove(p)
                except: pass
                
            save_path = os.path.join(self.hai_mask_folder, f"{idx}_T_0_0_{w}_{h}_merged.png")
            merged_tile.save(save_path)
            print(f"  [{orig_file}] 合并完成 -> {save_path}")


    def refine_masks_with_sam2_points(self, image, initial_mask, debug_stem=""):
        mask_data = np.array(initial_mask)
        labeled_array, num_features = label(mask_data > 128)
        if num_features == 0: return initial_mask
        
        # CHANGE: Initialize with original mask to perform a UNION (Merge) instead of replacement.
        # This ensures we don't lose the original detection coverage if SAM produces a smaller mask 
        # or fails, satisfying the "merge detection results" requirement.
        refined_full_mask_acc = mask_data.copy()
        
        print(f" 检测到 {num_features} 个独立的遮罩区块。正在逐一处理 (Point Mode Only)...")

        for i in range(1, num_features + 1):
            coords = np.argwhere(labeled_array == i)
            center_y, center_x = np.mean(coords, axis=0)
            point = [int(center_x), int(center_y)]
            
            inputs = self.sam_processor(
                images=image, 
                input_points=[[[point]]], 
                input_labels=[[[1]]],
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.sam_model(**inputs)
            
            # Post-process mask based on strategy
            masks = self.sam_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
            masks_np = masks[0].numpy() # Shape: (3, H, W)
            scores_np = outputs.iou_scores.cpu().numpy()[0, 0] # Shape: (3,)
            
            # --- CONTENT VALIDATION & SELECTION ---
            valid_indices = []
            
            if self.args.mosaic == 2: # Black Bar Strategy
                # Convert to grayscale for brightness checking
                img_gray = np.array(image.convert("L"))
                
                for m_idx in range(3):
                    mask_bool = masks_np[m_idx] > 0
                    if np.count_nonzero(mask_bool) == 0: continue
                    
                    masked_pixels = img_gray[mask_bool]
                    
                    # 1. White Pixel Ratio (>200 is considered white/very light)
                    white_ratio = np.mean(masked_pixels > 200)
                    
                    # 2. Mean Brightness (Black bars should be low, e.g. < 80-100)
                    mean_val = np.mean(masked_pixels)
                    
                    # Criteria: Almost no white pixels (<5%) AND mostly dark (<100 avg)
                    if white_ratio < 0.05 and mean_val < 100:
                        valid_indices.append(m_idx)
            else:
                # For mosaic/blur, accept all candidates
                valid_indices = [0, 1, 2]

            if not valid_indices:
                # If no valid mask found by SAM, we keep the original block (since acc initialized with mask_data)
                print(f"  [Filter] Block {i}: No valid SAM mask found. Keeping original detection.")
                continue

            # Select best mask from VALID candidates based on strategy
            valid_masks_np = masks_np[valid_indices]
            valid_scores = scores_np[valid_indices]
            valid_areas = [np.sum(m > 0) for m in valid_masks_np]
            
            best_relative_idx = 0 # Index within the valid_indices list

            if self.args.sam_strategy == "min":
                best_relative_idx = np.argmin(valid_areas)
            elif self.args.sam_strategy == "middle":
                 sorted_indices = np.argsort(valid_areas)
                 best_relative_idx = sorted_indices[len(valid_areas) // 2]
            elif self.args.sam_strategy == "max" or self.args.sam_strategy == "area":
                best_relative_idx = np.argmax(valid_areas)
            elif self.args.sam_strategy == "union":
                # Special case: Union doesn't pick an index, it merges
                combined = np.any(valid_masks_np > 0, axis=0)
                island_refined = combined.astype(np.uint8) * 255
            else: # "score"
                best_relative_idx = np.argmax(valid_scores)

            if self.args.sam_strategy != "union":
                original_idx = valid_indices[best_relative_idx]
                island_refined = (masks_np[original_idx] > 0).astype(np.uint8) * 255
            
            if MASK_EXPANSION_PIXELS > 0:
                struct = np.ones((MASK_EXPANSION_PIXELS, MASK_EXPANSION_PIXELS))
                island_refined = binary_dilation(island_refined > 0, structure=struct).astype(np.uint8) * 255
            
            # MERGE: Union of (Current Accumulator) OR (New SAM Mask)
            refined_full_mask_acc = np.maximum(refined_full_mask_acc, island_refined)
            
            # --- DEBUG VISUALIZATION ---
            if debug_stem:
                if not os.path.exists(DEFAULT_DEBUG_FOLDER): os.makedirs(DEFAULT_DEBUG_FOLDER)
                
                # 1. Base Image with Alpha
                vis_img = image.copy().convert("RGBA")
                
                # 2. Green Mask Overlay (Semi-transparent)
                green_overlay = Image.new("RGBA", image.size, (0, 255, 0, 100))
                mask_pil = Image.fromarray(island_refined).convert("L")
                vis_img.paste(green_overlay, (0, 0), mask_pil)
                
                # 3. Draw Red Point
                draw = ImageDraw.Draw(vis_img)
                r = 10 # Radius of the point marker
                draw.ellipse((point[0]-r, point[1]-r, point[0]+r, point[1]+r), outline=(255, 0, 0, 255), width=3)
                draw.line((point[0]-r, point[1], point[0]+r, point[1]), fill=(255, 0, 0, 255), width=3)
                draw.line((point[0], point[1]-r, point[0], point[1]+r), fill=(255, 0, 0, 255), width=3)
                
                # 4. Save
                save_name = f"{debug_stem}_block{i}_sam_vis.png"
                vis_img.save(os.path.join(DEFAULT_DEBUG_FOLDER, save_name))
        
        return Image.fromarray(refined_full_mask_acc)


    def mask_refinement_process(self):
        self.load_sam2()
        if not os.path.exists(self.hai_mask_folder): return
        
        input_files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        input_files.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])
        
        for idx, orig_file in enumerate(input_files):
            orig_path = os.path.join(self.args.input, orig_file)
            original_img = smart_resize(Image.open(orig_path).convert("RGB"))
            
            # Apply preprocessing if specified to match what might be needed, 
            # or keep it raw if SAM prefers. Usually SAM prefers the original content.
            # But if we did heavy black levels for detection, maybe we should keep it for alignment.
            if self.args.black_level > 0:
                original_img = self.preprocess_for_detection(original_img)
            
            tile_files = [f for f in os.listdir(self.hai_mask_folder) if f.startswith(f"{idx}_T_") and f.endswith(".png") and "merged" not in f]
            
            print(f"正在优化切片掩码: {orig_file} ({len(tile_files)} tiles)")
            for m_file in tile_files:
                # Parse coordinates to extract partial image
                # Format: {idx}_T_{lx1}_{ly1}_{lx2}_{ly2}_P_{px1}_{py1}_{px2}_{py2}...
                match = re.search(r"T_(\d+)_(\d+)_(\d+)_(\d+)_P_(-?\d+)_(-?\d+)_(\d+)_(\d+)", m_file)
                if not match:
                    print(f"Warning: Could not parse coords from {m_file}")
                    continue
                
                px1, py1, px2, py2 = map(int, match.groups()[4:8])
                
                # Extract the clean partial image from source
                clean_tile = extract_padded_tile(original_img, (px1, py1, px2, py2))
                
                try:
                    marked_tile = Image.open(os.path.join(self.hai_mask_folder, m_file)).convert("RGB")
                    
                    # Ensure sizes match
                    if clean_tile.size != marked_tile.size:
                        clean_tile = clean_tile.resize(marked_tile.size)

                    tile_mask = extract_mask_from_green(marked_tile)
                    if not tile_mask.getbbox(): continue

                    # debug_stem = f"{idx}_tile_{os.path.splitext(m_file)[0]}"
                    refined_mask = self.refine_masks_with_sam2_points(clean_tile, tile_mask) #, debug_stem=debug_stem)
                    
                    # Overwrite the tile with new green mask
                    green_canvas = Image.new("RGB", clean_tile.size, (0, 255, 0))
                    refined_green_tile = Image.composite(green_canvas, clean_tile, refined_mask)
                    refined_green_tile.save(os.path.join(self.hai_mask_folder, m_file))

                except Exception as e:
                    print(f"Error processing tile {m_file}: {e}")
                    continue
            
            print(f"切片优化完成 ({orig_file})。")


# --- MAIN ORCHESTRATION ---

def main():
    global LOGICAL_TILE_W, LOGICAL_TILE_H
    parser = argparse.ArgumentParser(description="Adaptive Buffer Manga Decensor")
    parser.add_argument("--mode", type=str, choices=["segment", "inpaint", "all", "hai", "refine", "extra"], default="all")
    parser.add_argument("--mosaic", type=int, default=1)
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FOLDER)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--hai_path", type=str, default=DEFAULT_HAI_PATH)
    parser.add_argument("--temp_tiles", type=str, default=DEFAULT_TEMP_TILES_FOLDER)
    parser.add_argument("--black_level", type=int, default=0, help="0-255: Darkens gray pixels below this threshold to pure black to improve detection.")
    parser.add_argument("--min_size", type=int, default=1200, help="Upscale images smaller than this dimension")
    parser.add_argument("--sam_strategy", type=str, choices=["score", "area", "union", "min", "middle", "max"], default="score", help="Strategy to select SAM mask: 'score' (default), 'min' (smallest area), 'middle' (median), 'max' (largest), 'union' (merge all)")
    
    args = parser.parse_args()
    
    # Initialize Managers
    mask_manager = DecensorMaskManager(args)
    inpaint_manager = DecensorInpainter(args)
    
    # # Update global variable via arg
    global MIN_DIMENSION
    MIN_DIMENSION = args.min_size

    if args.mode == "segment": 
        mask_manager.segment_images_to_temp()
    elif args.mode == "hai": 
        mask_manager.run_hai_detection(args.mosaic)
    elif args.mode == "inpaint": 
        inpaint_manager.inpaint_process()
    elif args.mode == "refine": 
        mask_manager.mask_refinement_process()
    elif args.mode == "all":
        print(">>> 启动阶段 1: 全图粗略检测 (Coarse Detection) <<<")
        LOGICAL_TILE_W = 2400
        LOGICAL_TILE_H = 2400
        mask_manager.segment_images_to_temp()
        mask_manager.run_hai_detection(args.mosaic, clear_existing=True)
        print(">>> 启动阶段 2: 重点区域二次检测 (Focused Detection) <<<")
        mask_manager.segment_focused_from_coarse()
        mask_manager.run_hai_detection(args.mosaic, clear_existing=True)
        print(">>> 启动阶段 3: 切片掩码优化 (Refining Mask Tiles) <<<")
        mask_manager.mask_refinement_process()
        print(">>> 启动阶段 4: 合并检测结果 (Merging Masks) <<<")
        mask_manager.merge_detection_results()
        mask_manager.unload_sam2()
        cleanup_gpu()
        inpaint_manager.inpaint_process()
    
    elif args.mode == "extra":
        original_input_dir = args.input
        original_output_dir = args.output
        pass1_output_dir = os.path.join(os.path.dirname(os.path.abspath(original_output_dir)), "temp_extra_pass1")
        
        # PASS 1: Detailed Tiling
        print(">>> EXTRA MODE: PASS 1 (Detailed Tiling) <<<")
        LOGICAL_TILE_W = 1248
        LOGICAL_TILE_H = 1248
        args.output = pass1_output_dir
        
        # Re-init managers with new output path
        mask_manager = DecensorMaskManager(args)
        inpaint_manager = DecensorInpainter(args)
        
        mask_manager.segment_images_to_temp()
        mask_manager.run_hai_detection(args.mosaic)
        
        # Refine -> Merge -> Inpaint
        mask_manager.mask_refinement_process()
        mask_manager.merge_detection_results()
        mask_manager.unload_sam2()
        
        inpaint_manager.inpaint_process()
        inpaint_manager.unload_pipeline()
        
        # PASS 2: Global Tiling
        print(">>> EXTRA MODE: PASS 2 (Global Tiling) <<<")
        LOGICAL_TILE_W = 2400
        LOGICAL_TILE_H = 2400
        args.input = pass1_output_dir
        args.output = original_output_dir
        
        # Re-init managers for pass 2 (input changed)
        mask_manager = DecensorMaskManager(args)
        inpaint_manager = DecensorInpainter(args)
        
        mask_manager.segment_images_to_temp()
        mask_manager.run_hai_detection(args.mosaic)
        
        # Refine -> Merge -> Inpaint
        mask_manager.mask_refinement_process()
        mask_manager.merge_detection_results()
        mask_manager.unload_sam2()
        
        inpaint_manager.inpaint_process()
        
        print(">>> EXTRA MODE: Cleaning up intermediate files <<<")
        if os.path.exists(pass1_output_dir):
            try: shutil.rmtree(pass1_output_dir)
            except Exception as e: print(f"Warning: Failed to clean up {pass1_output_dir}: {e}")
                
        args.input = original_input_dir

if __name__ == "__main__":
    main()