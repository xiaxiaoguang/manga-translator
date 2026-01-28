import os
import time
import torch
import numpy as np
import configparser
import shutil
import argparse

from PIL import Image, ImageFilter, ImageChops, ImageOps
from pywinauto import Application, Desktop
from scipy.ndimage import label, binary_dilation, center_of_mass # Required for mask island detection

from diffusers import DiffusionPipeline
from transformers import Sam2Processor, Sam2Model


# --- CONFIGURATION DEFAULTS ---
DEFAULT_HAI_PATH = r"E:\MangaTranslator\HAI\HAI\main.exe"
DEFAULT_INPUT_FOLDER = r"E:\MangaTranslator\HAI\HAI\input"
DEFAULT_OUTPUT_FOLDER = r"E:\MangaTranslator\HAI\HAI\output"
DEFAULT_DEBUG_FOLDER = r"E:\MangaTranslator\HAI\HAI\debug_masks"
DEFAULT_TEMP_TILES_FOLDER = r"E:\MangaTranslator\HAI\HAI\temp_tiles"
SAM2_MODEL_PATH = r"E:\MangaTranslator\MangaTranslator\models\sam\models--facebook--sam2.1-hiera-large\snapshots\665f8e2ad61cf5f53d65644ff27c8ee525124610"

# Performance & Quality Settings
MAX_DIMENSION = 2048 
LOGICAL_TILE_W = 2048
LOGICAL_TILE_H = 2048

# CNN Safety Buffer: We expand the tile by this much on ALL sides with white pixels.
CONTEXT_PADDING = 32
INPAINT_SIZE = 1024 # Target size for the adaptive inpaint window
MASK_EXPANSION_PIXELS=6
# Proxy configuration
proxy = "http://127.0.0.1:7897" 
os.environ["HTTP_PROXY"] = proxy
os.environ["HTTPS_PROXY"] = proxy
os.environ["HF_HUB_PROXY"] = proxy

class DecensorAutomation:
    def __init__(self, args):
        self.args = args
        self.hai_dir = os.path.dirname(args.hai_path)
        self.hai_mask_folder = os.path.join(self.hai_dir, "decensor_input")
        self.pipeline = None 

    def load_sam2(self):
        # Initialize SAM2 from local path using transformers
        print(f"正在从本地路径加载 SAM2.1 模型: {SAM2_MODEL_PATH}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load model and processor
        self.sam_processor = Sam2Processor.from_pretrained(SAM2_MODEL_PATH)
        self.sam_model = Sam2Model.from_pretrained(SAM2_MODEL_PATH).to(self.device)
        print(f"SAM2.1 加载完成，运行设备: {self.device}")
        
    def load_pipeline(self):
        if self.pipeline is None:
            print("正在初始化 Inpainting 流水线 (Loading model to CUDA)...")
            self.pipeline = DiffusionPipeline.from_pretrained(
                'ShinoharaHare/Waifu-Inpaint-XL',
                torch_dtype=torch.float16,
                trust_remote_code=True
            ).to('cuda')

    def smart_resize(self, img):
        w, h = img.size
        if max(w, h) > MAX_DIMENSION:
            scale = MAX_DIMENSION / float(max(w, h))
            new_w = int(w * scale)
            new_h = int(h * scale)
            print(f"图片过大 ({w}x{h})，降采样至: {new_w}x{new_h}")
            return img.resize((new_w, new_h), Image.LANCZOS)
        return img

    def get_tiles_with_padding(self, img_size):
        w, h = img_size
        tiles = []
        nx = int(np.ceil(w / LOGICAL_TILE_W))
        ny = int(np.ceil(h / LOGICAL_TILE_H))
        actual_tile_w = int(np.ceil(w / nx))
        actual_tile_h = int(np.ceil(h / ny))

        for iy in range(ny):
            for ix in range(nx):
                x1 = ix * actual_tile_w
                y1 = iy * actual_tile_h
                x2 = min(w, (ix + 1) * actual_tile_w)
                y2 = min(h, (iy + 1) * actual_tile_h)
                
                # These are the original padded bounds
                px1 = x1 - CONTEXT_PADDING
                py1 = y1 - CONTEXT_PADDING
                px2 = x2 + CONTEXT_PADDING
                py2 = y2 + CONTEXT_PADDING
                
                tiles.append({
                    'logical': (x1, y1, x2, y2),
                    'padded': (px1, py1, px2, py2)
                })
        return tiles
        
    def extract_padded_tile(self, img, padded_box):
        """
        Extracts a tile from img with guaranteed white padding.
        Uses the user-specified offset for placement.
        """
        px1, py1, px2, py2 = padded_box
        target_w = px2 - px1
        target_h = py2 - py1
        
        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        img_w, img_h = img.size
        
        # Calculate the actual area to crop from the source image
        # This is the 'logical' tile part
        l_w = target_w - (CONTEXT_PADDING)
        l_h = target_h - (CONTEXT_PADDING)
        
        # We assume the input coordinates were based on logical + context.
        # However, to match the user's paste logic (fixed offset), 
        # we extract based on the logical bounds stored in the box.
        overlap_x1 = max(0, px1 + CONTEXT_PADDING)
        overlap_y1 = max(0, py1 + CONTEXT_PADDING)
        overlap_x2 = min(img_w, px2 - CONTEXT_PADDING)
        overlap_y2 = min(img_h, py2 - CONTEXT_PADDING)
        
        if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
            part = img.crop((overlap_x1, overlap_y1, overlap_x2, overlap_y2))
            # Fixed Offset as requested
            canvas.paste(part, (CONTEXT_PADDING // 2, CONTEXT_PADDING // 2))
            
        return canvas
        
    def segment_images_to_temp(self):
        print(f"模式: Segmentation - 正在准备带固定偏置缓冲的 HAI 检测块...")
        if os.path.exists(self.args.temp_tiles): shutil.rmtree(self.args.temp_tiles)
        os.makedirs(self.args.temp_tiles)
        
        files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        for i,f in enumerate(files):
            img = self.smart_resize(Image.open(os.path.join(self.args.input, f)).convert("RGB"))
            w, h = img.size
            tile_configs = self.get_tiles_with_padding((w, h))
            
            for config in tile_configs:
                p_box, l_box = config['padded'], config['logical']
                tile = self.extract_padded_tile(img, p_box)
                tile_name = f"{i}_T_{l_box[0]}_{l_box[1]}_{l_box[2]}_{l_box[3]}_P_{p_box[0]}_{p_box[1]}_{p_box[2]}_{p_box[3]}.png"
                tile.save(os.path.join(self.args.temp_tiles, tile_name))

    def run_hai_detection(self, mosaic_index):
        print(f"模式: HAI - 启动自动化检测 (Index: {mosaic_index})...")
        if os.path.exists(self.hai_mask_folder): shutil.rmtree(self.hai_mask_folder)
        os.makedirs(self.hai_mask_folder)
        try:
            self.update_hai_config(self.args.temp_tiles)
            app = Application(backend="uia").start(self.args.hai_path, wait_for_idle=False)
            time.sleep(15)
            desktop = Desktop(backend="uia")
            main_window = desktop.window(title_re=".*hentAI.*")
            main_window.wait('exists', timeout=20)
            main_window.set_focus()
            all_buttons = main_window.descendants(control_type="Button")
            if len(all_buttons) > mosaic_index:
                all_buttons[mosaic_index].click_input()
                time.sleep(3)
            detection_window = desktop.window(title_re=".*Detection.*")
            detection_window.wait('exists', timeout=20)
            detection_window.set_focus()
            functional_buttons = [b for b in detection_window.descendants(control_type="Button") if b.window_text() not in ["最小化", "最大化", "关闭"]]
            if functional_buttons: functional_buttons[1].click_input()
            
            for _ in range(3600):
                try:
                    success_popup = desktop.window(title_re=".*Success.*")
                    if success_popup.exists():
                        success_popup.close()
                        break
                except: pass
                time.sleep(1)
                
            if detection_window.exists(): detection_window.close()
            app.kill() 
        except Exception as e: print(f"GUI 自动化异常: {e}")

    def update_hai_config(self, input_dir):
        config_path = os.path.join(self.hai_dir, "hconfig.ini")
        if os.path.exists(config_path):
            config = configparser.ConfigParser()
            config.read(config_path)
            if 'Paths' not in config: config['Paths'] = {}
            config['Paths']['input'] = input_dir
            with open(config_path, 'w') as f: config.write(f)

    def extract_mask_from_green(self, image_with_green):
        img = image_with_green.convert("RGB")
        data = np.array(img)
        mask_arr = ((data[:,:,1] > 185) & (data[:,:,0] < 120) & (data[:,:,2] < 120)).astype(np.uint8) * 255
        return Image.fromarray(mask_arr)

    def save_debug_images(self, stem, feat_id, tile_img, mask_img, inpainted_tile=None):
        if not os.path.exists(DEFAULT_DEBUG_FOLDER): os.makedirs(DEFAULT_DEBUG_FOLDER)
        base_name = f"{stem}_F{feat_id}"
        tile_img.save(os.path.join(DEFAULT_DEBUG_FOLDER, f"{base_name}_0_raw.png"))
        mask_img.save(os.path.join(DEFAULT_DEBUG_FOLDER, f"{base_name}_1_mask.png"))
        if inpainted_tile:
            inpainted_tile.save(os.path.join(DEFAULT_DEBUG_FOLDER, f"{base_name}_3_inpainted.png"))
    
    def refine_masks_with_sam2_points(self, image, initial_mask):
        """
        Refines the mask block-by-block. 
        Instead of a global query, we process each 'island' detected in the rough mask.
        """
        mask_data = np.array(initial_mask)
        labeled_array, num_features = label(mask_data > 128)
        if num_features == 0: 
            return initial_mask
        refined_full_mask_acc = np.zeros_like(mask_data)
        
        print(f" 检测到 {num_features} 个独立的遮罩区块。正在逐一处理...")

        for i in range(1, num_features + 1):
            # Extract coordinates for this specific island
            coords = np.argwhere(labeled_array == i) # [y, x]
            
            # Sampling logic for points
            points = []
            # 1. Center of the island
            center_y, center_x = np.mean(coords, axis=0)
            points.append([int(center_x), int(center_y)])
            # Prepare inputs for SAM2
            input_points = [[points]] 
            input_labels = [[[1] * len(points)]] 

            inputs = self.sam_processor(
                images=image, 
                input_points=input_points, 
                input_labels=input_labels, 
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.sam_model(**inputs)
            
            # Post-process results for this specific island
            masks = self.sam_processor.post_process_masks(
                outputs.pred_masks.cpu(), 
                inputs["original_sizes"]
            )[0]
            
            best_mask_idx = 0 
            island_refined = (masks[0, best_mask_idx].numpy() > 0).astype(np.uint8) * 255
            
            if MASK_EXPANSION_PIXELS > 0:
                struct = np.ones((MASK_EXPANSION_PIXELS, MASK_EXPANSION_PIXELS))
                island_refined = binary_dilation(island_refined > 0, structure=struct).astype(np.uint8) * 255
            refined_full_mask_acc = np.maximum(refined_full_mask_acc, island_refined)
            
        return Image.fromarray(refined_full_mask_acc)

    def mask_refinement_process(self):
        """Processes HAI output tiles, reconstructs the mask, refines per island, and saves green-masked images."""
        if not os.path.exists(self.args.output): os.makedirs(self.args.output)
        input_files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        self.load_sam2()
        for idx, orig_file in enumerate(input_files):
            orig_path = os.path.join(self.args.input, orig_file)
            original_img = self.smart_resize(Image.open(orig_path).convert("RGB"))
            w, h = original_img.size
            full_rough_mask = Image.new("L", (w, h), 0)
            
            # Reconstruct full rough mask from HAI tiles
            if not os.path.exists(self.hai_mask_folder): continue
            print(f"正在从切片中还原全图掩码: {orig_file}")
            filename = None
            for m_file in os.listdir(self.hai_mask_folder):
                if m_file.startswith(str(idx) + "_T_") and m_file.endswith(".png"):
                    parts = m_file.replace(".png", "").split("_")
                    lx1, ly1, lx2, ly2 = map(int, parts[2:6])
                    
                    marked_tile = Image.open(os.path.join(self.hai_mask_folder, m_file))
                    tile_mask = self.extract_mask_from_green(marked_tile)
                    
                    offset = CONTEXT_PADDING // 2
                    logical_mask_part = tile_mask.crop((offset, offset, offset + (lx2 - lx1), offset + (ly2 - ly1)))
                    full_rough_mask.paste(logical_mask_part, (lx1, ly1))
                    filename = m_file

            if not full_rough_mask.getbbox():
                print(f"跳过 {orig_file}: 未发现任何掩码区域。")
                continue

            print(f"正在使用 SAM2 掩码: {orig_file}")
            final_refined_mask = self.refine_masks_with_sam2_points(original_img, full_rough_mask)
            
            green_canvas = Image.new("RGB", (w, h), (0, 255, 0))
            green_masked_img = Image.composite(green_canvas, original_img, final_refined_mask)
            output_path = os.path.join(self.hai_mask_folder, filename)
            green_masked_img.save(output_path)
            print(f"优化后的绿色遮罩图已保存: {output_path}")
            
    def inpaint_process(self):
        self.load_pipeline()
        if not os.path.exists(self.args.output): os.makedirs(self.args.output)
        adaptive_strength = 0.7 if self.args.mosaic == 1 else 1.0

        input_files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        for idx,orig_file in enumerate(input_files):
            # file_stem = os.path.splitext(orig_file)[0]
            orig_path = os.path.join(self.args.input, orig_file)
            original_img = self.smart_resize(Image.open(orig_path).convert("RGB"))
            w, h = original_img.size
            
            # --- PHASE 1: RECONSTRUCT THE FULL PAGE MASK ---
            full_mask = Image.new("L", (w, h), 0)
            if not os.path.exists(self.hai_mask_folder): continue
            
            print(f"正在从切片中还原全图掩码: {orig_file}")
            for m_file in os.listdir(self.hai_mask_folder):
                if m_file.startswith(str(idx) + "_T_") and m_file.endswith(".png"):
                    parts = m_file.replace(".png", "").split("_")
                    try:
                        # Recovery indices
                        l_x1, l_y1, l_x2, l_y2 = map(int, parts[2:6])
                        marked_tile = Image.open(os.path.join(self.hai_mask_folder, m_file))
                        tile_mask = self.extract_mask_from_green(marked_tile)
                        full_mask.paste(tile_mask, (l_x1, l_y1))
                    except Exception as e:
                        print(f"还原切片掩码失败: {e}")
                        continue
            
            output_canvas = np.array(original_img).astype(np.float32)
            if not full_mask.getbbox():
                print(f"跳过 {orig_file}: 未检测到掩码。")
                final_img = Image.fromarray(np.clip(original_img, 0, 255).astype(np.uint8))
                final_img = ImageOps.grayscale(final_img).convert("RGB")
                save_path = os.path.join(self.args.output, f"_{idx}.png")
                original_img.save(save_path)
                continue
            # --- PHASE 2: ADAPTIVE SEGMENTATION & PROGRESSIVE INPAINTING ---
            mask_data = np.array(full_mask)
            labeled_array, num_features = label(mask_data > 128)
            print(f"在 {orig_file} 中检测到 {num_features} 个独立的遮罩区域。")
            coverage_mask = np.zeros((h, w), dtype=bool)

            for feature_id in range(1, num_features + 1):
                this_island_mask = (labeled_array == feature_id)
                if np.all(coverage_mask[this_island_mask]):
                    continue

                y_center, x_center = center_of_mass(mask_data, labeled_array, feature_id)
                x_center, y_center = int(x_center), int(y_center)
                
                half_size = INPAINT_SIZE // 2
                ax1 = max(0, x_center - half_size)
                ay1 = max(0, y_center - half_size)
                ax2 = min(w, ax1 + INPAINT_SIZE)
                ay2 = min(h, ay1 + INPAINT_SIZE)
                
                if ax2 == w: ax1 = max(0, w - INPAINT_SIZE)
                if ay2 == h: ay1 = max(0, h - INPAINT_SIZE)

                working_tile_arr = output_canvas[ay1:ay2, ax1:ax2]
                adaptive_tile = Image.fromarray(np.clip(working_tile_arr, 0, 255).astype(np.uint8))
                adaptive_mask_img = full_mask.crop((ax1, ay1, ax2, ay2))

                tw, th = adaptive_tile.size
                st_tw, st_th = tw - (tw % 64), th - (th % 64)
                tile_st = adaptive_tile.resize((st_tw, st_th), Image.LANCZOS)
                mask_st = adaptive_mask_img.resize((st_tw, st_th), Image.NEAREST)
                mask_st = mask_st.filter(ImageFilter.MaxFilter(11)).filter(ImageFilter.GaussianBlur(5))

                print(f"  正在重绘区域 {feature_id} (中心: {x_center}, {y_center})...")
                inpainted_st = self.pipeline(
                    prompt="reconstruct, masterpiece, best quality, monochrome, lineart, black and white, doujinshi, uncensored genitals, detailed anatomy",
                    negative_prompt="color, mosaic, green bars, black bars, blurry",
                    image=tile_st, mask_image=mask_st,
                    num_inference_steps=35, guidance_scale=8.0, strength=adaptive_strength 
                ).images[0]

                inpainted_final = inpainted_st.resize((tw, th), Image.LANCZOS)
                inpainted_final = ImageOps.grayscale(inpainted_final).convert("RGB")
                
                blend_mask = adaptive_mask_img.filter(ImageFilter.MaxFilter(13)).filter(ImageFilter.GaussianBlur(6))
                result_tile = Image.composite(inpainted_final, adaptive_tile, blend_mask)
                
                output_canvas[ay1:ay2, ax1:ax2] = np.array(result_tile).astype(np.float32)
                coverage_mask[ay1:ay2, ax1:ax2] = True
                
                # self.save_debug_images(idx, feature_id, adaptive_tile, adaptive_mask_img, inpainted_final)

            final_img = Image.fromarray(np.clip(output_canvas, 0, 255).astype(np.uint8))
            final_img = ImageOps.grayscale(final_img).convert("RGB")
            save_path = os.path.join(self.args.output, f"_{idx}_decensored.png")
            final_img.save(save_path)
            print(f"成功保存处理结果: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Buffer Manga Decensor")
    parser.add_argument("--mode", type=str, choices=["segment", "inpaint", "all", "hai", "refine"], default="all")
    parser.add_argument("--mosaic", type=int, default=1) # 1 is mosaic, 2 is bar 
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FOLDER)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--hai_path", type=str, default=DEFAULT_HAI_PATH)
    parser.add_argument("--temp_tiles", type=str, default=DEFAULT_TEMP_TILES_FOLDER)
    
    args = parser.parse_args()
    manager = DecensorAutomation(args)

    if args.mode == "segment": manager.segment_images_to_temp()
    elif args.mode == "hai": manager.run_hai_detection(args.mosaic)
    elif args.mode == "inpaint": manager.inpaint_process()
    elif args.mode == "refine": manager.mask_refinement_process()
    elif args.mode == "all":
        manager.segment_images_to_temp()
        manager.run_hai_detection(args.mosaic)
        manager.mask_refinement_process()
        # clean gpu cache, we cannot load sam2 simultanuously
        manager.inpaint_process()

if __name__ == "__main__":
    main()