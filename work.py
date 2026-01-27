import os
import time
import torch
import numpy as np
import configparser
import shutil
import argparse
from PIL import Image, ImageFilter
from pywinauto import Application, Desktop
from diffusers import DiffusionPipeline


# --- CONFIGURATION DEFAULTS ---
# These can be overridden via command line arguments
DEFAULT_HAI_PATH = r"E:\MangaTranslator\HAI\HAI\main.exe"
DEFAULT_INPUT_FOLDER = r"E:\MangaTranslator\HAI\HAI\input"
DEFAULT_OUTPUT_FOLDER = r"E:\MangaTranslator\HAI\HAI\output"
DEFAULT_DEBUG_FOLDER = r"E:\MangaTranslator\HAI\HAI\debug_masks"
DEFAULT_TEMP_TILES_FOLDER = r"E:\MangaTranslator\HAI\HAI\temp_tiles"

# Performance & Quality Settings
MAX_DIMENSION = 2048
TILE_SIZE = 2048
TILE_OVERLAP = 128

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
        self.pipeline = None # Lazy loading

    def load_pipeline(self):
        """Only load the heavy model if we are actually inpainting."""
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

    def get_tiles(self, img_size):
        w, h = img_size
        stride = TILE_SIZE - TILE_OVERLAP
        tiles = []
        y = 0
        while y < h:
            x = 0
            y_end = min(y + TILE_SIZE, h)
            y_start = y_end - TILE_SIZE if y_end == h else y
            y_start = max(0, y_start)
            while x < w:
                x_end = min(x + TILE_SIZE, w)
                x_start = x_end - TILE_SIZE if x_end == w else x
                x_start = max(0, x_start)
                tiles.append((x_start, y_start, x_end, y_end))
                if x_end >= w: break
                x += stride
            if y_end >= h: break
            y += stride
        return list(dict.fromkeys(tiles))

    def segment_images_to_temp(self):
        """Segment-only mode: Prepares tiles for HAI."""
        print(f"模式: Segmentation - 正在准备临时切片: {self.args.temp_tiles}")
        if os.path.exists(self.args.temp_tiles):
            shutil.rmtree(self.args.temp_tiles)
        os.makedirs(self.args.temp_tiles)

        files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        for f in files:
            img_path = os.path.join(self.args.input, f)
            img = Image.open(img_path)
            img = self.smart_resize(img)
            w, h = img.size
            file_stem = os.path.splitext(f)[0]
            tiles = self.get_tiles((w, h))
            print(f"文件 {f} -> {len(tiles)} 个切片...")
            for i, (x1, y1, x2, y2) in enumerate(tiles):
                tile = img.crop((x1, y1, x2, y2))
                tile_name = f"{file_stem}_tile_{x1}_{y1}_{x2}_{y2}.png"
                tile.save(os.path.join(self.args.temp_tiles, tile_name))

    def run_hai_detection(self, mosaic_index):
        """Automate the HAI process using robust coordinate clicking and success detection."""
        print(f"正在启动 HAI 探测自动化 (码类型索引: {mosaic_index})...")
        
        if os.path.exists(self.hai_mask_folder):
            shutil.rmtree(self.hai_mask_folder)
        os.makedirs(self.hai_mask_folder)

        try:
            self.update_hai_config(self.args.temp_tiles)
            app = Application(backend="uia").start(self.args.hai_path, wait_for_idle=False)
            
            print("等待 hentAI 窗口加载...")
            time.sleep(30)
            
            # Step 1: Handle Selection Window
            desktop = Desktop(backend="uia")
            main_window = desktop.window(title_re=".*hentAI.*")
            main_window.wait('exists', timeout=20)
            main_window.set_focus()
            
            all_buttons = main_window.descendants(control_type="Button")
            
            # Click the requested mosaic type (usually index 1 for Bar/Mosaic)
            if len(all_buttons) > mosaic_index:
                print(f"正在点击选项按钮索引: {mosaic_index}")
                all_buttons[mosaic_index].click_input()
                time.sleep(3)
            else:
                print(f"警告: 按钮数量不足 ({len(all_buttons)})，无法点击索引 {mosaic_index}")

            # Step 2: Handle Detection/Go Window
            print("正在切换/寻找执行按钮...")
            try:
                detection_window = desktop.window(title_re=".*Detection.*")
                detection_window.wait('exists', timeout=20)
                detection_window.set_focus()
                
                all_btns_updated = detection_window.descendants(control_type="Button")
                functional_buttons = [b for b in all_btns_updated if b.window_text() not in ["最小化", "最大化", "关闭", "Minimize", "Maximize", "Close"]]
                
                if functional_buttons:
                    # Index 1 is often the 'Go!' or 'Run' button based on previous logs
                    go_btn = functional_buttons[1]
                    print(f"定位到执行按钮: '{go_btn.window_text()}'，点击中...")
                    go_btn.click_input()
                else:
                    print("未找到功能性按钮，尝试发送回车...")
                    detection_window.type_keys("{ENTER}")
                
                # Step 3: Wait for Success!
                print("等待 'Success!' 弹窗出现 (检测中)...")
                success_detected = False
                for _ in range(120): # Wait up to 2 minutes
                    try:
                        # Some versions use a separate popup, some use a title change
                        success_popup = desktop.window(title_re=".*Success.*")
                        if success_popup.exists():
                            print("检测到 Success! 弹窗。")
                            success_popup.set_focus()
                            # Close the popup (usually ENTER or clicking the OK button)
                            success_popup.close()
                            success_detected = True
                            break
                    except:
                        pass
                    time.sleep(1)
                
                if not success_detected:
                    print("超时：未检测到 Success! 弹窗，将尝试强制关闭。")

                # Step 4: Cleanup
                print("正在关闭 Detection 窗口...")
                if detection_window.exists():
                    detection_window.close()
                
            except Exception as e:
                print(f"执行阶段异常: {e}")
            time.sleep(2)
            app.kill() 
            print("HAI 自动化探测流程结束。")

        except Exception as e:
            print(f"GUI 自动化发生异常: {e}")

    def update_hai_config(self, input_dir):
        config_path = os.path.join(self.hai_dir, "hconfig.ini")
        if os.path.exists(config_path):
            config = configparser.ConfigParser()
            config.read(config_path)
            if 'Paths' not in config: config['Paths'] = {}
            config['Paths']['input'] = input_dir
            with open(config_path, 'w') as f:
                config.write(f)

    def extract_mask_from_green(self, image_with_green):
        img = image_with_green.convert("RGB")
        data = np.array(img)
        mask_arr = ((data[:,:,1] > 185) & (data[:,:,0] < 110) & (data[:,:,2] < 110)).astype(np.uint8) * 255
        return Image.fromarray(mask_arr)

    def save_debug_images(self, stem, x1, y1, tile_img, mask_img, inpainted_tile=None):
        if not os.path.exists(self.args.debug): os.makedirs(self.args.debug)
        base_name = f"{stem}_{x1}_{y1}"
        tile_img.save(os.path.join(self.args.debug, f"{base_name}_0_raw.png"))
        mask_img.save(os.path.join(self.args.debug, f"{base_name}_1_mask.png"))
        overlay = tile_img.convert("RGBA")
        red_mask = Image.new("RGBA", tile_img.size, (255, 0, 0, 128))
        mask_rgba = mask_img.convert("L")
        overlay_final = Image.composite(red_mask, overlay, mask_rgba)
        overlay_final.convert("RGB").save(os.path.join(self.args.debug, f"{base_name}_2_overlay.png"))
        if inpainted_tile:
            inpainted_tile.save(os.path.join(self.args.debug, f"{base_name}_3_inpainted.png"))

    def inpaint_process(self):
        """Inpaint-only mode: Loads model and processes HAI outputs."""
        self.load_pipeline()
        if not os.path.exists(self.args.output): os.makedirs(self.args.output)
        
        input_files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        for orig_file in input_files:
            file_stem = os.path.splitext(orig_file)[0]
            orig_path = os.path.join(self.args.input, orig_file)
            original_img = self.smart_resize(Image.open(orig_path).convert("RGB"))
            width, height = original_img.size
            output_canvas = np.array(original_img).astype(np.float32)
            processed_any_tile = False

            if not os.path.exists(self.hai_mask_folder):
                print(f"找不到 HAI 掩码文件夹: {self.hai_mask_folder}")
                return

            for m_file in os.listdir(self.hai_mask_folder):
                if m_file.startswith(file_stem + "_tile_") and m_file.endswith(".png"):
                    parts = m_file.replace(".png", "").split("_")
                    try: x1, y1, x2, y2 = map(int, parts[-4:])
                    except: continue
                    
                    marked_tile_path = os.path.join(self.hai_mask_folder, m_file)
                    tile_mask = self.extract_mask_from_green(Image.open(marked_tile_path))
                    if not tile_mask.getbbox(): continue
                    
                    processed_any_tile = True
                    tile_img = original_img.crop((x1, y1, x2, y2))
                    tw, th = tile_img.size
                    st_tw, st_th = tw - (tw % 64), th - (th % 64)
                    tile_img_st = tile_img.resize((st_tw, st_th), Image.LANCZOS)
                    tile_mask_st = tile_mask.convert("L").filter(ImageFilter.MaxFilter(7)).resize((st_tw, st_th), Image.NEAREST)

                    print(f"正在重绘 {file_stem} 区域: {x1, y1, x2, y2}")
                    inpainted_tile_st = self.pipeline(
                        prompt="reconstruct, masterpiece, uncensored, genitals, detailed anatomy",
                        negative_prompt="green bars, black bars, mosaic, censored, blurry",
                        image=tile_img_st, mask_image=tile_mask_st,
                        num_inference_steps=30, strength=1.0
                    ).images[0]

                    inpainted_tile = inpainted_tile_st.resize((tw, th), Image.LANCZOS)
                    self.save_debug_images(file_stem, x1, y1, tile_img, tile_mask, inpainted_tile)
                    
                    tile_arr = np.array(inpainted_tile).astype(np.float32)
                    tile_weight = np.ones((th, tw), dtype=np.float32)
                    feather = TILE_OVERLAP // 2
                    for f in range(feather):
                        val = f / feather
                        if x1 > 0: tile_weight[:, f] *= val
                        if y1 > 0: tile_weight[f, :] *= val
                        if x2 < width: tile_weight[:, -f-1] *= val
                        if y2 < height: tile_weight[-f-1, :] *= val

                    for c in range(3):
                        output_canvas[y1:y2, x1:x2, c] = (1 - tile_weight) * output_canvas[y1:y2, x1:x2, c] + tile_weight * tile_arr[:, :, c]

            if processed_any_tile:
                final_img = Image.fromarray(np.clip(output_canvas, 0, 255).astype(np.uint8))
                save_path = os.path.join(self.args.output, f"decensored_{file_stem}.png")
                final_img.save(save_path)
                print(f"成功保存: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Manga Decensor Management Script")
    parser.add_argument("--mode", type=str, choices=["segment", "inpaint", "all", "hai"], default="all",
                        help="运行模式: segment(仅切图), inpaint(仅重绘), hai(仅自动化HAI), all(完整运行)")
    parser.add_argument("--mosaic", type=int, default=1, help="码类型")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_FOLDER, help="输入图片路径")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_FOLDER, help="输出结果路径")
    parser.add_argument("--hai_path", type=str, default=DEFAULT_HAI_PATH, help="HAI 路径")
    parser.add_argument("--temp_tiles", type=str, default=DEFAULT_TEMP_TILES_FOLDER, help="临时切片路径")
    parser.add_argument("--debug", type=str, default=DEFAULT_DEBUG_FOLDER, help="诊断图路径")
    
    args = parser.parse_args()
    manager = DecensorAutomation(args)

    if args.mode == "segment":
        manager.segment_images_to_temp()
    elif args.mode == "hai":
        manager.run_hai_detection(args.mosaic)
    elif args.mode == "inpaint":
        manager.inpaint_process()
    elif args.mode == "all":
        manager.segment_images_to_temp()
        manager.run_hai_detection(args.mosaic)
        manager.inpaint_process()

if __name__ == "__main__":
    main()