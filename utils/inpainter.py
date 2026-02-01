import os
import time
import torch
import gc
import numpy as np
import configparser
import shutil
import argparse
import re

from PIL import Image, ImageFilter, ImageChops, ImageOps, ImageEnhance
from scipy.ndimage import label, binary_dilation, center_of_mass, find_objects

# --- IMPORTS FOR LOADING ---
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, AutoPipelineForInpainting, StableDiffusionXLPipeline
from transformers import Sam2Processor, Sam2Model

# Placeholder for local tools if they exist, otherwise ignore
try:
    from .tools import *
except ImportError:
    pass
class DecensorInpainter:
    """
    Original Inpainter Class using Waifu-Inpaint-XL.
    Now upgraded with Morphological Clustering (binary_dilation).
    """
    def __init__(self, args):
        self.args = args
        self.hai_dir = os.path.dirname(args.hai_path) if hasattr(args, 'hai_path') else "hai_output"
        self.hai_mask_folder = os.path.join(self.hai_dir, "decensor_input")
        self.pipeline = None
        # Default settings for Waifu-Inpaint-XL (SDXL)
        self.target_resolution = INPAINT_SIZE 

    def preprocess_for_detection(self, img):
        if self.args.black_level > 0:
            lookup = []
            for i in range(256):
                if i < self.args.black_level:
                    lookup.append(0)
                else:
                    val = int(255 * (i - self.args.black_level) / (255 - self.args.black_level))
                    lookup.append(min(255, val))
            
            if img.mode == 'RGB':
                r, g, b = img.split()
                r = r.point(lookup)
                g = g.point(lookup)
                b = b.point(lookup)
                img = Image.merge('RGB', (r, g, b))
            else:
                img = img.point(lookup)
            img = ImageEnhance.Contrast(img).enhance(1.1)
        return img

    def load_pipeline(self):
        if self.pipeline is None:
            print("Initializing Inpainting Pipeline (Waifu-Inpaint-XL)...")
            self.pipeline = DiffusionPipeline.from_pretrained(
                'ShinoharaHare/Waifu-Inpaint-XL',
                torch_dtype=torch.float16,
                use_safetensors=True
            ).to('cuda')
    
    def unload_pipeline(self):
        self.pipeline = None
        cleanup_gpu()

    def get_prompts(self):
        # Prompts optimized for Waifu-Inpaint-XL
        return {
            "prompt": "reconstruct, masterpiece, best quality, monochrome, lineart, black and white, doujinshi, uncensored genitals, detailed anatomy",
            "negative_prompt": "color, mosaic, green bars, black bars, blurry, low quality, error"
        }

    def save_debug_images(self, stem, feat_id, tile_img, mask_img, inpainted_tile=None):
        if not os.path.exists(DEFAULT_DEBUG_FOLDER): os.makedirs(DEFAULT_DEBUG_FOLDER)
        base_name = f"{stem}_F{feat_id}"
        tile_img.save(os.path.join(DEFAULT_DEBUG_FOLDER, f"{base_name}_0_raw.png"))
        mask_img.save(os.path.join(DEFAULT_DEBUG_FOLDER, f"{base_name}_1_mask.png"))
        if inpainted_tile:
            inpainted_tile.save(os.path.join(DEFAULT_DEBUG_FOLDER, f"{base_name}_3_inpainted.png"))

    def inpaint_process(self):
        self.load_pipeline()
        if not os.path.exists(self.args.output): os.makedirs(self.args.output)
        
        adaptive_strength = 0.7 if self.args.mosaic == 1 else 1
        prompts = self.get_prompts()

        input_files = [f for f in os.listdir(self.args.input) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp'))]
        input_files.sort(key=lambda f: [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', f)])

        for idx, orig_file in enumerate(input_files):
            print(f"Processing: {orig_file}")
            orig_path = os.path.join(self.args.input, orig_file)
            
            original_img = smart_resize(Image.open(orig_path).convert("RGB"))
            w, h = original_img.size
            if self.args.black_level > 0:
                original_img = self.preprocess_for_detection(original_img)

            # --- MASK LOADING ---
            full_mask = Image.new("L", (w, h), 0)
            prefix = f"{idx}_" 
            
            if os.path.exists(self.hai_mask_folder):
                for m_file in os.listdir(self.hai_mask_folder):
                    if m_file.startswith(prefix) and "_T_" in m_file and m_file.endswith(".png"):
                        parts = m_file.replace(".png", "").split("_")
                        try:
                            l_x1, l_y1, l_x2, l_y2 = map(int, parts[2:6])
                            marked_tile = Image.open(os.path.join(self.hai_mask_folder, m_file))
                            tile_mask = extract_mask_from_green(marked_tile)
                            full_mask.paste(tile_mask, (l_x1, l_y1))
                        except Exception: continue
            
            if not full_mask.getbbox():
                original_img.save(os.path.join(self.args.output, f"{idx}.png"))
                continue

            output_canvas = np.array(original_img).astype(np.float32)
            mask_data = np.array(full_mask)

            # --- UPDATED CLUSTERING LOGIC ---
            # 1. Morphological Grouping (The "Dilation" Strategy)
            # This merges nearby bars/mosaics into single task islands.
            dilation_iter = 30 # Adjust: Higher = larger groups, lower = finer separation
            group_map = binary_dilation(mask_data > 128, iterations=dilation_iter)
            
            # Label the dilated groups
            labeled_groups, num_groups = label(group_map)
            coverage_mask = np.zeros((h, w), dtype=bool)

            # 1. Count pixels for every group ID efficiently
            # group_ids will contain [0, 1, 2...] and counts will be their pixel sizes
            group_ids, group_sizes = np.unique(labeled_groups, return_counts=True)
            
            # 2. Store (id, size) tuples, ignoring ID 0 (background)
            valid_groups = []
            for g_id, g_size in zip(group_ids, group_sizes):
                if g_id == 0: continue # Skip background
                valid_groups.append((g_id, g_size))
            
            # 3. Sort by size (largest first) and slice the top 5
            valid_groups.sort(key=lambda x: x[1], reverse=True)
            target_groups = valid_groups[:5]

            print(f"  > Found {num_groups} mask clusters. Processing top {len(target_groups)} largest.")
            for group_id, size in target_groups:
                # Mask for the current CLUSTER
                current_group_indices = (labeled_groups == group_id)
                # Intersection of Group Box and Actual Mask
                active_pixels_in_group = (mask_data > 128) & current_group_indices
                
                if not np.any(active_pixels_in_group): continue
                # We skip the check below because each group_id is unique by definition
                # if np.all(coverage_mask[active_pixels_in_group]): continue

                # Center of Mass & Bounding Box of the CLUSTER
                y_center, x_center = center_of_mass(current_group_indices)
                
                # Determine crop size based on target resolution
                current_inpaint_size = self.target_resolution 
                
                ax1, ay1 = max(0, int(x_center) - current_inpaint_size // 2), max(0, int(y_center) - current_inpaint_size // 2)
                ax2, ay2 = min(w, ax1 + current_inpaint_size), min(h, ay1 + current_inpaint_size)
                
                # Boundary Checks
                if ax2 - ax1 < current_inpaint_size:
                    if ax1 == 0: ax2 = min(w, current_inpaint_size)
                    if ax2 == w: ax1 = max(0, w - current_inpaint_size)
                if ay2 - ay1 < current_inpaint_size:
                    if ay1 == 0: ay2 = min(h, current_inpaint_size)
                    if ay2 == h: ay1 = max(0, h - current_inpaint_size)

                tmp_canvas = Image.fromarray((output_canvas).astype(np.uint8))
                adaptive_tile = tmp_canvas.crop((ax1, ay1, ax2, ay2))
                group_mask_pil = Image.fromarray((active_pixels_in_group * 255).astype(np.uint8), mode='L')
                adaptive_mask_img = group_mask_pil.crop((ax1, ay1, ax2, ay2))

                # Standardize inputs (SD requirement)
                st_tw, st_th = (adaptive_tile.width // 64) * 64, (adaptive_tile.height // 64) * 64
                if st_tw == 0 or st_th == 0: continue

                # Check if we need to upscale small tiles (crucial for SDXL models)
                # If tile is small but target is large (SDXL), we must upscale
                scale_factor = 1.0
                if self.target_resolution > 512:
                     if max(st_tw, st_th) < 768:
                         scale_factor = min(2.0, self.target_resolution / max(st_tw, st_th))
                
                final_tw, final_th = int(st_tw * scale_factor), int(st_th * scale_factor)
                final_tw, final_th = (final_tw // 8) * 8, (final_th // 8) * 8 # Ensure div by 8

                tile_st = adaptive_tile.resize((final_tw, final_th), Image.LANCZOS)
                mask_st = adaptive_mask_img.resize((final_tw, final_th), Image.NEAREST)
                
                # INPAINT
                inpainted_st = self.pipeline(
                    prompt=prompts["prompt"],
                    negative_prompt=prompts["negative_prompt"],
                    image=tile_st, mask_image=mask_st,
                    num_inference_steps=30, 
                    guidance_scale=7.5,
                    strength=adaptive_strength 
                ).images[0]
                
                # Resize back and Blend
                inpainted_final = inpainted_st.resize(adaptive_tile.size, Image.LANCZOS)
                inpainted_final = ImageOps.grayscale(inpainted_final).convert("RGB")
                
                # blend_mask = adaptive_mask_img.filter(ImageFilter.MaxFilter(1)).filter(ImageFilter.GaussianBlur(3))
                # result_tile = Image.composite(inpainted_final, adaptive_tile, blend_mask)
                
                output_canvas[ay1:ay2, ax1:ax2] = np.array(inpainted_final).astype(np.float32)
                # coverage_mask[ay1:ay2, ax1:ax2] = True # Mark bounding box as processed to avoid re-clustering

            final_img = ImageOps.grayscale(Image.fromarray(np.clip(output_canvas, 0, 255).astype(np.uint8))).convert("RGB")
            final_img.save(os.path.join(self.args.output, f"{idx}_decensored.png"))


# --- NEW IMPROVED INPAINTER ---
class DecensorInpainter2(DecensorInpainter):
    """
    Advanced Inpainter using 'NoobAI XL 1.1'.
    A modern (2025 era) model fine-tuned specifically on Danbooru tags.
    It excels at 'lineart' and 'monochrome' styles where Pony models might force 3D/color.
    """
    def __init__(self, args):
        super().__init__(args)
        # NoobAI is SDXL based, so it prefers 1024px
        self.target_resolution = 1024 

    def load_pipeline(self):
        if self.pipeline is None:
            print("Initializing Advanced Pipeline (NoobAI XL 1.1)...")
            # We use the Base SDXL pipeline -> Inpainting Adapter strategy
            # This is robust for models that are distributed as single checkpoints
            
            repo_id = "Laxhar/noobai-XL-1.1"
            ckpt_name = "noobai-XL-1.1.safetensors"
            
            # Local cache check
            local_models_dir = os.path.join(os.getcwd(), "models")
            if not os.path.exists(local_models_dir):
                os.makedirs(local_models_dir, exist_ok=True)
                
            local_ckpt = os.path.join(local_models_dir, ckpt_name)
            
            ckpt_path = ""
            if os.path.exists(local_ckpt):
                print(f"Loading local checkpoint: {local_ckpt}")
                ckpt_path = local_ckpt
            else:
                print(f"Downloading NoobAI XL 1.1 from HF: {repo_id}")
                try:
                    # Attempt to download the single file if possible
                    # Note: You might need to adjust filename if the repo structure changes
                    # This model is often just 'model.safetensors' or similar in diffusers repo, 
                    # but usually shared as a single file in 'Laxhar/noobai-XL-1.1'
                    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name)
                except Exception:
                    print("Could not download specific filename, trying default diffusers load...")
                    # Fallback to standard diffusers load if it is a repo
                    self.pipeline = AutoPipelineForInpainting.from_pretrained(
                        repo_id,
                        torch_dtype=torch.float16,
                        use_safetensors=True
                    ).to("cuda")
                    self.pipeline.enable_vae_slicing()
                    return

            # If we got a checkpoint path, load via SingleFile
            print("Loading SDXL Base Pipeline from single file...")
            temp_pipe = StableDiffusionXLPipeline.from_single_file(
                ckpt_path,
                torch_dtype=torch.float16,
                use_safetensors=True
            )
            
            print("Converting to AutoPipelineForInpainting...")
            self.pipeline = AutoPipelineForInpainting.from_pipe(temp_pipe).to("cuda")
            self.pipeline.enable_vae_slicing()
            
            del temp_pipe
            gc.collect()
            torch.cuda.empty_cache()

    def get_prompts(self):
        # Prompts optimized for NoobAI XL & Monochrome Manga
        # NoobAI responds very well to "masterpiece, best quality" + danbooru tags
        return {
            "prompt": "masterpiece, best quality, monochrome, greyscale, lineart, manga, comic, sketchy, detailed anatomy, uncensored, penis, vagina, genitals, (white background:1.2), (simple background:1.1)",
            "negative_prompt": "color, colored, 3d, realistic, photorealistic, volumetric lighting, painting, acrylic, source_pony, source_furry, mosaic, censor, bar, blurry, text, watermark, bad anatomy, bad hands, extra digits"
        }