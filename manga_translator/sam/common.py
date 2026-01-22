import torch
import numpy as np
from PIL import Image
from transformers import Sam2Processor, Sam2Model
from typing import List
from ..config import SAMConfig, SAMPrecision, SAM
import os
from pathlib import Path
import cv2
from ..utils import InfererModule, TextBlock, ModelWrapper, Quadrilateral
import gc

class CommonSAM:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None

    async def load(self, device: str):
        raise NotImplementedError

    async def predict(self, image: np.ndarray, bboxes: List[np.ndarray], config: SAMConfig, verbose: bool = False) -> List[np.ndarray]:
        raise NotImplementedError

def setup_huggingface_proxy(host="127.0.0.1", port="7897"):
    proxy_url = f"http://{host}:{port}"
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    os.environ['HF_HUB_PROXY'] = proxy_url
    os.environ['GIT_HTTP_PROXY'] = proxy_url
    print(f"Proxy configured for Hugging Face: {proxy_url}")
    

class OfflineSAM(CommonSAM, ModelWrapper):
    _MODEL_SUB_DIR = 'sam'

    async def _recognize(self, *args, **kwargs):
        return await self.infer(*args, **kwargs)
    
    async def _load(self):
        pass
    
    async def _unload(self):
        pass
    
    async def _infer(self, image: np.ndarray, textlines: List[Quadrilateral], args: SAMConfig, verbose: bool = False) -> List[Quadrilateral]:
        pass


class SAM2Inpainter(OfflineSAM):
    async def load(self, device: str):
        if self.model is not None and self.device == device:
            return
        self.device = device
        repo_id = "facebook/sam2.1-hiera-large" 
        cache_dir = "models/sam"
        local_snapshot_path = Path(r"E:\MangaTranslator\MangaTranslator\models\sam\models--facebook--sam2.1-hiera-large\snapshots")

        target_path = repo_id # Default to repo_id for online/cache lookup
        use_local = False

        if local_snapshot_path.exists():
            snapshots = list(local_snapshot_path.iterdir())
            if snapshots:
                # Use the latest or first snapshot found
                target_path = str(snapshots[0])
                use_local = True
                print(f"SAM 2.1: Loading from local snapshot: {target_path}")

        try:
            setup_huggingface_proxy()
            # If use_local is True, we tell transformers to ONLY look at that path
            self.processor = Sam2Processor.from_pretrained(
                target_path, 
                local_files_only=use_local
            )
            self.model = Sam2Model.from_pretrained(
                target_path,
                local_files_only=use_local
            ).to(device)
            
            self.model.eval()
            print("SAM 2.1 model loaded successfully.")

        except Exception as e:
            print(f"Error loading SAM 2.1: {e}")
            # Final fallback: Try standard cache if the custom path failed
            if use_local:
                print("Custom path failed, falling back to default cache...")
                self.processor = Sam2Processor.from_pretrained(repo_id)
                self.model = Sam2Model.from_pretrained(repo_id).to(device)
                self.model.eval()
            else:
                raise
            
    async def predict(
        self, 
        image: np.ndarray, 
        bboxes: List[List[int]], 
        config: SAMConfig, 
        verbose: bool = False
    ) -> List[np.ndarray]:

        if not bboxes or config.method == SAM.none:
            return [np.zeros(image.shape[:2], dtype=np.uint8) for _ in bboxes]

        # 1. Memory Setup
        device_type = self.device.split(':')[0]
        dtype_map = {
            SAMPrecision.fp32: torch.float32,
            SAMPrecision.fp16: torch.float16,
            SAMPrecision.bf16: torch.bfloat16
        }
        target_dtype = dtype_map.get(config.precision, torch.float32)
        
        # 2. Pre-Inference Cleanup
        if device_type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

        img_h, img_w = image.shape[:2]
        pil_img = Image.fromarray(image)
        
        # 3. Prompt Preparation
        expand_ratio = 0.5
        all_input_boxes = []
        all_input_points = []
        all_input_labels = []

        for box in bboxes:
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            anchors = [
                [(x1 + x2) / 2, (y1 + y2) / 2],
                [x1 + bw*0.1, y1 + bh*0.1], [x2 - bw*0.1, y1 + bh*0.1],
                [x1 + bw*0.1, y2 - bh*0.1], [x2 - bw*0.1, y2 - bh*0.1]
            ]
            all_input_points.append(anchors)
            all_input_labels.append([1, 1, 1, 1, 1])
            
            # Use ratio-based expansion
            px1 = max(0, x1 - (bw * expand_ratio))
            py1 = max(0, y1 - (bh * expand_ratio))
            px2 = min(img_w, x2 + (bw * expand_ratio))
            py2 = min(img_h, y2 + (bh * expand_ratio))
            all_input_boxes.append([px1, py1, px2, py2])

        # 4. CHUNKED PROCESSING
        # Process prompts in small groups to save VRAM
        # Adjust chunk_size (e.g., 5 or 10) based on your GPU capacity
        chunk_size = 8 
        final_masks = []

        try:
            # We process the image once to get features, then iterate prompts
            # Note: For Sam2Model, we pass all prompts at once or chunk them.
            # Chunking the prompt tensors is the most memory-efficient way.
            
            for i in range(0, len(all_input_boxes), chunk_size):
                chunk_boxes = all_input_boxes[i : i + chunk_size]
                chunk_points = all_input_points[i : i + chunk_size]
                chunk_labels = all_input_labels[i : i + chunk_size]

                inputs = self.processor(
                    images=pil_img, 
                    input_boxes=[chunk_boxes], 
                    input_points=[chunk_points],
                    input_labels=[chunk_labels],
                    return_tensors="pt"
                ).to(self.device)

                with torch.autocast(device_type=device_type, dtype=target_dtype):
                    with torch.no_grad():
                        outputs = self.model(**inputs)

                # Post-process chunk
                processed_chunk = self.processor.post_process_masks(
                    masks=outputs.pred_masks, 
                    original_sizes=inputs.original_sizes,
                    binarize=True
                )[0]

                for j in range(processed_chunk.shape[0]):
                    # Area ranking
                    idx = torch.argmax(processed_chunk[j].sum(dim=(1, 2))).item()
                    mask_data = (processed_chunk[j, idx].cpu().numpy() > 0).astype(np.uint8) * 255
                    final_masks.append(mask_data)

                # Intermediate cleanup
                del inputs, outputs, processed_chunk
                if device_type == 'cuda':
                    torch.cuda.empty_cache()

        finally:
            # Final cleanup
            if device_type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
        return final_masks