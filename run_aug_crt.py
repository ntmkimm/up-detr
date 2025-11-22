# Full pipeline: Augmentation + MobileCLIP2 + YOLO VPSeg + UP-DETR + Fast Tracker (KCF)
# ---------------------------------------------------------------------------------
# OPTIMIZATIONS:
# 1. Tracker: CSRT -> KCF (Kernelized Correlation Filters) for speed.
# 2. Color Utils: KMeans -> cv2.calcHist (Histogram Correlation) to remove bottleneck.
# ---------------------------------------------------------------------------------

import os
import random
import math
import time
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# --- UP-DETR imports ---
from models.backbone import Backbone, Joiner
from models.detr import DETR
from models.position_encoding import PositionEmbeddingSine
from models.transformer import Transformer
from models.updetr import UPDETR

# -----------------------------
# OPTIMIZED Color Analysis Utilities
# -----------------------------
class ColorUtils:
    @staticmethod
    def calculate_color_similarity(crop1: np.ndarray, crop2: np.ndarray) -> float:
        """
        Replaced KMeans (slow) with Histogram Correlation (fast).
        """
        try:
            if crop1.size == 0 or crop2.size == 0: return 0.0

            # Convert to HSV or LAB for better color separation
            hsv1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2HSV)
            hsv2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2HSV)

            # Handle mask if 4 channels (BGRA)
            mask1 = None
            mask2 = None
            if crop1.shape[2] == 4:
                mask1 = crop1[:, :, 3]
            if crop2.shape[2] == 4:
                mask2 = crop2[:, :, 3]

            # Calculate Histograms (H and S channels usually enough)
            # 30 bins for Hue, 32 for Saturation
            hist1 = cv2.calcHist([hsv1], [0, 1], mask1, [30, 32], [0, 180, 0, 256])
            hist2 = cv2.calcHist([hsv2], [0, 1], mask2, [30, 32], [0, 180, 0, 256])

            # Normalize
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            # Compare Histograms (Method 0: Correlation, Method 2: Intersection, Method 3: Bhattacharyya)
            # Correlation returns 1.0 for perfect match
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # Clip to 0-1 just in case
            return max(0.0, min(1.0, similarity))

        except Exception as e:
            # print(f"Color Error: {e}")
            return 0.0

# -----------------------------
# Utility functions
# -----------------------------

def box_xyxy_to_xywh(box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return x1, y1, x2 - x1, y2 - y1

def box_xywh_to_xyxy(box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    return x, y, x + w, y + h

def clip_box(box, shape):
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)

# --- UP-DETR utility functions ---
def box_cxcywh_to_xyxy_updetr(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes_updetr(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy_updetr(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=out_bbox.device)
    return b

# -----------------------------
# Model Wrappers
# -----------------------------

class YOLOWrapper:
    def __init__(self, model_path: str):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except ImportError:
            self.model = None
            print("Warning: 'ultralytics' not found.")

    def predict(self, image: np.ndarray, conf: float = 0.1) -> List[Dict]:
        if self.model is None: return []
        # optimize: verbose=False, augment=False
        preds = self.model(image, save=False, conf=conf, retina_masks=True, verbose=False, augment=False)
        res = preds[0]
        dets = []
        if res.boxes is not None:
            for i in range(len(res.boxes)):
                xyxy = tuple(res.boxes.xyxy[i].cpu().numpy().astype(int))
                mask = res.masks.data[i].cpu().numpy() if res.masks is not None else None
                dets.append({'bbox': xyxy, 'mask': mask, 'score': float(res.boxes.conf[i])})
        return dets

class MobileCLIP2Wrapper:
    def __init__(self, model_name: str = "MobileCLIP2-S0", pretrained: str = "dfndr2b"):
        try:
            import open_clip
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            self.model.to(self.device).eval()
        except ImportError:
            self.model = None
            print("Warning: 'open_clip' not found.")

    def get_embedding(self, image_crop: np.ndarray) -> Optional[torch.Tensor]:
        if self.model is None: return None
        try:
            # Resize early to avoid heavy PIL conversion of large crops if not needed
            if image_crop.shape[0] > 300 or image_crop.shape[1] > 300:
                image_crop = cv2.resize(image_crop, (224, 224))

            if image_crop.shape[2] == 4:
                img = cv2.cvtColor(image_crop, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
            
            pil_img = Image.fromarray(img)
            tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.model.encode_image(tensor)
                return F.normalize(emb, p=2, dim=-1)
        except Exception:
            return None

class UPDETRWrapper:
    def __init__(self, model_path: str, num_classes: int = 91, num_queries: int = 100, device: str = 'cuda'):
        self.device = device
        self.num_queries = num_queries
        self.model = self._build_updetr(num_classes, num_queries)
        checkpoint = torch.load(model_path, map_location=torch.device(device))['model']
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(device).eval()
        
        # Reduced resolution for faster inference if accuracy permits
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize(384), # Reduced from 420 for slight speedup
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.patch_transform = transforms.Compose([
            transforms.Resize((128, 128)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.support_tensors = None

    def _build_updetr(self, num_classes, num_queries):
        hidden_dim = 256
        backbone = Backbone("resnet50", train_backbone=False, return_interm_layers=False, dilation=False)
        pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        backbone_with_pos_enc = Joiner(backbone, pos_enc)
        backbone_with_pos_enc.num_channels = backbone.num_channels
        transformer = Transformer(d_model=hidden_dim, normalize_before=True, return_intermediate_dec=True)
        return UPDETR(backbone_with_pos_enc, transformer, num_classes=num_classes, num_queries=num_queries, num_patches=10)

    def set_support_images(self, support_dir: str):
        paths = list(Path(support_dir).glob("*.jpg")) + list(Path(support_dir).glob("*.png"))
        if not paths: raise ValueError("No support images found")
        patches = [Image.open(p).convert("RGB") for p in paths]
        
        max_real = self.num_queries // 10 
        if len(patches) > max_real: patches = random.sample(patches, max_real)
        
        tensor_list = [self.patch_transform(p) for p in patches]
        self.support_tensors = torch.stack(tensor_list, dim=0).unsqueeze(0).to(self.device)
        print(f"[UPDETR] Support initialized. Patches: {len(tensor_list)}")

    def predict(self, image: np.ndarray, conf: float = 0.1, upper_conf=0.7) -> List[Dict]:
        if self.support_tensors is None: return []
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_tensor = self.image_transform(img_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(frame_tensor, self.support_tensors)
        
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        scores = probas.max(-1).values
        keep = (scores > conf) & (scores < upper_conf)
        bboxes_scaled = rescale_bboxes_updetr(outputs['pred_boxes'][0, keep], (image.shape[1], image.shape[0]))
        scores_filtered = scores[keep]
        
        return [{'bbox': tuple(box.cpu().numpy().astype(int)), 'mask': None, 'score': float(score)}
                for box, score in zip(bboxes_scaled, scores_filtered)]

# -----------------------------
# High-level pipeline functions
# -----------------------------

def crop_with_mask(image: np.ndarray, bbox: Tuple[int, int, int, int], mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = [max(0, int(c)) for c in bbox]
    if x2 <= x1 or y2 <= y1: return None
    crop = image[y1:y2, x1:x2]
    if mask is None: return crop
    
    mask_crop = mask[y1:y2, x1:x2]
    # Optimization: Resize mask only if shapes differ significantly
    if mask_crop.shape[:2] != crop.shape[:2]:
        mask_bin = cv2.resize((mask_crop > 0.4).astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST) * 255
    else:
        mask_bin = (mask_crop > 0.4).astype(np.uint8) * 255

    bgr = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgr[:, :, 3] = mask_bin
    return bgr

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    # Optimization: Calculate mean once
    mean_color = tuple(map(int, image.mean(axis=(0,1))))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderValue=mean_color)

def augment_support_patch(image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    augmented = []
    # Reduced augmentation to save init time
    augmented.append(("_rot45", rotate_image(image, 45)))
    augmented.append(("_rot315", rotate_image(image, 315)))
    return augmented

def refine_support_images(support_dir: str, refined_dir: str, yolo, conf=0.01, augment: bool = True) -> str:
    os.makedirs(refined_dir, exist_ok=True)
    paths = list(Path(support_dir).glob("*.jpg")) + list(Path(support_dir).glob("*.png"))
    for p in paths:
        img = cv2.imread(str(p))
        if img is None: continue
        dets = yolo.predict(img, conf=conf)
        crop = img
        if len(dets) > 0:
            det = max(dets, key=lambda d: (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]))
            c = crop_with_mask(img, det["bbox"], det.get("mask"))
            if c is not None: crop = c
        
        base_name = p.stem
        cv2.imwrite(str(Path(refined_dir) / f"{base_name}_orig.jpg"), crop)
        if augment:
            aug_list = augment_support_patch(crop)
            for suffix, aug_img in aug_list:
                cv2.imwrite(str(Path(refined_dir) / f"{base_name}{suffix}.jpg"), aug_img)
    return refined_dir

def prepare_output_dir(output_dir):
    if output_dir is None: return
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=False)

# -----------------------------
# Pipeline Core
# -----------------------------

def process_video_pipeline(
    video_path: str,
    support_dir: str,
    yolo: YOLOWrapper,
    updetr: UPDETRWrapper,
    mobileclip: MobileCLIP2Wrapper,
    output_dir: str,
    device: str = 'cuda',
    conf_updetr: Tuple[float, float] = (0.3, 0.7),
    conf_yolo_refined: float = 0.001,
    sim_threshold: float = 0.45,
    augment: bool = True,
    frame_gap_visualize: int = 5
):
    prepare_output_dir(output_dir)

    # --- 1. Prepare Support Images ---
    support_refined = os.path.join(output_dir, "support_refined")
    support_dir = refine_support_images(support_dir, support_refined, yolo, conf=conf_yolo_refined, augment=augment)
    updetr.set_support_images(support_dir)

    # --- 2. Initialize Verification Data ---
    dynamic_references = []
    support_paths = list(Path(support_dir).glob("*.jpg")) + list(Path(support_dir).glob("*.png"))
    for p in support_paths:
        img = cv2.imread(str(p))
        if img is not None:
            emb = mobileclip.get_embedding(img)
            if emb is not None:
                dynamic_references.append((emb, img))
    
    print(f"[INFO] Initialized {len(dynamic_references)} reference samples.")

    # --- Verification Function (Now Faster) ---
    def calc_sim_for_crop(crop_img):
        if crop_img is None: return 0.0, None
        emb = mobileclip.get_embedding(crop_img)
        if emb is None: return 0.0, None
        
        scores = []
        for ref_emb, ref_crop in dynamic_references:
            # Cosine Sim
            emb_sim = F.cosine_similarity(emb, ref_emb).item()
            # Histogram Sim (Optimized)
            col_sim = ColorUtils.calculate_color_similarity(crop_img, ref_crop)
            
            # Weighted sum
            combined = 0.7 * emb_sim + 0.3 * col_sim
            scores.append(combined)
        
        avg_sim = np.mean(scores) if scores else 0.0
        return avg_sim, emb

    # --- 3. Video Loop ---
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # State
    tracker = None 
    tracker_bbox = None
    frames_since_det = 999
    det_interval = 5 
    
    # Thresholds
    TRACKER_DROP_THRESHOLD = 0.3 
    
    from tqdm import tqdm
    pbar = tqdm(total=total_frames, desc="Tracking")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # --- A. Update Tracker (KCF) ---
        tracker_success = False
        if tracker is not None:
            success, box_wh = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box_wh]
                tracker_bbox = clip_box((x, y, x+w, y+h), frame.shape)
                tracker_success = True
            else:
                tracker = None
                tracker_bbox = None
        
        # --- B. Logic to Run Detector ---
        run_detector = (not tracker_success) or (frames_since_det >= det_interval)
        valid_dets = []
        
        if run_detector:
            # 1. UP-DETR Detect
            candidates = updetr.predict(frame, conf=conf_updetr[0], upper_conf=conf_updetr[1])
            
            # 2. Verification & Filtering
            for det in candidates:
                crop = crop_with_mask(frame, det['bbox'], det.get('mask'))
                if crop is None: continue
                
                avg_sim, emb = calc_sim_for_crop(crop)
                if emb is None: continue

                if avg_sim > sim_threshold:
                    det['sim'] = avg_sim
                    det['emb'] = emb
                    valid_dets.append(det)
            
            frames_since_det = 0
        else:
            frames_since_det += 1

        # --- C. Decision Making ---
        final_bbox = None
        status = "Lost"
        best_sim_score = 0.0
        
        # Case 1: Detector found valid match(es)
        if len(valid_dets) > 0:
            bboxes = np.array([d['bbox'] for d in valid_dets])
            avg_bbox = np.mean(bboxes, axis=0).astype(int)
            final_bbox = tuple(avg_bbox)
            
            best_sim_score = np.mean([d['sim'] for d in valid_dets])
            status = "Detected (Avg)"
            
            # --- NEW FAST TRACKER ---
            # Using KCF instead of CSRT
            try:
                tracker = cv2.TrackerKCF_create() 
            except AttributeError:
                 # Fallback for older OpenCV versions
                tracker = cv2.TrackerKCF_create() if hasattr(cv2, 'TrackerKCF_create') else cv2.TrackerCSRT_create()

            xywh = box_xyxy_to_xywh(final_bbox)
            tracker.init(frame, xywh)
            tracker_bbox = final_bbox
                    
        # Case 2: Tracker is valid
        elif tracker_success:
            # Verify tracker quality occasionally
            track_crop = crop_with_mask(frame, tracker_bbox)
            # We still do verification, but now it's faster due to optimized ColorUtils
            track_sim, _ = calc_sim_for_crop(track_crop)
            
            if track_sim < TRACKER_DROP_THRESHOLD:
                tracker = None
                tracker_bbox = None
                final_bbox = None
                status = "Lost (Low Sim)"
            else:
                final_bbox = tracker_bbox
                status = "Tracking (KCF)"
                best_sim_score = track_sim
        
        # --- D. Visualization ---
        if final_bbox is not None and frame_idx % frame_gap_visualize == 0:
            vis_frame = frame.copy()
            x1, y1, x2, y2 = final_bbox
            color = (0, 255, 0) if "Detected" in status else (255, 255, 0)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            info = f"{status} | Sim: {best_sim_score:.2f}"
            cv2.putText(vis_frame, info, (x1, max(0, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg"), vis_frame)

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()
    print(f"Done. Results in {output_dir}")

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == '__main__':
    
    # --- Config ---
    YOLO_WEIGHTS = 'yolov8n.pt'
    UPDETR_WEIGHTS = 'up-detr-coco-fine-tuned-300ep.pth' 
    MOBILECLIP_NAME = 'MobileCLIP2-S0'
    TYPE = 'train'
    
    ROOT_DATASET_DIR = Path(f'/mlcv2/Datasets/ZaloAI2025/track1/{TYPE}/samples')
    TOTAL_OUTPUT_DIR = Path(f'/mlcv2/WorkingSpace/Personal/quannh/Project/Project/ZaloAI2025/src/idea/{TYPE}_fast_tracker')
    TOTAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- Init Models ---
    yolo = YOLOWrapper(YOLO_WEIGHTS)
    updetr = UPDETRWrapper(UPDETR_WEIGHTS, num_queries=100, device='cuda' if torch.cuda.is_available() else 'cpu')
    mobileclip = MobileCLIP2Wrapper(model_name=MOBILECLIP_NAME)
    
    # --- Loop Videos ---
    for VIDEO_PATH in ROOT_DATASET_DIR.glob('**/drone_video.mp4'):
        SUPPORT_DIR = VIDEO_PATH.parent / 'object_images'
        OUTPUT_DIR = TOTAL_OUTPUT_DIR / VIDEO_PATH.parent.stem
        
        # Per-class tuning
        conf_updetr = (0.3, 0.7)
        sim_threshold = 0.40
        augment = True
        
        if "person" in str(VIDEO_PATH.parent.stem).lower():
            conf_updetr = (0.7, 1.0)
            sim_threshold = 0.45
            augment = False
            
        print(f"Processing {VIDEO_PATH.parent.name}...")
        
        process_video_pipeline(
            video_path=str(VIDEO_PATH),
            support_dir=str(SUPPORT_DIR),
            yolo=yolo,
            updetr=updetr,
            mobileclip=mobileclip,
            output_dir=OUTPUT_DIR,
            device='cuda',
            conf_updetr=conf_updetr,
            sim_threshold=sim_threshold,
            augment=augment,
            frame_gap_visualize=5
        )