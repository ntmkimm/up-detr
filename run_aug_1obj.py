# Full pipeline: Augmentation + MobileCLIP2 + YOLO VPSeg + UP-DETR + Tracker (Hungarian + Kalman motion)
# ---------------------------------------------------------------------------------
# This file implements a complete pipeline combining:
# - Augmentation: creating synthetic backgrounds with pasted object crops
# - MobileCLIP2 embeddings for visual verification
# - YOLO (VPE) for prompt-based detection / masks
# - UP-DETR for few-shot detection (support-based)
# - A robust tracker implementing: Kalman motion model + Hungarian assignment
#
# Notes:
# - Fill the model paths and weight files before running.
# - This is designed as an end-to-end Python module you can run locally.
# - Dependencies: torch, torchvision, open_clip (or MobileCLIP2 interface), ultralytics (YOLO),
#   scipy, numpy, opencv-python, PIL
# ---------------------------------------------------------------------------------

import os
import random
import math
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou

# --- UP-DETR imports ---
# Assumption: The following files from the UP-DETR repository are placed in a 'models' subdirectory.
# (backbone.py, detr.py, position_encoding.py, transformer.py, updetr.py)
from models.backbone import Backbone, Joiner
from models.detr import DETR
from models.position_encoding import PositionEmbeddingSine
from models.transformer import Transformer
from models.updetr import UPDETR

# -----------------------------
# Utility functions
# -----------------------------

def box_xyxy_to_xywh(box: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    """Converts a bounding box from [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return cx, cy, w, h


def box_xywh_to_xyxy(box: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    """Converts a bounding box from [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = box
    x1 = int(round(cx - 0.5 * w))
    y1 = int(round(cy - 0.5 * h))
    x2 = int(round(cx + 0.5 * w))
    y2 = int(round(cy + 0.5 * h))
    return x1, y1, x2, y2


def iou_xyxy(boxA, boxB):
    """Calculates the Intersection over Union (IoU) between two bounding boxes."""
    a = torch.tensor(boxA).unsqueeze(0).float()
    b = torch.tensor(boxB).unsqueeze(0).float()
    return float(box_iou(a, b).item())

# --- UP-DETR utility functions ---
def box_cxcywh_to_xyxy_updetr(x):
    """Converts UP-DETR's [cx, cy, w, h] format to [x1, y1, x2, y2]."""
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes_updetr(out_bbox, size):
    """Rescales bounding boxes from relative to absolute image coordinates."""
    img_w, img_h = size
    b = box_cxcywh_to_xyxy_updetr(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h],
                         dtype=torch.float32, device=out_bbox.device)
    return b

# -----------------------------
# Simple Kalman Filter for constant velocity model
# -----------------------------

class KalmanFilter:
    """A simple Kalman Filter for tracking object motion."""
    def __init__(self, dt: float = 1.0, std_pos: float = 1.0, std_vel: float = 1.0):
        self._dim_x = 8  # State: [cx, cy, w, h, vx, vy, vw, vh]
        self._dim_z = 4  # Measurement: [cx, cy, w, h]

        self.F = np.eye(self._dim_x)
        for i in range(4):
            self.F[i, 4 + i] = dt

        self.H = np.zeros((self._dim_z, self._dim_x))
        for i in range(4):
            self.H[i, i] = 1.0

        q_pos, q_vel = std_pos ** 2, std_vel ** 2
        self.Q = np.eye(self._dim_x)
        for i in range(4):
            self.Q[i, i] = q_pos
            self.Q[4 + i, 4 + i] = q_vel
            
        self.R = np.eye(self._dim_z) * (std_pos ** 2)
        self.x = np.zeros((self._dim_x, 1))
        self.P = np.eye(self._dim_x) * 10.0

    def initiate(self, measurement: np.ndarray):
        """Initializes the filter state with the first detection."""
        self.x[:4, 0] = measurement.reshape(4)
        self.P = np.eye(self._dim_x) * 10.0

    def predict(self):
        """Predicts the next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].reshape(-1)

    def update(self, measurement: np.ndarray):
        """Updates the state with a new measurement."""
        z = measurement.reshape((4, 1))
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self._dim_x) - K @ self.H) @ self.P
        return self.x[:4].reshape(-1)

# -----------------------------
# Track class for managing tracked objects
# -----------------------------

class Track:
    """Represents a single tracked object."""
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], embedding: Optional[torch.Tensor], similarity: float, frame_idx: int, score: float = 1.0, max_age: int = 30):
        self.track_id = track_id
        self.kf = KalmanFilter()
        self.kf.initiate(np.array(box_xyxy_to_xywh(bbox), dtype=float))
        self.bbox = bbox
        self.last_embedding = embedding.cpu() if embedding is not None else None
        self.last_frame = frame_idx
        self.score = score
        self.time_since_update = 0
        self.max_age = max_age
        self.similarity = similarity

    def predict(self):
        """Predicts the new bounding box for the current frame."""
        pred_xywh = self.kf.predict()
        self.bbox = box_xywh_to_xyxy(pred_xywh)
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox: Tuple[int, int, int, int], embedding: Optional[torch.Tensor], similarity: float, frame_idx: int, score: float = 1.0):
        """Updates the track with a new detection."""
        meas_xywh = np.array(box_xyxy_to_xywh(bbox), dtype=float)
        self.kf.update(meas_xywh)
        self.bbox = bbox
        if embedding is not None:
            self.last_embedding = embedding.cpu()
        self.last_frame = frame_idx
        self.score = score
        self.similarity = similarity
        self.time_since_update = 0

    def is_dead(self):
        """Checks if the track has been lost for too long."""
        return self.time_since_update > self.max_age

# -----------------------------
# Hungarian + embedding & IoU association Tracker
# -----------------------------

class Tracker:
    """Manages all tracks and associates detections using the Hungarian algorithm."""
    def __init__(self, max_age: int = 30, iou_threshold: float = 0.3, emb_threshold: float = 0.35,
                 lambda_emb: float = 0.6, lambda_iou: float = 0.4):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.emb_threshold = emb_threshold
        self.lambda_emb = lambda_emb
        self.lambda_iou = lambda_iou
        self.tracks: List[Track] = []
        self._next_id = 1

    def _compute_cost_matrix(self, detections: List[Dict], tracks: List[Track]) -> np.ndarray:
        """Computes a cost matrix based on IoU and embedding similarity."""
        num_dets, num_trks = len(detections), len(tracks)
        if num_trks == 0 or num_dets == 0:
            return np.zeros((num_dets, num_trks))
        
        costs = np.zeros((num_dets, num_trks))
        for i, det in enumerate(detections):
            for j, trk in enumerate(tracks):
                iou_cost = 1.0 - iou_xyxy(det['bbox'], trk.bbox)
                
                emb_cost = 1.0
                if det.get('embedding') is not None and trk.last_embedding is not None:
                    sim = F.cosine_similarity(det['embedding'].cpu(), trk.last_embedding).item()
                    emb_cost = 1.0 - sim
                
                costs[i, j] = self.lambda_emb * emb_cost + self.lambda_iou * iou_cost
        return costs

    def step(self, detections: List[Dict], frame_idx: int):
        """Performs a full tracking step for a new frame."""
        for tr in self.tracks:
            tr.predict()

        if not detections:
            self.tracks = [t for t in self.tracks if not t.is_dead()]
            return

        cost_matrix = self._compute_cost_matrix(detections, self.tracks)
        
        if cost_matrix.size > 0:
            row_idx, col_idx = linear_sum_assignment(cost_matrix)
            matches, unmatched_dets, unmatched_trks = [], set(range(len(detections))), set(range(len(self.tracks)))
            
            for r, c in zip(row_idx, col_idx):
                det, trk = detections[r], self.tracks[c]
                iou_val = iou_xyxy(det['bbox'], trk.bbox)
                emb_sim = F.cosine_similarity(det['embedding'].cpu(), trk.last_embedding).item() if det.get('embedding') is not None and trk.last_embedding is not None else -1.0

                if emb_sim >= self.emb_threshold or iou_val >= self.iou_threshold:
                    trk.update(det['bbox'], det.get('embedding'), det.get('similarity'), frame_idx, score=det.get('score', 1.0))
                    if r in unmatched_dets: unmatched_dets.remove(r)
                    if c in unmatched_trks: unmatched_trks.remove(c)
        else:
            unmatched_dets = set(range(len(detections)))
            
        for i in unmatched_dets:
            det = detections[i]
            new_track = Track(self._next_id, det['bbox'], det.get('embedding'), det.get('similarity'), frame_idx, score=det.get('score', 1.0), max_age=self.max_age)
            self.tracks.append(new_track)
            self._next_id += 1
            
        self.tracks = [t for t in self.tracks if not t.is_dead()]

    def get_active_tracks(self) -> List[Track]:
        if len(self.tracks) > 0:
            return [self.tracks[-1]]
        else:
            return []
        # return [t for t in self.tracks if t.time_since_update == 0]

# -----------------------------
# Model Wrappers
# -----------------------------

import numpy as np

def nms_detections(dets, iou_thresh=0.5):
    """
    dets: list các detection {'bbox': (x1,y1,x2,y2), 'mask': ..., 'score': float}
    """
    if len(dets) == 0:
        return []
    
    # Convert bbox list → numpy
    bboxes = np.array([d['bbox'] for d in dets], dtype=np.float32)
    scores = np.array([d['score'] for d in dets], dtype=np.float32)
    
    # Sort theo score giảm dần
    order = scores.argsort()[::-1]
    
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Lấy bbox để tính IoU
        xx1 = np.maximum(bboxes[i, 0], bboxes[order[1:], 0])
        yy1 = np.maximum(bboxes[i, 1], bboxes[order[1:], 1])
        xx2 = np.minimum(bboxes[i, 2], bboxes[order[1:], 2])
        yy2 = np.minimum(bboxes[i, 3], bboxes[order[1:], 3])

        # Intersection
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        # Area
        area_i = (bboxes[i, 2] - bboxes[i, 0]) * (bboxes[i, 3] - bboxes[i, 1])
        area_others = (bboxes[order[1:], 2] - bboxes[order[1:], 0]) * \
                      (bboxes[order[1:], 3] - bboxes[order[1:], 1])

        # IoU
        iou = inter / (area_i + area_others - inter + 1e-6)

        # Giữ lại các bbox có IoU < threshold
        order = order[1:][iou < iou_thresh]
    
    # Trả về dets sau NMS
    return [dets[i] for i in keep]


class YOLOWrapper:
    """Wrapper for the YOLO detection model."""
    def __init__(self, model_path: str):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
        except ImportError:
            self.model = None
            print("Warning: 'ultralytics' not found. YOLO detection is disabled.")

    def predict(self, image: np.ndarray, conf: float = 0.1) -> List[Dict]:
        if self.model is None: return []
        
        preds = self.model(image, save=False, conf=conf, retina_masks=True, verbose=False)
        res = preds[0]
        dets = []
        if res.boxes is not None:
            for i in range(len(res.boxes)):
                xyxy = tuple(res.boxes.xyxy[i].cpu().numpy().astype(int))
                mask = res.masks.data[i].cpu().numpy() if res.masks is not None else None
                dets.append({'bbox': xyxy, 'mask': mask, 'score': float(res.boxes.conf[i])})
        return dets

class MobileCLIP2Wrapper:
    """Wrapper for the MobileCLIP2 model for embedding extraction."""
    def __init__(self, model_name: str = "MobileCLIP2-S0", pretrained: str = "dfndr2b"):
        try:
            import open_clip
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
            self.model.to(self.device).eval()
        except ImportError:
            self.model = None
            print("Warning: 'open_clip' not found. MobileCLIP2 embedding is disabled.")

    def get_embedding(self, image_crop: np.ndarray) -> Optional[torch.Tensor]:
        if self.model is None: return None
        
        img_rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGRA2RGB if image_crop.shape[2] == 4 else cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(tensor)
            return F.normalize(emb, p=2, dim=-1)
        

def nms_bboxes(dets, iou_thresh=0.5):
    if len(dets) == 0:
        return []
    
    bboxes = np.array([d['bbox'] for d in dets], dtype=np.float32)
    scores = np.array([d['score'] for d in dets], dtype=np.float32)
    
    # sort by score desc
    order = scores.argsort()[::-1]
    keep = []
    
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        
        xx1 = np.maximum(bboxes[i, 0], bboxes[order[1:], 0])
        yy1 = np.maximum(bboxes[i, 1], bboxes[order[1:], 1])
        xx2 = np.minimum(bboxes[i, 2], bboxes[order[1:], 2])
        yy2 = np.minimum(bboxes[i, 3], bboxes[order[1:], 3])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        area_i = (bboxes[i, 2] - bboxes[i, 0]) * (bboxes[i, 3] - bboxes[i, 1])
        area_o = (bboxes[order[1:], 2] - bboxes[order[1:], 0]) * (bboxes[order[1:], 3] - bboxes[order[1:], 1])
        
        iou = inter / (area_i + area_o - inter + 1e-6)
        
        order = order[1:][iou < iou_thresh]
    
    return [dets[i] for i in keep]

class UPDETRWrapper:
    """Wrapper for the UP-DETR few-shot detection model."""
    def __init__(self, model_path: str, num_classes: int = 91, device: str = 'cuda'):
        self.device = device
        self.model = self._build_updetr(num_classes)
        checkpoint = torch.load(model_path, map_location=torch.device(device))['model']
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(device).eval()

        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(420),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.patch_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.support_tensors = None

    def _build_updetr(self, num_classes):
        """Helper to construct the UP-DETR model architecture."""
        hidden_dim = 256
        backbone = Backbone("resnet50", train_backbone=False, return_interm_layers=False, dilation=False)
        pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        backbone_with_pos_enc = Joiner(backbone, pos_enc)
        backbone_with_pos_enc.num_channels = backbone.num_channels
        transformer = Transformer(d_model=hidden_dim, normalize_before=True, return_intermediate_dec=True)
        return UPDETR(backbone_with_pos_enc, transformer, num_classes=num_classes, num_queries=100, num_patches=10)

    def set_support_images(self, support_dir: str):
        """Loads and transforms support images."""
        paths = list(Path(support_dir).glob("*.jpg")) + list(Path(support_dir).glob("*.png"))
        if not paths: raise ValueError("No support images found in support_dir for UP-DETR")
        patches = [Image.open(p).convert("RGB") for p in paths]
        self.support_tensors = torch.stack([self.patch_transform(p) for p in patches], dim=0).unsqueeze(0).to(self.device)

    def predict(self, image: np.ndarray, conf: float = 0.1, upper_conf=0.7) -> List[Dict]:
        if self.support_tensors is None:
            return []
        
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_tensor = self.image_transform(img_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(frame_tensor, self.support_tensors)

        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        scores = probas.max(-1).values

        keep = (scores > conf) & (scores < upper_conf)

        bboxes_scaled = rescale_bboxes_updetr(
            outputs['pred_boxes'][0, keep],
            (image.shape[1], image.shape[0])
        )
        scores_filtered = scores[keep]

        dets = [
            {
                'bbox': tuple(box.cpu().numpy().astype(int)),
                'mask': None,
                'score': float(score)
            }
            for box, score in zip(bboxes_scaled, scores_filtered)
        ]

        #  Apply NMS
        dets = nms_bboxes(dets, iou_thresh=0.5)

        return dets

# -----------------------------
# High-level pipeline functions
# -----------------------------

def crop_with_mask(image: np.ndarray, bbox: Tuple[int, int, int, int], mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """Crops an object from an image using its bounding box and an optional mask."""
    x1, y1, x2, y2 = [max(0, int(c)) for c in bbox]
    if x2 <= x1 or y2 <= y1: return None
    crop = image[y1:y2, x1:x2]
    
    if mask is None: return crop
    
    mask_crop = mask[y1:y2, x1:x2]
    mask_bin = (mask_crop > 0.4).astype(np.uint8) * 255
    if mask_bin.shape[:2] != crop.shape[:2]:
        mask_bin = cv2.resize(mask_bin, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    bgr = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    bgr[:, :, 3] = mask_bin
    return bgr


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    mean_color = image.mean(axis=(0,1)).astype(np.uint8).tolist()  # [B,G,R]

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        borderValue=mean_color
    )
    return rotated

def augment_support_patch(image: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    augmented = []
    h, w = image.shape[:2]

    # Rotation
    rot45 = rotate_image(image, 45)
    # augmented.append(("_rot45", rot45))
    
    rot135 = rotate_image(image, 135)
    # augmented.append(("_rot135", rot135))

    # Scale only (shrink)
    scale_factor = 0.2
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    if new_w > 0 and new_h > 0:
        small = cv2.resize(rot45, (new_w, new_h), interpolation=cv2.INTER_AREA)
        augmented.append(("_small_rot45", small))
        
        small = cv2.resize(rot135, (new_w, new_h), interpolation=cv2.INTER_AREA)
        augmented.append(("_small_rot135", small))

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
        
        # Save orig
        base_name = p.stem
        cv2.imwrite(str(Path(refined_dir) / f"{base_name}_orig.jpg"), crop)
        # Save augmentations
        if augment:
            aug_list = augment_support_patch(crop)
            for suffix, aug_img in aug_list:
                cv2.imwrite(str(Path(refined_dir) / f"{base_name}{suffix}.jpg"), aug_img)
    return refined_dir


import shutil
def prepare_output_dir(output_dir):
    """Delete folder if exists, then recreate it clean."""
    if output_dir is None:
        return

    if os.path.exists(output_dir):
        print(f"[INFO] Removing existing output dir: {output_dir}")
        shutil.rmtree(output_dir)

    print(f"[INFO] Creating new output dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=False)
    

import matplotlib.pyplot as plt
import numpy as np

# Lấy bảng màu tab20 (20 màu phân biệt tốt)
color_map = plt.get_cmap("tab20").colors

def get_track_color(track_id):
    c = color_map[track_id % 20]
    return (int(c[2]*255), int(c[1]*255), int(c[0]*255))  # RGB -> BGR

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
    frame_sample_rate: int = 1,
    sim_threshold: float = 0.45,
    tracker_max_age: int = 30,
    augment: bool = True
):
    """Main function to run the complete tracking pipeline on a video."""
    prepare_output_dir(output_dir)

    # --- Clean support images using YOLO before UP-DETR ---
    support_refined = os.path.join(output_dir, "support_refined")
    support_dir = refine_support_images(support_dir, support_refined, yolo, conf=conf_yolo_refined, augment=augment)
    updetr.set_support_images(support_dir)
    tracker = Tracker(max_age=tracker_max_age, emb_threshold=sim_threshold)

    # --- Compute Average Support Embedding for Verification ---
    support_paths = list(Path(support_dir).glob("*.jpg")) + list(Path(support_dir).glob("*.png"))
    support_embs = [mobileclip.get_embedding(cv2.imread(str(p))) for p in support_paths if cv2.imread(str(p)) is not None]
    support_embs = [emb for emb in support_embs if emb is not None]
    support_emb_avg = torch.stack(support_embs).mean(dim=0) if support_embs else None

    # --- Video Processing Loop ---
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    from tqdm import tqdm
    pbar = tqdm(total=total_frames, desc="Processing video")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx % frame_sample_rate == 0:
            # 1. Detect with both YOLO and UP-DETR
            all_detections = updetr.predict(frame, conf=conf_updetr[0], upper_conf=conf_updetr[1])
            # 2. Filter, embed, and prepare detections for the tracker
            dets_for_tracker = []
            for det in all_detections:
                crop = crop_with_mask(frame, det['bbox'], det.get('mask'))
                if crop is None: continue
                
                emb = mobileclip.get_embedding(crop)
                if emb is not None and support_emb_avg is not None:
                    sim = F.cosine_similarity(emb, support_emb_avg).item() 
                    if sim < sim_threshold:
                        continue # Skip if not visually similar to support images
                det['similarity'] = sim 
                det['embedding'] = emb
                dets_for_tracker.append(det)

            # 3. Update tracker
            tracker.step(dets_for_tracker, frame_idx)

            # 4. # =========== visualize start ==========
            vis_frame = frame.copy()
            overlay_text = []  # gom text lại 1 chỗ

            for tr in tracker.get_active_tracks():
                
                x1, y1, x2, y2 = tr.bbox
                if (x2 - x1) * (y2 - y1) <= 50:
                    continue
                score = tr.score
                sim = tr.similarity

                # màu dễ nhìn
                color = get_track_color(tr.track_id)

                # bbox
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

                # gom text vào overlay
                overlay_text.append(
                    f"ID {tr.track_id} | Score {score:.2f} | Sim {sim:.2f}"
                )

            # ===== Render text vào góc trái =====
            if overlay_text:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1

                # Background for better readability
                text_block = "\n".join(overlay_text)
                lines = text_block.split("\n")

                # Vẽ nền đen mờ
                padding = 5
                text_w = max([cv2.getTextSize(t, font, font_scale, thickness)[0][0] for t in lines])
                text_h = int(len(lines) * 22)

                cv2.rectangle(vis_frame, (5, 5), (15 + text_w, 15 + text_h), (0, 0, 0), -1)
                cv2.addWeighted(vis_frame, 1.0, vis_frame, 0, 0)  # đảm bảo alpha OK

                # Vẽ từng dòng
                y = 20
                for line in lines:
                    cv2.putText(vis_frame, line, (10, y), font, font_scale, (255, 255, 255), 2)
                    y += 22

            # Save frame
            if tracker.get_active_tracks():
                cv2.imwrite(os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg"), vis_frame)
            # =========== visualize end ==========
            
        frame_idx += 1
        pbar.update(1)
        
    cap.release()
    pbar.close()
    print(f"Processing complete. Results saved to {output_dir}")

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == '__main__':
    
    YOLO_WEIGHTS = 'yolov8n.pt'
    UPDETR_WEIGHTS = 'up-detr-coco-fine-tuned-300ep.pth'
    MOBILECLIP_NAME = 'MobileCLIP2-S0'
    TYPE = 'public_test'
    ROOT_DATASET_DIR = Path(f'/mlcv2/Datasets/ZaloAI2025/track1/{TYPE}/samples')
    TOTAL_OUTPUT_DIR = Path(f'/mlcv2/WorkingSpace/Personal/quannh/Project/Project/ZaloAI2025/src/idea/{TYPE}_aug1_1obj_updetr_mobileclip2_tracker')
    TOTAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    yolo = YOLOWrapper(YOLO_WEIGHTS)
    updetr = UPDETRWrapper(UPDETR_WEIGHTS, device='cuda' if torch.cuda.is_available() else 'cpu',)
    mobileclip = MobileCLIP2Wrapper(model_name=MOBILECLIP_NAME)
    
    for VIDEO_PATH in sorted(ROOT_DATASET_DIR.glob('**/drone_video.mp4')):
        SUPPORT_DIR = VIDEO_PATH.parent / 'object_images'
        if VIDEO_PATH.parent.stem.lower().find('life') == -1:
            continue
        OUTPUT_DIR = TOTAL_OUTPUT_DIR / VIDEO_PATH.parent.stem
        conf_updetr = (0.01, 0.7)  # default
        sim_threshold = 0.35
        augment = True
        if "person" in str(VIDEO_PATH.parent.stem).lower():
            conf_updetr = (0.7, 1)  
            sim_threshold = 0.45
            augment = False
        # else:
        #     continue
        process_video_pipeline(
            video_path=str(VIDEO_PATH),
            support_dir=str(SUPPORT_DIR),
            yolo=yolo,
            updetr=updetr,
            mobileclip=mobileclip,
            output_dir=OUTPUT_DIR,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            conf_updetr=conf_updetr,
            conf_yolo_refined=0.001,
            frame_sample_rate=1,
            sim_threshold=sim_threshold,
            tracker_max_age=2,
            augment=augment
        )
    