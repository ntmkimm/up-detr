import torch
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
import cv2
from tqdm import tqdm
import time
import shutil
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision.ops import box_iou

from models.backbone import Backbone, Joiner
from models.detr import DETR
from models.position_encoding import PositionEmbeddingSine
from models.transformer import Transformer
from models.updetr import UPDETR

# Ngăn chặn việc khởi tạo không cần thiết của một số thư viện
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


device = "cuda" if torch.cuda.is_available() else "cpu"

support_dir = Path("/mlcv2/Datasets/ZaloAI2025/track1/public_test/samples/CardboardBox_0/object_images")
video_path = Path("/mlcv2/Datasets/ZaloAI2025/track1/public_test/samples/CardboardBox_0/drone_video.mp4")
output_dir = Path(f"../{video_path.parent.name}_updetr_tracking_shape_rerank")
model_file = './up-detr-coco-fine-tuned-300ep.pth'

num_classes = 91
threshold = 0.3  # Ngưỡng tin cậy cho việc phát hiện
iou_threshold_track = 0.2  # Ngưỡng IoU để xem một bbox là "cùng một đối tượng"

if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

torch.set_grad_enabled(False)


# ==== Build UP-DETR ====
def build_updetr(num_classes=91):
    hidden_dim = 256
    backbone = Backbone("resnet50", train_backbone=False, return_interm_layers=False, dilation=False)
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels

    transformer = Transformer(d_model=hidden_dim, normalize_before=True, return_intermediate_dec=True)

    model = UPDETR(
        backbone_with_pos_enc,
        transformer,
        num_classes=num_classes,
        num_queries=100,
        num_patches=10,
        feature_recon=True,
        query_shuffle=True,
    )
    return model


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h],
                         dtype=torch.float32, device=out_bbox.device)
    return b


# ==== Load model checkpoint ====
model = build_updetr(num_classes=num_classes)
checkpoint = torch.load(model_file)['model']
msg = model.load_state_dict(checkpoint, strict=False)
print(msg)
model = model.to(device).eval()


# ==== Transforms ====
image_transform = T.Compose([
    T.Resize(420),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])
patch_transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])


# ==== Load support patches ====
support_paths = sorted(list(support_dir.glob("*.jpg")) + list(support_dir.glob("*.png")))
assert len(support_paths) > 0, "No support images found!"

print(f"[INFO] Loading {len(support_paths)} support images...")
support_patches = [Image.open(p).convert("RGB") for p in support_paths]
support_tensors = torch.stack([patch_transform(p) for p in support_patches],
                              dim=0).unsqueeze(0).to(device)

# === Placeholder for tracking ===
tracking_patch = torch.zeros_like(support_tensors[:, :1])
has_tracking = False
print(f"[INFO] Support tensor shape: {support_tensors.shape}")


# ==== Inference loop ====
cap = cv2.VideoCapture(str(video_path))
frame_idx = 0
total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
pbar = tqdm(total=total_frame, desc="Processing video")

last_bbox = None  # Lưu bbox [xmin, ymin, xmax, ymax] của frame trước
last_shape_ratio = None # Lưu tỉ lệ w/h của bbox trước

while cap.isOpened():
    ret, frame_bgr = cap.read()
    if not ret:
        break

    t0 = time.time()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tensor = image_transform(frame_pil).unsqueeze(0).to(device)

    # === combine supports + tracking patch ===
    if has_tracking:
        current_support = torch.cat([support_tensors, tracking_patch], dim=1)
    else:
        current_support = support_tensors

    # === forward ===
    with torch.no_grad():
        outputs = model(frame_tensor, current_support)

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    scores = probas.max(-1).values

    keep = scores > threshold
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], frame_pil.size)
    scores = scores[keep]
    
    best_bbox = None
    
    # === Tracking Logic: Re-ranking based on IoU and Score ===
    if bboxes_scaled.shape[0] > 0:
        if last_bbox is not None:
            last_bbox_tensor = torch.tensor(last_bbox, device=device, dtype=torch.float32).unsqueeze(0)
            ious = box_iou(bboxes_scaled, last_bbox_tensor).squeeze(1)
            
            # Lấy các bbox có IoU đủ lớn với bbox của frame trước
            iou_candidates_mask = ious > iou_threshold_track
            
            if iou_candidates_mask.any():
                candidate_bboxes = bboxes_scaled[iou_candidates_mask]
                candidate_scores = scores[iou_candidates_mask]
                
                # --- Re-ranking by Shape Similarity ---
                # Tính tỉ lệ w/h cho các bbox ứng viên
                widths = candidate_bboxes[:, 2] - candidate_bboxes[:, 0]
                heights = candidate_bboxes[:, 3] - candidate_bboxes[:, 1]
                current_ratios = widths / (heights + 1e-6)
                
                # So sánh độ tương đồng về hình dạng với bbox trước
                shape_similarity = torch.abs(current_ratios - last_shape_ratio)
                
                # Ưu tiên bbox có hình dạng giống nhất (độ chênh lệch nhỏ nhất)
                best_candidate_idx = torch.argmin(shape_similarity)
                best_bbox = candidate_bboxes[best_candidate_idx].cpu().numpy()
                print(f"[TRACK] Found {iou_candidates_mask.sum()} candidates. Best by shape similarity.")

            else:
                # Mất dấu: Không có bbox nào đủ gần. Chọn bbox có score cao nhất để bắt đầu lại.
                print("[TRACK] Lost track (no IoU match). Re-initializing with highest score detection.")
                best_idx = scores.argmax()
                best_bbox = bboxes_scaled[best_idx].cpu().numpy()
        else:
            # Frame đầu tiên hoặc sau khi mất dấu: Chọn bbox có score cao nhất
            print("[TRACK] Initializing track with the highest score detection.")
            best_idx = scores.argmax()
            best_bbox = bboxes_scaled[best_idx].cpu().numpy()

    # === Update tracking state ===
    if best_bbox is not None:
        last_bbox = best_bbox.astype(int)
        
        # Cập nhật tỉ lệ hình dạng
        w = last_bbox[2] - last_bbox[0]
        h = last_bbox[3] - last_bbox[1]
        last_shape_ratio = w / (h + 1e-6)

        # Cập nhật tracking patch
        xmin, ymin, xmax, ymax = last_bbox
        crop = frame_pil.crop((xmin, ymin, xmax, ymax))
        tracking_patch = patch_transform(crop).unsqueeze(0).unsqueeze(0).to(device)
        has_tracking = True
    else:
        # Không phát hiện được đối tượng nào -> reset tracking
        if has_tracking:
            print("[TRACK] No object detected. Resetting tracker.")
        last_bbox = None
        last_shape_ratio = None
        has_tracking = False

    # === Visualization ===
    if last_bbox is not None:
        xmin, ymin, xmax, ymax = last_bbox
        cv2.rectangle(frame_bgr, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame_bgr, f"Tracking", (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    t1 = time.time()
    pbar.set_postfix({"fps": 1.0 / (t1 - t0)})
    pbar.update(1)

    # Lưu frame kết quả
    if last_bbox is not None:
        cv2.imwrite(f"{str(output_dir)}/frame_{frame_idx:05d}.jpg", frame_bgr)
    
    # cv2.imwrite(f"current_frame.jpg", frame_bgr) # Để debug nếu cần
    frame_idx += 1

cap.release()
pbar.close()

print("✅ Done!")