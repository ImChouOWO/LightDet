"""
Image
   ↓
Backbone
   ↓
Feature Map [B,C,H,W]
   ↓
Flatten
   ↓
Cell Features [B,N,C]
   ↓
bbox
   ↓
中心點 → 分配 cell
   ↓
ROIAlign
   ↓
ROI Feature [M,C]
   ↓
放入對應 cell
   ↓
roi_feats [B,N,K,C]
roi_mask  [B,N,K]
"""



import torch
import torch.nn.functional as F
from torchvision.ops import roi_align


def build_cell_roi_tensor(
    feat_map,
    boxes_per_image,
    image_size,
    roi_out_size=3,
    max_rois_per_cell=12
):
    B, C, H, W = feat_map.shape
    N = H * W
    device = feat_map.device
    dtype = feat_map.dtype

    cell_feats = feat_map.flatten(2).transpose(1, 2).contiguous()

    h_img, w_img = image_size

    all_boxes = []
    all_batch_ids = []

    for b, boxes in enumerate(boxes_per_image):
        if boxes.numel() == 0:
            continue

        boxes = boxes.to(device=device, dtype=dtype).clone()

        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w_img - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h_img - 1)

        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]

        if boxes.numel() == 0:
            continue

        batch_ids = torch.full(
            (boxes.shape[0],),
            b,
            device=device,
            dtype=torch.long
        )

        all_boxes.append(boxes)
        all_batch_ids.append(batch_ids)

    if len(all_boxes) == 0:
        roi_mean = torch.zeros(B, N, C, device=device, dtype=dtype)
        roi_feats = torch.empty(B, N, 0, C, device=device, dtype=dtype)
        roi_mask = torch.empty(B, N, 0, device=device, dtype=torch.bool)
        fused_feats = torch.cat([cell_feats, roi_mean], dim=-1)
        return cell_feats, roi_feats, roi_mask, roi_mean, fused_feats

    boxes = torch.cat(all_boxes, dim=0)
    batch_ids = torch.cat(all_batch_ids, dim=0)

    cx = (boxes[:, 0] + boxes[:, 2]) * 0.5
    cy = (boxes[:, 1] + boxes[:, 3]) * 0.5

    gx = torch.clamp((cx / w_img * W).long(), 0, W - 1)
    gy = torch.clamp((cy / h_img * H).long(), 0, H - 1)
    cell_ids = gy * W + gx

    boxes_feat = boxes.clone()
    boxes_feat[:, [0, 2]] = boxes_feat[:, [0, 2]] * (W / w_img)
    boxes_feat[:, [1, 3]] = boxes_feat[:, [1, 3]] * (H / h_img)

    rois = torch.cat(
        [
            batch_ids.to(dtype).unsqueeze(1),
            boxes_feat
        ],
        dim=1
    )

    pooled_chunks = []
    chunk_size = 512

    for start in range(0, rois.shape[0], chunk_size):
        end = start + chunk_size

        pooled_chunk = roi_align(
            feat_map,
            rois[start:end],
            output_size=(roi_out_size, roi_out_size),
            spatial_scale=1.0,
            aligned=True
        )

        pooled_chunk = F.adaptive_avg_pool2d(pooled_chunk, 1).flatten(1)
        pooled_chunks.append(pooled_chunk)

    pooled = torch.cat(pooled_chunks, dim=0)

    global_cell_ids = batch_ids * N + cell_ids

    roi_sum_flat = torch.zeros(B * N, C, device=device, dtype=dtype)
    roi_count_flat = torch.zeros(B * N, 1, device=device, dtype=dtype)

    roi_sum_flat.index_add_(0, global_cell_ids, pooled)

    ones = torch.ones(
        global_cell_ids.shape[0],
        1,
        device=device,
        dtype=dtype
    )

    roi_count_flat.index_add_(0, global_cell_ids, ones)

    roi_sum = roi_sum_flat.view(B, N, C)
    roi_count = roi_count_flat.view(B, N, 1).clamp(min=1.0)

    roi_mean = roi_sum / roi_count

    roi_feats = torch.empty(B, N, 0, C, device=device, dtype=dtype)
    roi_mask = torch.empty(B, N, 0, device=device, dtype=torch.bool)

    fused_feats = torch.cat([cell_feats, roi_mean], dim=-1)

    return cell_feats, roi_feats, roi_mask, roi_mean, fused_feats


if __name__ == "__main__":
    B, C, H, W = 2, 256, 20, 20
    feat_map = torch.randn(B, C, H, W)

    image_size = (640, 640)

    boxes_per_image = [
        torch.tensor([
            [100, 120, 180, 220],
            [300, 300, 420, 460],
        ], dtype=torch.float32),
        torch.tensor([
            [50, 60, 110, 140],
        ], dtype=torch.float32),
    ]

    cell_feats, roi_feats, roi_mask, fused_feats = build_cell_roi_tensor(
        feat_map,
        boxes_per_image,
        image_size=image_size,
        roi_out_size=7,
        max_rois_per_cell=12
    )

    print("cell_feats:", cell_feats.shape)  # [B, N, C]
    print("roi_feats :", roi_feats.shape)   # [B, N, K, C]
    print("roi_mask  :", roi_mask.shape)    # [B, N, K]
    print("fused_feats  :",fused_feats.shape)