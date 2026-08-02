# LightDet 訓練設定說明

LightDet 是一個文字條件式物件偵測模型。模型接收整張影像與文字查詢，輸出符合查詢描述的邊界框與信心分數。

```text
image + query_text -> bbox + score
```

模型目前不進行多類別分類。邊界框採 normalized `xyxy` 格式：

```text
x1, y1, x2, y2 ∈ [0, 1]
x1 <= x2
y1 <= y2
```

## 執行方式

```bash
python3 train.py
```

設定檔主要分為：

```text
model.yaml    模型結構
config.yaml   資料、訓練、最佳化、Loss、評估與紀錄
```

## Training Resource

| 項目 | 設定 |
|---|---:|
| Batch size | 48 |
| Image size | 1024 × 1024 |
| GPU | NVIDIA RTX 5000 Ada × 1 |
| Training VRAM | 約 20–27 GB |
| Validation VRAM | 約 16–20 GB |

實際記憶體使用量會受到 CUDA、PyTorch、AMP、DataLoader、模型版本與驗證設定影響。

# model.yaml

```yaml
model:
  hidden_dim: 256

  backbone:
    in_channels: 3
    base_channels:
      - 64
      - 128
      - 256
      - 512
      - 1024
    base_depths:
      - 2
      - 3
      - 2
    width_multiple: 0.75
    depth_multiple: 0.67
    max_channels: 1024
    channel_divisor: 8

  fpn:
    out_channels: 256
    norm_layer: null

  image_projector:
    in_channels: 256
    out_channels: 256
    layer_num: 3
    expand_ratio: 2.0
    level_names:
      - c3
      - c4
      - c5
    token_grids:
      - [24, 24]
      - [12, 12]
      - [10, 10]

  freeze_img_projection: false

  num_object_queries: 100
  query_group_init_std: 0.02
  fusion_token_num: 16

  num_heads: 8
  num_layers: 2
  mlp_ratio: 3.5
  dropout: 0.1

  staged_query_refinement: true
  score_num_heads: 8
  score_num_layers: 2
  score_mlp_ratio: 3.0
  score_dropout: 0.1
  score_bbox_conditioning: true
  score_bbox_detach: true
  score_fusion: geometric_mean
  score_fusion_eps: 0.000001

  text_max_length: 256
  freeze_bert: true
  precomputed_bert_path: units/model/cards/cache/bert_raw_cache.pt

  use_auxiliary_head: true
  auxiliary_in_eval: false
  initialize_aux_from_main: true
```

## Model Parameters

### Model Base

| 參數 | 功用 |
|---|---|
| `hidden_dim` | 模型內部影像、文字、Query 與 Transformer token 的共同維度。 |

### Backbone

| 參數 | 功用 |
|---|---|
| `in_channels` | 輸入影像通道數，RGB 為 3。 |
| `base_channels` | Backbone 各階段的基礎通道數。 |
| `base_depths` | Backbone 各階段的基礎 block 數量。 |
| `width_multiple` | 統一縮放各階段通道寬度。 |
| `depth_multiple` | 統一縮放各階段深度。 |
| `max_channels` | 限制 Backbone 最大通道數。 |
| `channel_divisor` | 將通道數調整為指定倍數，便於硬體計算。 |

### FPN

| 參數 | 功用 |
|---|---|
| `out_channels` | 將 C3、C4、C5 統一投影到相同通道數。 |
| `norm_layer` | FPN 使用的正規化層；`null` 表示不額外指定。 |

### Image Projector

| 參數 | 功用 |
|---|---|
| `in_channels` | 投影層接收的 FPN 特徵通道數。 |
| `out_channels` | 投影後輸出的特徵維度。 |
| `layer_num` | 每個尺度使用的投影 block 數量。 |
| `expand_ratio` | 中間通道擴張倍率，計算方式為 `in_channels × expand_ratio`。 |
| `level_names` | 使用的 FPN 特徵層名稱。 |
| `token_grids` | 各尺度壓縮後的 token 網格大小。 |
| `freeze_img_projection` | 是否凍結影像投影層參數。 |

目前 visual token 數量為：

```text
24 × 24 + 12 × 12 + 10 × 10 = 820
```

### Localization Transformer

| 參數 | 功用 |
|---|---|
| `num_object_queries` | 可同時產生的物件候選 Query 數量。 |
| `query_group_init_std` | Object Query 初始化時的標準差。 |
| `fusion_token_num` | 用於整合全局影像與文字資訊的可學習 token 數量。 |
| `num_heads` | 定位 Transformer 的注意力 head 數。 |
| `num_layers` | 定位 Transformer 層數。 |
| `mlp_ratio` | Feed-forward 中間維度的擴張倍率。 |
| `dropout` | 定位 Transformer 的 dropout 比例。 |

### Score and Text Refinement

| 參數 | 功用 |
|---|---|
| `staged_query_refinement` | 是否啟用第二階段 Query refinement。 |
| `score_num_heads` | 第二階段 Transformer 的注意力 head 數。 |
| `score_num_layers` | 第二階段 Transformer 層數。 |
| `score_mlp_ratio` | 第二階段 MLP 擴張倍率。 |
| `score_dropout` | 第二階段 dropout 比例。 |
| `score_bbox_conditioning` | 是否將 bbox 資訊加入 score refinement。 |
| `score_bbox_detach` | 是否阻止 score 分支梯度回傳至 bbox 分支。 |
| `score_fusion` | Quality score 與文字相似度的融合方式。 |
| `score_fusion_eps` | 分數融合時避免數值不穩定的小常數。 |

### Text Branch

| 參數 | 功用 |
|---|---|
| `text_max_length` | 文字 tokenizer 的最大 token 長度。 |
| `freeze_bert` | 是否凍結 BERT 參數。 |
| `precomputed_bert_path` | 預先計算的 BERT 文字特徵快取位置。 |

### H-DETR Auxiliary Branch

| 參數 | 功用 |
|---|---|
| `use_auxiliary_head` | 是否啟用輔助預測分支。 |
| `auxiliary_in_eval` | 評估時是否保留輔助分支輸出。 |
| `initialize_aux_from_main` | 是否使用主分支權重初始化輔助分支。 |

# config.yaml

## Data

| 參數 | 功用 |
|---|---|
| `dataset_dir` | 資料集根目錄。 |
| `image_size` | 輸入影像尺寸。 |
| `max_text_aug_per_image` | 每張影像最多使用的文字增強數量。 |
| `cache_images` | 是否快取預處理後的影像。 |
| `image_cache_dir` | 影像快取路徑。 |
| `prebuild_image_cache` | 是否在訓練前建立完整快取。 |
| `cache_workers` | 建立快取時使用的 worker 數量。 |
| `prefetch_factor` | 每個 DataLoader worker 預先載入的 batch 數。 |
| `pin_memory` | 是否使用鎖頁記憶體加速 CPU 至 GPU 傳輸。 |
| `persistent_workers` | 每個 epoch 後是否保留 DataLoader workers。 |
| `query_budget` | 是否限制每張影像使用的 Query 數量。 |
| `negative_query_path` | 負向文字查詢池路徑。 |
| `negative_sample_ratio` | 負向查詢的取樣比例。 |
| `negative_phrase_max_per_image` | 每張影像最多加入的負向描述數量。 |
| `use_negative_queries_in_val` | 驗證階段是否加入負向文字查詢。 |

## Train

| 參數 | 功用 |
|---|---|
| `epochs` | 總訓練 epoch 數。 |
| `batch_size` | 每次迭代的樣本數。 |
| `warmup_epochs` | 學習率暖身週期。 |
| `num_workers` | DataLoader worker 數量。 |
| `device` | 使用的 GPU 編號。 |
| `seed` | 隨機種子。 |
| `deterministic` | 是否使用可重現但較慢的確定性運算。 |
| `use_amp` | 是否使用混合精度訓練。 |
| `amp_dtype` | AMP 使用的資料型別。 |
| `allow_tf32` | 是否允許 NVIDIA GPU 使用 TF32。 |
| `matmul_precision` | PyTorch 矩陣乘法精度策略。 |
| `compile` | 是否使用 `torch.compile`。 |
| `use_ema` | 是否使用模型權重指數移動平均。 |
| `ema_decay` | EMA 衰減係數。 |
| `grad_clip_norm` | 梯度裁切上限。 |

## Optimizer

`components` 格式：

```text
[排程模式, 最大學習率, 最小學習率, 起始比例, 結束比例]
```

| 參數 | 功用 |
|---|---|
| `vision` | Backbone、FPN 與影像投影層的學習率排程。 |
| `text` | 文字分支的學習率排程。 |
| `transformer` | Transformer 的學習率排程。 |
| `head` | 預測頭的學習率排程。 |
| `weight_decay` | AdamW 權重衰減。 |
| `max_warmup_steps` | Warmup 最大 step 數。 |
| `fused` | 是否使用 fused optimizer。 |

## Loss

模型主要使用 Hungarian matching，並結合以下訓練目標：

```text
BBox L1
GIoU
IoU-aware score
Text alignment
Duplicate suppression
Hard-negative mining
Auxiliary loss
```

| 區塊 | 功用 |
|---|---|
| `matcher` | 控制 Hungarian matching 的 bbox、GIoU、score 與文字對齊成本。 |
| `weight` | 控制各 loss 的權重及動態排程。 |
| `classification` | 控制 IoU-aware classification 與負樣本忽略門檻。 |
| `quality` | 控制 Quality Focal Loss 與品質標籤範圍。 |
| `text_alignment` | 控制影像 Query 與文字特徵對齊。 |
| `matcher_schedule` | 控制 score matching 成本逐步啟用。 |
| `score_sampling` | 控制額外正樣本與困難負樣本取樣。 |
| `hybrid` | 控制主分支與輔助分支 loss。 |
| `text_negative` | 控制負向文字樣本的訓練方式。 |
| `duplicate_suppression` | 降低多個 Query 對同一目標的重複預測。 |
| `hard_negative` | 強化高分但低 IoU 的錯誤候選框。 |
| `ranking` | 控制額外的候選框排序 loss。 |
| `query_refinement` | 控制第二階段 Query refinement 的限制。 |
| `pos_weight` | 控制 BCE 正樣本權重。 |

## Evaluation

| 參數 | 功用 |
|---|---|
| `val_loss_interval` | 完整計算 validation loss 的間隔。 |
| `eval_interval` | 執行偵測評估的間隔。 |
| `max_val_batches` | 驗證最多使用的 batch 數；`null` 表示全部。 |
| `score_thr` | 評估時的最低 score 門檻。 |
| `top_k` | 每筆樣本最多保留的預測框數。 |
| `nms_iou_thr` | NMS 的 IoU 門檻。 |
| `use_nms` | 是否啟用 NMS。 |
| `compute_raw_oracle` | 是否計算未排序候選框的理論召回能力。 |
| `raw_oracle_iou_thresholds` | Raw Oracle Recall 使用的 IoU 門檻。 |
| `best_metric` | 選擇最佳 checkpoint 的主要指標。 |

主要評估指標：

| 指標 | 說明 |
|---|---|
| `mAP50` | IoU=0.5 時的 Average Precision。 |
| `mAP50-95` | IoU 0.5 至 0.95 的平均 AP。 |
| `Precision` | 預測框中正確命中的比例。 |
| `Recall` | GT 目標中被模型找出的比例。 |
| `Recall@K` | 僅檢查分數排名前 K 個預測時的召回率。 |
| `Raw Oracle Recall` | 不考慮最終排序時，所有候選框能達到的理論召回率。 |

## Logging

| 參數 | 功用 |
|---|---|
| `save_dir` | 權重與訓練紀錄輸出位置。 |
| `weights_path` | 初始化模型權重路徑。 |
| `resume_path` | 接續訓練使用的 checkpoint。 |
| `prefer_ema` | 驗證與保存最佳模型時優先使用 EMA。 |
| `save_latest_interval` | 儲存 latest checkpoint 的間隔。 |
| `save_epoch_interval` | 額外保存 epoch checkpoint 的間隔。 |
| `emit_step_metrics` | 是否記錄 step-level 指標。 |
| `log_interval` | 輸出訓練資訊的 step 間隔。 |

# Dataset Structure

```text
datasets/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

每筆標記需包含文字查詢與對應的 `bbox_xyxy`。訓練前需確認 bbox 已正規化，或由 DataLoader 轉換為 normalized `xyxy`。

# Notes

- `score_logit` 應直接輸入 `BCEWithLogitsLoss`，不要預先執行 sigmoid。
- Sigmoid 僅在推論與評估階段使用。
- 評估為文字條件式 binary detection，不是多類別 COCO classification。
- `hidden_dim` 必須能被 Transformer 的 `num_heads` 與 `score_num_heads` 整除。
- 修改 `token_grids`、Transformer 層數或 batch size 時，需重新確認 VRAM 使用量。
