# LightDet

LightDet 是一個以 **影像 + 中文文字 Query** 為輸入的文字條件式物件偵測模型。

本專案目前以「船舶」為主要偵測目標。模型會根據文字描述，例如：

```text
紅色的船
白色船隻
船體是藍色的船
```

在影像中預測對應的 bounding boxes 與 confidence。

專案同時支援負文字 Query，例如：

```text
紅色浮標
白色汽車
畫面中的人
```

負文字 Query 不提供 GT bbox，模型需將其所有 prediction confidence 壓低，以強化文字與影像之間的語意對齊。

---

## 主要功能

* 影像與中文文字 Query 多模態融合
* Query-conditioned bounding box prediction
* One-to-Many Matcher
* L1 Bounding Box Loss
* Generalized IoU Loss
* Quality Focal Loss
* 文字負樣本加權 QFL
* Ranking Loss
* EMA 模型權重
* BF16 / FP16 AMP
* TF32
* Query Budget Batch Sampler
* Image Cache
* BERT Raw Feature Cache
* Weights-only warm start
* 完整 checkpoint resume
* mAP50 / mAP50-95 / Precision / Recall 評估
* JSONL / CSV 訓練紀錄

---

## 專案結構

```text
LightDet/
├── datasets/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── .cache/
│       └── images_512_uint8/
│
└── units/
    ├── model/
    │   ├── bert/
    │   │   └── models--hfl--chinese-macbert-base/
    │   ├── cards/
    │   │   ├── cache/
    │   │   │   ├── bert_raw_cache.pt
    │   │   │   └── negative_query_pool.json
    │   │   ├── config/
    │   │   │   ├── model.yaml
    │   │   │   └── train.yaml
    │   │   └── loss.py
    │   ├── pipeline/
    │   │   └── data.py
    │   ├── resnet/
    │   │   └── checkpoints/
    │   ├── runs/
    │   │   └── train/
    │   └── train.py
    │
    └── tool/
        └── card.py
```

---

## 模型輸入與輸出

每一筆 Query-level 樣本包含：

```text
影像 + 文字 Query + 該文字對應的 GT bbox
```

例如：

```text
影像：紅色船與白色船
Query：紅色的船
GT：紅色船的 bbox
```

模型輸出：

```python
{
    "bbox": Tensor[Q, N, 4],
    "score_logit": Tensor[Q, N, 1]
}
```

其中：

* `Q`：batch 內文字 Query 數量
* `N`：每個 Query 的 prediction 數量
* bbox 格式：normalized `xyxy`
* score：尚未經 sigmoid 的 logit

---

## 文字樣本設計

### 正文字 Query

正文字 Query 由標記檔中的船舶屬性與增強描述產生，例如：

```text
紅色的船
含紅色的船
紅色船隻
船體是紅色的船
```

每一條正文字 Query 都會對應該文字條件下的 GT bbox。

### 負文字 Query

負文字 Query 從 JSON pool 中抽樣，例如：

```text
紅色浮標
大型海上平台
白色汽車
穿紅色衣服的人
```

負文字 Query 的 GT 為空集合：

```python
targets["boxes"] = []
```

因此：

* 不計算 BBox Loss
* 不計算 GIoU Loss
* 所有 prediction 的 QFL target 都為 0

---

## 負文字 Query Pool

預設路徑：

```text
units/model/cards/cache/negative_query_pool.json
```

格式範例：

```json
{
  "version": 1,
  "target": "船",
  "categories": {
    "hard": {
      "sampling_weight": 1.0,
      "loss_weight": 2.0,
      "queries": [
        "紅色浮標",
        "大型海上平台",
        "漂浮中的塑膠桶"
      ]
    },
    "severe": {
      "sampling_weight": 1.0,
      "loss_weight": 4.0,
      "queries": [
        "白色汽車",
        "畫面中的人",
        "天空中的飛機"
      ]
    }
  }
}
```

參數說明：

* `sampling_weight`：控制該類別被抽中的機率
* `loss_weight`：控制該 Query 在 QFL 中的加權倍率
* `queries`：候選負文字描述

建議初始設定：

```text
hard loss_weight   = 2.0
severe loss_weight = 4.0
negative ratio     = 0.05
```

---

## Loss 結構

總 Loss：

[
L_{total}
=========

\lambda_{bbox}L_{bbox}
+
\lambda_{giou}L_{GIoU}
+
\lambda_{score}L_{QFL}
+
\lambda_{rank}\alpha_{rank}L_{Ranking}
]

### BBox Loss

只對 Matcher 配對成功的 prediction 計算：

[
L_{bbox}
========

\frac{1}{N_{pos}}
\sum
\left|
B_{pred}-B_{gt}
\right|
]

### GIoU Loss

[
L_{GIoU}
========

\frac{1}{N_{pos}}
\sum
\left(
1-GIoU(B_{pred},B_{gt})
\right)
]

### Quality Focal Loss

[
L_{QFL}
=======

BCE(z,y)\cdot|y-\sigma(z)|^\beta
]

目前：

```text
beta = 2.0
```

正樣本 target 由 1 漸進轉為 IoU：

[
y_{pos}
=======

(1-\alpha_q)+\alpha_q\cdot IoU
]

一般負 prediction：

[
y=0
]

負文字 Query：

[
y=0
]

並乘上 query loss weight：

[
L_{text-neg}
============

w_q\cdot QFL(z,0)
]

`loss_text_negative` 已包含在整體 QFL 中，只作為監控指標，不會再次加入總 Loss。

### Ranking Loss

Ranking Loss 用於：

1. 讓高 IoU 正樣本 score 高於低 IoU 正樣本
2. 讓正樣本 score 高於高分負樣本

[
L_{rank}
========

\max(0,m-(s_{positive}-s_{negative}))
]

Ranking 透過 warmup 排程逐步啟用。

---

## Matcher

目前 Matcher 主要依據：

[
C_{ij}
======

## \lambda_{bbox}C_{L1}

\lambda_{giou}GIoU
]

建議設定：

```yaml
loss:
  matcher:
    cost_bbox: 5.0
    cost_giou: 2.0
    cost_score: 0.0
```

`cost_score: 0.0` 表示 assignment 不直接依賴 confidence，避免訓練前期 score 不穩定造成錯誤配對。

---

## 安裝環境

建議：

```text
Python >= 3.10
PyTorch >= 2.0
CUDA GPU
```

安裝基本套件：

```bash
pip install torch torchvision
pip install numpy pyyaml tqdm transformers scipy pillow
```

---

## model.yaml

```yaml
model:
  img_in_channels: 1024
  hidden_dim: 512
  num_heads: 8
  num_layers: 2
  mlp_ratio: 3.5
  image_grid_size: 10
  text_max_length: 32
  fusion_token_num: 16
  dropout: 0.1
  freeze_bert: true
  precomputed_bert_path: units/model/cards/cache/bert_raw_cache.pt
```

使用既有 checkpoint 時，以下結構必須一致：

```text
hidden_dim
num_heads
num_layers
image_grid_size
fusion_token_num
```

例如舊 checkpoint 使用 `num_layers: 2`，目前模型也必須保持 2 層，否則 `strict=True` 載入會出現 missing keys。

---

## train.yaml

```yaml
data:
  dataset_dir: /home/soic/Desktop/LightDet/datasets
  image_size: 512
  max_text_aug_per_image: 1
  cache_images: true
  image_cache_dir: /home/soic/Desktop/LightDet/datasets/.cache/images_512_uint8
  prebuild_image_cache: true
  prefetch_factor: 4
  pin_memory: true
  persistent_workers: true
  query_budget: true
  cache_workers: 8
  negative_query_path: /home/soic/Desktop/LightDet/units/model/cards/cache/negative_query_pool.json
  negative_sample_ratio: 0.05
  use_negative_queries_in_val: false

train:
  epochs: 300
  batch_size: 48
  warmup_epochs: 5
  num_workers: 16
  device: cuda:1
  seed: 47
  deterministic: false
  use_amp: true
  amp_dtype: bf16
  use_ema: true
  ema_decay: 0.999
  grad_clip_norm: 1.0
  allow_tf32: true
  matmul_precision: high
  channels_last: false
  compile: false
  startup_smoke_test: true

optim:
  lr_vision: 0.0001
  lr_text: 0.00001
  lr_transformer: 0.0001
  lr_head: 0.0001
  weight_decay: 0.0001
  min_lr_ratio: 0.05
  max_warmup_steps: 3000
  fused: true

loss:
  matcher:
    cost_bbox: 5.0
    cost_giou: 2.0
    cost_score: 0.0

  score_sampling:
    hard_negative_ratio: 5
    positive_ratio: 0.05
    max_positive_per_gt: 2
    aux_positive_label: 0.7
    expand_cost_bbox: 5.0
    expand_cost_giou: 2.0

  quality:
    iou_pos_thr: 0.15
    quality_min: 0.25
    quality_max: 1.0
    qfl_beta: 2.0
    quality_warmup_epoch: 20

  ranking:
    lambda_rank: 0.10
    rank_margin: 0.1
    rank_min_quality_gap: 0.1
    rank_max_pairs: 512
    rank_start_epoch: 15
    rank_warmup_epoch: 30
    rank_alpha_min: 0.0

  text_negative:
    max_query_loss_weight: 10.0

  weight:
    dynamic: true
    bbox: 5.0
    giou: 2.0
    score: 2.0
    bbox_start: 5.0
    bbox_end: 3.0
    bbox_decay_until: 0.5
    score_start: 2.0
    score_end: 4.0
    score_warm_until: 0.4

  pos_weight:
    value: 1.0

eval:
  val_loss_interval: 5
  eval_interval: 1
  max_val_batches: null
  score_thr: 0.001
  top_k: 20
  nms_iou_thr: 0.5
  use_nms: true
  best_metric: map50_95
  use_topk_fallback: false

log:
  save_dir: runs/train/lightdet_neg_pool
  resume_path: null
  save_latest_interval: 1
  save_epoch_interval: 50
  emit_step_metrics: false
  log_interval: 50
  progress_leave: true
  progress_mininterval: 0.5
```

---

## 啟動訓練

```bash
cd /home/soic/Desktop/LightDet/units/model
python3 train.py
```

---

## 從頭訓練

```python
model.train(
    cfg="/home/soic/Desktop/LightDet/units/model/cards/config/train.yaml",
    data="/home/soic/Desktop/LightDet/datasets",

    weights=None,
    resume=None,

    epochs=300,
    imgsz=512,
    batch=48,
    device=1,
    workers=16,

    project="runs/train",
    name="lightdet_scratch",
)
```

---

## Weights-only Warm Start

適合：

* 修改 Loss
* 修改負文字 Query
* 修改 Ranking 排程
* 修改資料分布
* 希望沿用舊模型能力，但重置 optimizer 與 scheduler

```python
model.train(
    cfg="/home/soic/Desktop/LightDet/units/model/cards/config/train.yaml",
    data="/home/soic/Desktop/LightDet/datasets",

    weights=(
        "/home/soic/Desktop/LightDet/units/model/runs/train/"
        "lightdet_rank_smooth_010/best_map50_95.pt"
    ),
    resume=None,
    prefer_ema=True,

    epochs=300,
    imgsz=512,
    batch=48,
    device=1,
    workers=16,

    project="runs/train",
    name="lightdet_neg_pool",
)
```

此模式只恢復模型或 EMA 權重。

以下項目會重置：

```text
optimizer
scheduler
GradScaler
epoch
best metric
RNG state
```

---

## 完整 Resume

適用於同一實驗中斷後繼續訓練。

```python
model.train(
    cfg="/home/soic/Desktop/LightDet/units/model/cards/config/train.yaml",
    data="/home/soic/Desktop/LightDet/datasets",

    weights=None,
    resume=(
        "/home/soic/Desktop/LightDet/units/model/runs/train/"
        "lightdet_neg_pool/latest.pt"
    ),

    epochs=300,
    imgsz=512,
    batch=48,
    device=1,
    workers=16,

    project="runs/train",
    name="lightdet_neg_pool",
)
```

也可以使用：

```python
resume=True
```

此時自動尋找：

```text
runs/train/<name>/latest.pt
```

完整 resume 會恢復：

```text
model
EMA
optimizer
scheduler
GradScaler
epoch
best metric
RNG state
```

---

## Checkpoint 檢查

```python
model.inspect_checkpoint(
    "/home/soic/Desktop/LightDet/units/model/runs/train/"
    "lightdet_neg_pool/latest.pt"
)
```

會顯示：

```text
model
ema
optimizer
scaler
scheduler
epoch
rng_state
weights_only
full_resume
```

---

## BERT Cache

模型使用凍結的中文 MacBERT：

```text
hfl/chinese-macbert-base
```

BERT cache 預設路徑：

```text
units/model/cards/cache/bert_raw_cache.pt
```

負文字池加入新文字時，訓練程式會：

```text
讀取現有 cache
→ 檢查缺少的文字
→ 只補建 missing texts
→ 更新 bert_raw_cache.pt
```

---

## Query Budget Batch

當：

```yaml
query_budget: true
batch_size: 48
```

`batch_size=48` 代表每個 batch 的 Query 預算，不是固定 48 張影像。

例如：

```text
影像 A：9 個 Query
影像 B：13 個 Query
影像 C：10 個 Query
影像 D：14 個 Query
```

總 Query：

```text
9 + 13 + 10 + 14 = 46
```

這 4 張影像可以組成一個 batch。

---

## 訓練輸出

範例：

```text
Epoch 8/300 [Train]:
loss=0.7469
bbox=0.0142
giou=0.2635
score=0.0743
rank=0.0000
txtneg=0.0001
nq=3
```

欄位說明：

* `loss`：總 Loss
* `bbox`：BBox L1 Loss
* `giou`：GIoU Loss
* `score`：整體 QFL
* `rank`：加入總 Loss 的 Ranking 貢獻
* `raw`：未乘權重的 Ranking Loss
* `ra`：Ranking alpha
* `lrk`：有效 Ranking 權重
* `txtneg`：文字負樣本 QFL 統計值
* `nq`：目前 batch 中負文字 Query 數量

例如：

```text
nq=3
```

代表目前 batch 中有 3 條負文字 Query。

---

## 驗證

驗證階段同樣輸入：

```text
影像 + 文字 Query
```

目前輸出：

```text
mAP50
mAP50-95
Precision
Recall
TP
FP
GT
Pred
```

當：

```yaml
use_negative_queries_in_val: false
```

一般驗證集不加入負文字 Query，主要評估正文字 Query 下的定位能力。

後續可額外加入：

```text
negative_score_mean
negative_score_max
negative_fp_rate@0.1
negative_fp_rate@0.25
```

---

## 訓練輸出檔案

```text
runs/train/<experiment>/
├── latest.pt
├── best_map50_95.pt
├── epoch_050.pt
├── metrics_epoch.jsonl
├── metrics_step.jsonl
├── metrics_epoch.csv
└── latest_metrics.json
```

* `latest.pt`：最新 checkpoint
* `best_map50_95.pt`：最佳 mAP50-95 checkpoint
* `epoch_XXX.pt`：固定週期 checkpoint
* `metrics_epoch.jsonl`：epoch-level 指標
* `metrics_step.jsonl`：step-level 指標
* `metrics_epoch.csv`：CSV 格式指標
* `latest_metrics.json`：最新指標

---

## EMA

設定：

```yaml
use_ema: true
ema_decay: 0.999
```

訓練更新原始模型，驗證預設使用 EMA 模型。

Checkpoint 同時保存：

```text
model
ema
```

weights-only warm start 時：

```python
prefer_ema=True
```

會優先載入 EMA 權重。

---

## 常見問題

### `enable_nested_tensor` 警告

```text
enable_nested_tensor is True, but self.use_nested_tensor is False
because encoder_layer.norm_first was True
```

這是 PyTorch Transformer 的效能提示，不會中斷訓練。

### Missing Transformer Layer Keys

```text
Missing key(s):
transformer.transformer.layers.2.*
```

表示目前模型使用 3 層 Transformer，但 checkpoint 只有 2 層。

修正：

```yaml
num_layers: 2
```

### `nq=0`

代表目前 batch 沒有排入負文字 Query，不表示整體訓練未使用負文字。

### `txtneg` 很低

例如：

```text
txtneg=0.0001
```

通常代表模型已將該 batch 的負文字 Query confidence 壓低。

---

## 建議實驗流程

```text
1. 使用舊最佳 EMA checkpoint warm start
2. hard/severe 權重先設為 2/4
3. negative_sample_ratio 設為 0.05
4. 觀察 Recall 是否下降
5. 觀察 txtneg、score 與 nq
6. Ranking 經 warmup 後逐步啟用
7. 若負文字 confidence 仍偏高，再調整為 3/6
8. 使用 mAP50-95 選擇最佳 checkpoint
```

---

## 目前研究方向

* 文字條件式船舶偵測
* 中文文字與海事影像對齊
* 負文字 Query 抑制
* Confidence 與定位品質一致性
* 輕量化多模態偵測
* Edge AI 推論部署
* 船舶監控與航行影像應用
