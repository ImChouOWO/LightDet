# LightDet

LightDet 是一個 `Vision-Text to BBox model`，透過文字與圖像輸入定位畫面中的目標物件。

`多模態` `Transformer` `Token Fusion` <br>
`Decoder Only` `DETR Like`

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

## 建置與運行

LightDet 的基本流程如下：

```text
下載專案
→ 建立 Python 環境
→ 安裝相依套件
→ 準備資料集與標記
→ 設定 model.yaml 與 train.yaml
→ 執行 model.train()
→ 取得 checkpoint 與驗證指標
```

---

## 1. 下載專案

```bash
git clone https://github.com/ImChouOWO/LightDet.git
cd LightDet
```

---

## 2. 建立 Python 環境

建議環境：

```text
Python >= 3.10
PyTorch >= 2.0
```

Linux 或 macOS：

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
```

Windows：

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

---

## 3. 安裝相依套件

```bash
pip install torch torchvision
pip install numpy pyyaml tqdm transformers scipy pillow
```

---

## 4. 準備資料集

資料集目錄：

```text
/path/to/LightDet/datasets/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── .cache/
```

影像與標記檔名稱需要相互對應：

```text
images/train/000001.jpg
labels/train/000001.json

images/val/000101.jpg
labels/val/000101.json
```

支援的影像格式：

```text
.jpg
.jpeg
.png
.bmp
.tif
.tiff
.webp
```

每一張影像可以包含多個物件。資料載入器會根據物件屬性與增強文字建立 Query-level 樣本：

```text
影像 + 文字 Query + Query 對應的 GT bbox
```

例如：

```text
影像：紅色船與白色船
Query：紅色的船
GT：紅色船的 bbox
```

<details>
<summary><strong>標記檔格式</strong></summary>

每張影像對應一個 JSON 標記檔，最外層必須是陣列：

```json
[
  {
    "source_name": "000001.jpg",
    "class_id": 0,
    "bbox_xyxy": [120, 80, 420, 300],
    "attributes": {
      "main_colors": [
        "紅色"
      ]
    },
    "query_texts_aug": [
      "紅色的船",
      "含紅色的船",
      "船體是紅色的船"
    ]
  },
  {
    "source_name": "000001.jpg",
    "class_id": 0,
    "bbox_xyxy": [500, 100, 760, 340],
    "attributes": {
      "main_colors": [
        "白色"
      ]
    },
    "query_texts_aug": [
      "白色的船",
      "畫面中的白色船隻"
    ]
  }
]
```

欄位說明：

| 欄位 | 必要 | 說明 |
|---|---:|---|
| `source_name` | 是 | 對應的影像檔名。相同 JSON 內通常使用相同檔名 |
| `bbox_xyxy` | 是 | 原始影像像素座標，格式為 `[x1, y1, x2, y2]` |
| `class_id` | 否 | 類別編號，未設定時預設為 `0` |
| `attributes.main_colors` | 建議 | 物件主要顏色，用於建立顏色文字 Query |
| `query_texts_aug` | 建議 | 額外的文字 Query 描述 |

座標條件：

```text
x1 < x2
y1 < y2
```

資料載入器會依序執行：

```text
原始像素 xyxy
→ 依 image_size 縮放
→ 裁切至影像邊界
→ 正規化至 0～1
```

因此標記檔中的 `bbox_xyxy` 應保存原始影像的像素座標，不需要預先正規化。

</details>

### Query Budget Batch

當設定：

```yaml
data:
  query_budget: true

train:
  batch_size: 48
```

`batch_size=48` 代表每個 batch 的 Query 預算，而不是固定載入 48 張影像。

例如：

```text
影像 A：9 個 Query
影像 B：13 個 Query
影像 C：10 個 Query
影像 D：14 個 Query

總 Query：46
```

這四張影像可以組成同一個 batch。

---

## 5. 準備 BERT 與文字快取

LightDet 預設使用：

```text
hfl/chinese-macbert-base
```

本地模型目錄：

```text
/path/to/LightDet/units/model/bert/models--hfl--chinese-macbert-base/
```

文字特徵快取：

```text
/path/to/LightDet/units/model/cards/cache/bert_raw_cache.pt
```

當新增文字 Query 時，訓練程式會：

```text
讀取現有 cache
→ 檢查缺少的文字
→ 建立缺少的 BERT 特徵
→ 更新 bert_raw_cache.pt
```

模型設定：

```yaml
model:
  freeze_bert: true
  precomputed_bert_path: /path/to/LightDet/units/model/cards/cache/bert_raw_cache.pt
```

---

## 6. 設定負文字 Query Pool

預設位置：

```text
/path/to/LightDet/units/model/cards/cache/negative_query_pool.json
```

格式：

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

| 參數 | 說明 |
|---|---|
| `sampling_weight` | 控制該分類被抽中的相對機率 |
| `loss_weight` | 控制該負文字 Query 在 QFL 中的權重 |
| `queries` | 可抽樣的負文字描述 |

建議初始設定：

```text
hard loss_weight   = 2.0
severe loss_weight = 4.0
negative ratio     = 0.05
```

負文字 Query 沒有對應 GT：

```python
targets["boxes"] = []
```

因此：

```text
不計算 BBox Loss
不計算 GIoU Loss
所有 prediction 的 score target 為 0
```

---

## 7. 設定模型結構

設定檔：

```text
/path/to/LightDet/units/model/cards/config/model.yaml
```

範例：

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
  precomputed_bert_path: /path/to/LightDet/units/model/cards/cache/bert_raw_cache.pt
```

使用既有 checkpoint 時，下列結構必須與 checkpoint 一致：

```text
hidden_dim
num_heads
num_layers
image_grid_size
fusion_token_num
```

例如舊 checkpoint 使用：

```yaml
num_layers: 2
```

目前模型也必須保持兩層，否則使用 `strict=True` 載入時會出現 missing keys。

---

## 8. 設定訓練參數

設定檔：

```text
/path/to/LightDet/units/model/cards/config/train.yaml
```

主要設定：

```yaml
data:
  dataset_dir: /path/to/LightDet/datasets
  image_size: 512
  max_text_aug_per_image: 1

  cache_images: true
  image_cache_dir: /path/to/LightDet/datasets/.cache/images_512_uint8
  prebuild_image_cache: true
  cache_workers: 8

  prefetch_factor: 4
  pin_memory: true
  persistent_workers: true
  query_budget: true

  negative_query_path: /path/to/LightDet/units/model/cards/cache/negative_query_pool.json
  negative_sample_ratio: 0.05
  use_negative_queries_in_val: false

train:
  epochs: 300
  batch_size: 48
  warmup_epochs: 5
  num_workers: 16
  device: cuda:0
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
  save_dir: runs/train/lightdet_exp
  resume_path: null
  save_latest_interval: 1
  save_epoch_interval: 50
  emit_step_metrics: false
  log_interval: 50
```

`model.train()` 中明確指定的參數會覆蓋 `train.yaml` 中的對應設定。

---

## 9. 啟動訓練

在執行檔中建立模型：

```python
from train import LightDet

model = LightDet(
    model="/path/to/LightDet/units/model/cards/config/model.yaml"
)

model.train(
    cfg="/path/to/LightDet/units/model/cards/config/train.yaml",
    data="/path/to/LightDet/datasets",

    weights=None,
    resume=None,
    prefer_ema=True,

    epochs=300,
    imgsz=512,
    batch=48,
    device=0,
    workers=16,
    seed=47,
    deterministic=False,

    project="/path/to/LightDet/units/model/runs/train",
    name="lightdet_exp",
)
```

執行：

```bash
cd /path/to/LightDet/units/model
python3 train.py
```

### 訓練模式

三種訓練模式統一由 `weights` 與 `resume` 控制：

| 模式 | `weights` | `resume` | 用途 |
|---|---|---|---|
| 從頭訓練 | `None` | `None` | 不載入舊 checkpoint，重新建立完整訓練狀態 |
| Weights-only warm start | checkpoint 路徑 | `None` | 載入模型或 EMA 權重，但重置 optimizer、scheduler、epoch 與最佳指標 |
| 完整續訓 | `None` | checkpoint 路徑或 `True` | 恢復模型、EMA、optimizer、scheduler、GradScaler、epoch、最佳指標與 RNG state |

注意：

```text
weights 與 resume 不可同時使用
```

Weights-only warm start：

```python
model.train(
    cfg="/path/to/LightDet/units/model/cards/config/train.yaml",
    data="/path/to/LightDet/datasets",

    weights="/path/to/checkpoints/best_map50_95.pt",
    resume=None,
    prefer_ema=True,

    epochs=300,
    imgsz=512,
    batch=48,
    device=0,
    workers=16,

    project="/path/to/LightDet/units/model/runs/train",
    name="lightdet_warm_start",
)
```

完整續訓：

```python
model.train(
    cfg="/path/to/LightDet/units/model/cards/config/train.yaml",
    data="/path/to/LightDet/datasets",

    weights=None,
    resume="/path/to/LightDet/units/model/runs/train/lightdet_exp/latest.pt",

    epochs=300,
    imgsz=512,
    batch=48,
    device=0,
    workers=16,

    project="/path/to/LightDet/units/model/runs/train",
    name="lightdet_exp",
)
```

也可以使用：

```python
resume=True
```

此時會尋找：

```text
<project>/<name>/latest.pt
```

---

## 10. `model.train()` 參數

### 基本與 checkpoint 參數

| 參數 | 型別 | 用途 |
|---|---|---|
| `cfg` | `str` | `train.yaml` 路徑 |
| `data` | `str \| None` | 資料集根目錄，覆蓋 `data.dataset_dir` |
| `epochs` | `int \| None` | 訓練總 epoch |
| `imgsz` | `int \| None` | 輸入影像尺寸 |
| `batch` | `int \| None` | Batch 或 Query Budget 上限 |
| `device` | `int \| str \| None` | 運算裝置，例如 `0`、`1`、`"cuda:0"` 或 `"cpu"` |
| `workers` | `int \| None` | DataLoader worker 數量 |
| `seed` | `int \| None` | 隨機種子 |
| `deterministic` | `bool \| None` | 是否啟用 deterministic 訓練 |
| `project` | `str` | 實驗輸出根目錄 |
| `name` | `str` | 實驗名稱 |
| `weights` | `str \| None` | Weights-only warm start checkpoint |
| `resume` | `str \| bool \| None` | 完整續訓 checkpoint；`True` 代表尋找 `<project>/<name>/latest.pt` |
| `prefer_ema` | `bool` | Warm start 時優先載入 EMA 權重 |

### 資料載入參數

| 參數 | 型別 | 用途 |
|---|---|---|
| `cache_images` | `bool \| None` | 是否使用影像快取 |
| `image_cache_dir` | `str \| None` | 影像快取輸出目錄 |
| `prebuild_image_cache` | `bool \| None` | 是否在訓練前預先建立快取 |
| `prefetch_factor` | `int \| None` | 每個 worker 預先載入的 batch 數 |
| `pin_memory` | `bool \| None` | 是否啟用 pinned memory |
| `persistent_workers` | `bool \| None` | epoch 之間是否保留 DataLoader workers |
| `negative_query_path` | `str \| None` | 負文字 Query Pool 路徑 |
| `negative_sample_ratio` | `float \| None` | 負文字 Query 取樣比例，範圍為 `0～1` |
| `use_negative_queries_in_val` | `bool \| None` | 驗證階段是否加入負文字 Query |

### Optimizer 與執行參數

| 參數 | 型別 | 用途 |
|---|---|---|
| `lr` | `float \| None` | 同時設定 vision、transformer 與 head learning rate |
| `lr_vision` | `float \| None` | Vision backbone learning rate |
| `lr_text` | `float \| None` | Text encoder learning rate |
| `lr_transformer` | `float \| None` | Transformer learning rate |
| `lr_head` | `float \| None` | BBox 與 score head learning rate |
| `weight_decay` | `float \| None` | Optimizer weight decay |
| `amp_dtype` | `str \| None` | AMP dtype，例如 `"bf16"` 或 `"fp16"` |
| `compile_model` | `bool \| None` | 是否使用 `torch.compile` |
| `channels_last` | `bool \| None` | 是否使用 channels-last memory format |
| `startup_smoke_test` | `bool \| None` | 正式訓練前是否執行啟動測試 |
| `use_ema` | `bool \| None` | 是否啟用 EMA |
| `ema_decay` | `float \| None` | EMA decay，範圍為 `[0,1)` |

### 驗證與後處理參數

| 參數 | 型別 | 用途 |
|---|---|---|
| `score_thr` | `float \| None` | 驗證時的最低 confidence threshold |
| `top_k` | `int \| None` | 每個 Query 保留的最高分 prediction 數量 |
| `nms_iou_thr` | `float \| None` | NMS IoU threshold |
| `use_nms` | `bool \| None` | 是否在驗證階段使用 NMS |
| `use_topk_fallback` | `bool \| None` | 無 prediction 通過 threshold 時是否保留 top-k fallback |

### QFL、Ranking 與負樣本參數

| 參數 | 型別 | 用途 |
|---|---|---|
| `hard_negative_ratio` | `int \| None` | Score loss 中 hard negative 與 positive 的比例 |
| `positive_ratio` | `float \| None` | 額外正樣本的候選比例 |
| `max_positive_per_gt` | `int \| None` | 每個 GT 最多使用的正 prediction 數量 |
| `iou_pos_thr` | `float \| None` | 判定 quality positive 的最低 IoU |
| `quality_min` | `float \| None` | Quality target 下限 |
| `quality_max` | `float \| None` | Quality target 上限 |
| `qfl_beta` | `float \| None` | Quality Focal Loss 的 beta |
| `quality_warmup_epoch` | `int \| None` | Quality target 從常數轉為 IoU 的 warmup epoch |
| `lambda_rank` | `float \| None` | Ranking Loss 最大權重 |
| `rank_start_epoch` | `int \| None` | Ranking Loss 開始啟用的 epoch |
| `rank_warmup_epoch` | `int \| None` | Ranking Loss warmup 結束 epoch |
| `rank_alpha_min` | `float \| None` | Ranking warmup 的最小 alpha |
| `max_query_loss_weight` | `float \| None` | 負文字 Query loss weight 上限 |

未支援的參數會觸發：

```text
TypeError: Unsupported train arguments
```

---

## 11. Checkpoint 檢查

```python
checkpoint_info = model.inspect_checkpoint(
    "/path/to/checkpoints/latest.pt"
)

print(checkpoint_info)
```

檢查內容包含：

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

判斷方式：

| Checkpoint 狀態 | 建議載入方式 |
|---|---|
| 只有模型或 EMA 權重 | 使用 `weights=...` |
| 包含 optimizer、scheduler 與 epoch | 可使用 `resume=...` |
| 模型結構已修改 | 不建議完整 resume |
| Loss 或資料策略已修改 | 建議使用 weights-only warm start |

---

## 12. 訓練輸出

訓練過程範例：

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

| 欄位 | 說明 |
|---|---|
| `loss` | 加權後的總 Loss |
| `bbox` | BBox L1 Loss |
| `giou` | GIoU Loss |
| `score` | 整體 Quality Focal Loss |
| `rank` | 加入總 Loss 的 Ranking Loss 貢獻 |
| `raw` | 尚未乘上權重的 Ranking Loss |
| `ra` | Ranking warmup alpha |
| `lrk` | 目前實際生效的 Ranking 權重 |
| `txtneg` | 負文字 Query 的 QFL 監控值 |
| `nq` | 當前 batch 中的負文字 Query 數量 |

```text
nq=0
```

只代表目前 batch 沒有抽到負文字 Query，不代表整個訓練未使用負文字資料。

驗證輸出包含：

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

當設定：

```yaml
eval:
  best_metric: map50_95
```

最佳 checkpoint 會依照 `mAP50-95` 選擇。

---

## 13. 輸出檔案

```text
/path/to/LightDet/units/model/runs/train/<experiment>/
├── latest.pt
├── best_map50_95.pt
├── epoch_050.pt
├── metrics_epoch.jsonl
├── metrics_step.jsonl
├── metrics_epoch.csv
└── latest_metrics.json
```

| 檔案 | 說明 |
|---|---|
| `latest.pt` | 最新 epoch 的完整 checkpoint |
| `best_map50_95.pt` | 驗證集 mAP50-95 最佳 checkpoint |
| `epoch_XXX.pt` | 固定週期保存的 checkpoint |
| `metrics_epoch.jsonl` | Epoch-level 指標 |
| `metrics_step.jsonl` | Step-level 指標 |
| `metrics_epoch.csv` | CSV 格式訓練指標 |
| `latest_metrics.json` | 最新一次訓練與驗證結果 |

---

## 14. EMA

設定：

```yaml
train:
  use_ema: true
  ema_decay: 0.999
```

Checkpoint 會同時保存：

```text
model
ema
```

Weights-only warm start 時：

```python
prefer_ema=True
```

會優先載入 EMA 權重。

---

## 20. 常見問題

### CUDA 無法使用

檢查：

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

如果輸出：

```text
False
```

請確認目前 Python 環境安裝的 PyTorch 是否支援所使用的運算裝置。

---

### GPU 編號錯誤

```python
device=1
```

代表使用第 2 張 GPU。

只有一張 GPU 時應使用：

```python
device=0
```

或：

```python
device="cuda:0"
```

---

### BERT Cache 缺少文字

加入新的文字 Query 後，訓練程式會自動補建缺少的文字特徵。

若快取格式已改變，可以刪除：

```text
/path/to/LightDet/units/model/cards/cache/bert_raw_cache.pt
```

再重新啟動訓練。

刪除後會重新建立全部文字特徵。

---

### Transformer Missing Keys

錯誤：

```text
Missing key(s):
transformer.transformer.layers.2.*
```

通常代表目前模型層數與 checkpoint 不一致。

檢查：

```yaml
model:
  num_layers: 2
```

模型結構必須與 checkpoint 建立時的設定一致。

---

### `enable_nested_tensor` 警告

```text
enable_nested_tensor is True, but self.use_nested_tensor is False
because encoder_layer.norm_first was True
```

這是 PyTorch Transformer 的效能提示，不會中斷訓練，也不代表模型結構錯誤。

---

### `weights` 與 `resume` 同時設定

以下寫法不支援：

```python
model.train(
    weights="/path/to/best.pt",
    resume="/path/to/latest.pt",
)
```

請根據目的選擇其中一種：

```text
weights：只載入模型權重並開始新實驗
resume：恢復完整訓練狀態
```

---

### `resume=True` 找不到 checkpoint

`resume=True` 會尋找：

```text
<project>/<name>/latest.pt
```

需要確認：

```python
project="/path/to/LightDet/units/model/runs/train"
name="lightdet_exp"
```

與原實驗輸出目錄一致。

---

### `txtneg` 數值很低

例如：

```text
txtneg=0.0001
```

通常代表模型已將負文字 Query 的 confidence 壓低。

仍需搭配下列指標判斷：

```text
Recall
Precision
FP
mAP50
mAP50-95
```

避免負樣本抑制過強導致正樣本 Recall 下降。

---

## 快速啟動

```bash
cd /path/to/LightDet

python3 -m venv venv
source venv/bin/activate

pip install torch torchvision
pip install numpy pyyaml tqdm transformers scipy pillow

cd units/model
python3 train.py
```

訓練結果會輸出至：

```text
/path/to/LightDet/units/model/runs/train/<experiment>/
```
