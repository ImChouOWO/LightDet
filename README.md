# LightDet

![Framework](https://img.shields.io/badge/framework-PyTorch-EE4C2C)
![Architecture](https://img.shields.io/badge/architecture-DETR%20%2B%20FPN%20%2B%20Transformer-8A2BE2)
![Modality](https://img.shields.io/badge/input-Text%20%2B%20Image-2F80ED)
![Task](https://img.shields.io/badge/task-Multimodal%20Object%20Detection-4CAF50)
![Precision](https://img.shields.io/badge/precision-BF16%20AMP-FF9800)
![GPU](https://img.shields.io/badge/GPU-NVIDIA-76B900)
![Status](https://img.shields.io/badge/status-active%20research-yellow)
---
![Modality](https://img.shields.io/badge/modality-Vision--Language-2F80ED)
![Task](https://img.shields.io/badge/task-Object%20Detection-4CAF50)
![Architecture](https://img.shields.io/badge/architecture-DETR--like-8A2BE2)
![Decoder](https://img.shields.io/badge/decoder-Decoder--only-FF9800)
![Training](https://img.shields.io/badge/training-H--DETR-E91E63)
![Alignment](https://img.shields.io/badge/alignment-ODVG%20Token%20Alignment-00A6A6)

LightDet 是一個以 DETR 為基礎的文字引導物件偵測模型（text-guided object detection）。模型接收影像與文字描述，輸出與文字語意對應的邊界框與信心分數。

```text
Image + Text Query -> Bounding Boxes + Scores
```

目前版本採用 Decoder-only Object Query、兩階段 Query Refinement、H-DETR Main/Auxiliary 分支，以及 ODVG 風格的 token-level text alignment。



<details>
<summary><em>當前限制 / Current Limitations</em></summary>

<br>

受限於現有硬體與運算資源，本專案目前尚未針對模型寬度、深度、Transformer 容量、訓練資料規模及訓練週期進行完整的規模化實驗。

目前模型僅在既有測試資料集中展現出最低可用程度的準確率。相關結果應視為架構可行性的初步驗證，而非模型的最終效能。

後續仍需投入更多運算與研究資源，以探討模型規模化效果、改善多模態特徵對齊、提升候選框排序與定位能力，並在更大且更多樣化的資料集上驗證其泛化能力。

---

Due to current hardware and computing-resource constraints, this project has not yet conducted systematic model-scaling experiments, including increases in model width, depth, Transformer capacity, training-data scale, or training duration.

The current model has demonstrated only a minimally usable level of accuracy on the existing test dataset. These results should be considered an initial validation of architectural feasibility rather than the final performance of the model.

Further research and additional computing resources are required to evaluate model scaling, improve multimodal feature alignment, enhance ranking and localization performance, and verify generalization on larger and more diverse datasets.

</details>

---

## 簡介

### 模型能力
![模型能力](https://github.com/ImChouOWO/LightDet/blob/main/units/runs/predict/lightdet_odvg/prediction.jpg)

> 輸入文字：紅色的船　`F: 0.695`　`Q: 0.937`　`A: 0.515`  
> `F：最終置信度`　`Q：定位品質分數`　`A：描述對齊分數`  
> `F為 Q、Ａ之幾何平均`


#### 使用資料集
 | 單位（張）| 訓練集 |驗證集|
|---|---|---|
|數量|15000|1500|

>[!NOTE]
>本模型訓練及驗證皆於自行收集之海事資料

| Recall | Recall@1 | Recall@5 | Recall@10 |
|---:|---:|---:|---:|
| 64.91% | 22.27% | 50.95% | 61.07% |

`模型版本：FPN_gate_fused`


### 模型資料流：

![模型資料流](https://github.com/ImChouOWO/LightDet/blob/main/img/dataFlow.png)

>[!NOTE] 
> `訓練時`，模型輸出候選框與分數，經 Hungarian Matcher 與 Ground Truth 配對後計算 Loss</br>
> `推論時`，依分預測數進行Threshold 與 Top-K 篩選最終結果

---

### 模型結構
![模型架構](https://github.com/ImChouOWO/LightDet/blob/main/img/LightDet.png)

>[!NOTE] 
> 模型淺層時透過FPN擷取圖像資訊，深層時以`Token fusion`融合文字與圖像資訊並透過`Transformer`進行高階語意的擷取。

目前模型包含以下主要設計：

| 模組 | 說明 |
|---|---|
| `Learnable Object Queries` | 以 `Learnable parameters` 作為Query的學習依據，用於候選物件定位 |
| `Decoder-only Localization` | 以 `Query` 從融合後的影像與文字記憶中擷取定位資訊 |
| `Staged Query Refinement` | 第二階段進一步估計定位品質與文字對齊程度 |
| `H-DETR Auxiliary Branch` | 訓練期間增加輔助分支，提高正樣本監督密度 |
| `Hungarian Matching` | 根據 `BBox、GIoU、Score` 與文字對齊成本進行一對一匹配 |
| `Token-level Alignment` | 學習 `Object Query` 與文字片段 token 的對應關係 |
| `Duplicate Suppression` | 抑制多個 `Query` 對同一物件產生重複預測 |
| `Hard Negative Mining` | 強化高分錯誤候選框與負文字片段的辨識能力 |
| `EMA` | 保存模型參數的指數移動平均版本 |

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
│       └── images_1024_uint8/
│
├── units/
│   ├── model/
│   │   ├── bert/
│   │   ├── cards/
│   │   │   ├── cache/
│   │   │   │   ├── bert_raw_cache.pt
│   │   │   │   └── negative_query_pool.json
│   │   │   ├── config/
│   │   │   │   ├── model.yaml
│   │   │   │   └── train.yaml
│   │   │   ├── loss.py
│   │   │   └── ranking_loss.py
│   │   ├── pipeline/
│   │   │   └── data.py
│   │   ├── tool/
│   │   ├── runs/
│   │   │   └── train/
│   │   └── train.py
│   │
│   └── tool/
│       └── card.py
│
├── requirements.txt
└── README.md
```

---

## 安裝

### 1. 下載專案

```bash
git clone https://github.com/ImChouOWO/LightDet.git
cd LightDet
```

### 2. 建立虛擬環境

建議使用 Python 3.10 以上版本。

Linux / macOS：

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
```

Windows PowerShell：

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. 安裝 PyTorch

請依照 CUDA、CPU 或其他運算平台安裝相容的 PyTorch 與 TorchVision。

```bash
pip install torch torchvision
```

### 4. 安裝其餘相依套件

```bash
pip install -r requirements.txt
pip install numpy pyyaml scipy
```

目前 `requirements.txt` 包含：

```text
pillow
tqdm
transformers
huggingface-hub
```

---

## 資料集格式

資料集根目錄預設為：

```text
datasets/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── .cache/
```

影像與標記檔必須使用相同檔名：

```text
datasets/images/train/000001.jpg
datasets/labels/train/000001.json

datasets/images/val/000101.jpg
datasets/labels/val/000101.json
```

支援的常見影像格式：

```text
.jpg  .jpeg  .png  .bmp  .tif  .tiff  .webp
```

### 標記檔範例

每張影像對應一個 JSON 物件，內容包含影像資訊、完整文字描述，以及文字片段與邊界框之間的 Grounding 關係。

```json
{
  "filename": "2021_10_20_13_28_41_00000.jpg",
  "height": 1296,
  "width": 2304,
  "grounding": {
    "caption": "黃綠白相間的船。單色紅色的船。含綠色的船。",
    "regions": [
      {
        "semantic_key": "colors_exact:白色|黃色|綠色",
        "phrase": "黃綠白相間的船",
        "tokens_positive": [
          [0, 7]
        ],
        "bbox": [
          [1858, 493, 2017, 558]
        ]
      },
      {
        "semantic_key": "single_color:紅色",
        "phrase": "單色紅色的船",
        "tokens_positive": [
          [8, 14]
        ],
        "bbox": [
          [1510, 433, 1535, 465]
        ]
      },
      {
        "semantic_key": "contains_color:綠色",
        "phrase": "含綠色的船",
        "tokens_positive": [
          [15, 20]
        ],
        "bbox": [
          [1858, 493, 2017, 558],
          [1590, 454, 1653, 501]
        ]
      }
    ]
  },
  "metadata": {
    "source_name": "2021_10_20_13_28_41_00000",
    "source_label": "2021_10_20_13_28_41_00000.json",
    "object_count": 3,
    "region_count": 3,
    "phrase_variants": {
      "單色紅色的船": [
        "純紅色的船",
        "只有紅色的船"
      ]
    },
    "original_query_texts": [
      "黃綠白相間的船",
      "紅色的船",
      "綠色的船"
    ]
  }
}
```

| 欄位 | 必要 | 說明 |
|---|---:|---|
| `filename` | 是 | 對應的影像檔名 |
| `height` | 是 | 原始影像高度 |
| `width` | 是 | 原始影像寬度 |
| `grounding.caption` | 是 | 包含所有目標描述的完整文字 |
| `grounding.regions` | 是 | 文字描述與邊界框的對應清單 |
| `regions[].semantic_key` | 建議 | 用於區分語意類型或描述規則 |
| `regions[].phrase` | 是 | 對應目標的文字描述 |
| `regions[].tokens_positive` | 是 | `phrase` 在 `caption` 中的字元區間 `[start, end]` |
| `regions[].bbox` | 是 | 該文字描述對應的一個或多個邊界框 |
| `metadata.object_count` | 否 | 影像中的實際物件數量 |
| `metadata.region_count` | 否 | Grounding Region 的數量 |
| `metadata.phrase_variants` | 否 | 各文字描述可使用的擴增句型 |
| `metadata.original_query_texts` | 否 | 原始標記檔中的文字查詢 |

邊界框使用原始影像的像素座標：

```text
[x1, y1, x2, y2]

x1 < x2
y1 < y2
```

同一個 `region` 可以包含多個邊界框，表示同一段文字描述對應多個物件；同一個物件也可以出現在不同的 `region` 中，表示該物件可由不同語意描述進行定位。

`tokens_positive` 使用字元索引標記 `phrase` 在完整 `caption` 中的位置。資料載入時會依照 tokenizer 的 offset mapping，將字元區間轉換為 token-level 對齊標籤。

邊界框不需要預先正規化。資料載入流程會依照 `image_size` 進行影像縮放，並將座標轉換為模型訓練使用的格式。

---

## Query Budget Batch

當 `query_budget: true` 時，`batch_size` 代表一個 batch 可容納的 Grounding Region 預算，而不是固定的影像數量。

```yaml
data:
  query_budget: true

train:
  batch_size: 48
```

例如：

```text
影像 A：9 個 Region
影像 B：13 個 Region
影像 C：10 個 Region
影像 D：14 個 Region
--------------------
總計：46 個 Region
```

上述四張影像可被組合成同一個 batch。

---

## Query Budget Batch

當 `query_budget: true` 時，`batch_size` 代表一個 batch 可容納的文字 Query 預算，而不是固定的影像數量。

```yaml
data:
  query_budget: true

train:
  batch_size: 48
```

例如：

```text
影像 A：9 個 Query
影像 B：13 個 Query
影像 C：10 個 Query
影像 D：14 個 Query
--------------------
總計：46 個 Query
```

上述四張影像可被組合成同一個 batch。

---

## 文字編碼與快取

LightDet 預設使用中文 BERT：

```text
hfl/chinese-macbert-base
```

預計使用的文字特徵快取位置：

```text
units/model/cards/cache/bert_raw_cache.pt
```

模型設定：

```yaml
model:
  freeze_bert: true
  precomputed_bert_path: units/model/cards/cache/bert_raw_cache.pt
```

當 BERT 被凍結時，可預先計算並重複使用文字特徵，以降低訓練期間的重複編碼成本。

---

## 負文字片段

負文字設定檔預設位於：

```text
units/model/cards/cache/negative_query_pool.json
```

目前資料流程採用 ODVG phrase-level negative 設計。抽樣到的負文字片段會附加到完整 caption 中，但不配置對應的邊界框，主要用於 token alignment 監督。

相關訓練設定：

```yaml
data:
  negative_query_path: units/model/cards/cache/negative_query_pool.json
  negative_sample_ratio: 0.1
  negative_phrase_max_per_image: 3
  negative_phrase_separator: "；負向描述："
  use_negative_queries_in_val: false
```

| 參數 | 說明 |
|---|---|
| `negative_sample_ratio` | 訓練樣本加入負文字片段的比例 |
| `negative_phrase_max_per_image` | 每張影像最多加入的負文字片段數 |
| `negative_phrase_separator` | 正向描述與負向描述之間的分隔字串 |
| `use_negative_queries_in_val` | 驗證階段是否加入負文字片段 |

---

## 模型設定

模型設定檔：

```text
units/model/cards/config/model.yaml
```

目前主要設定如下：

```yaml
model:
  img_in_channels: 1024
  cnn_layers: 3
  hidden_dim: 512
  image_grid_size: 10
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

### 關鍵參數

| 參數 | 說明 |
|---|---|
| `num_object_queries` | 每張影像使用的可學習物件 Query 數量 |
| `fusion_token_num` | 影像與文字融合時使用的 Fusion Token 數量 |
| `num_layers` | 第一階段定位 Transformer 層數 |
| `staged_query_refinement` | 是否啟用第二階段品質與文字對齊 refinement |
| `score_bbox_conditioning` | 第二階段是否使用預測框資訊作為條件 |
| `score_bbox_detach` | 傳入第二階段前是否停止 BBox 梯度回傳 |
| `score_fusion` | 定位分數與文字對齊分數的融合方式 |
| `text_max_length` | 最大文字 token 長度 |
| `use_auxiliary_head` | 是否啟用 H-DETR 輔助訓練分支 |

> 修改模型結構參數後，舊 checkpoint 可能無法完整載入。若新增或移除模組，應優先使用 weights-only warm start，而不是恢復舊 optimizer 狀態。

---

## 訓練設定

>[!NOTE]! [訓練參數說明](https://github.com/ImChouOWO/LightDet/blob/main/units/model/cards/config/readme.md)

訓練設定檔：

```text
units/model/cards/config/train.yaml
```

目前預設核心設定：

```yaml
data:
  dataset_dir: datasets
  image_size: 1024
  max_text_aug_per_image: 1
  cache_images: true
  image_cache_dir: datasets/.cache/images_1024_uint8
  prebuild_image_cache: true
  cache_workers: 8
  prefetch_factor: 2
  pin_memory: true
  persistent_workers: true
  query_budget: true

train:
  epochs: 300
  batch_size: 48
  warmup_epochs: 3
  num_workers: 16
  device: 0
  seed: 49
  deterministic: false
  use_amp: true
  amp_dtype: bf16
  allow_tf32: true
  matmul_precision: high
  compile: false
  startup_smoke_test: true
  use_ema: true
  ema_decay: 0.999
  grad_clip_norm: 1.0
```

`device: cuda:1` 代表使用第 2 張 CUDA GPU。只有一張 GPU 時請改為：

```yaml
train:
  device: cuda:0
```

---

## Optimizer 與 Learning Rate

目前使用 AdamW，並可對不同模型元件設定獨立的 learning-rate schedule。

```yaml
optim:
  components:
    vision: ["cosine", 0.0001, 0.000005, 0.00, 1.00]
    text: ["cosine", 0.00001, 0.0000005, 0.00, 1.00]
    transformer: ["cosine", 0.0001, 0.000005, 0.00, 1.00]
    head: ["cosine", 0.0001, 0.000005, 0.00, 1.00]

  weight_decay: 0.0001
  max_warmup_steps: 3000
  fused: true
```

每個元件的格式為：

```text
[排程模式, 最大 LR, 最小 LR, 起始訓練比例, 結束訓練比例]
```

---

## Loss 與 Matching

目前版本的主要 Loss 組成：

| Loss / 機制 | 用途 |
|---|---|
| BBox L1 Loss | 回歸預測框座標 |
| GIoU Loss | 改善預測框與 GT 的幾何重疊 |
| IA-BCE / IoU-aware Classification | 使分數反映預測框定位品質 |
| Token Alignment Loss | 建立 Object Query 與文字 token 的對應 |
| Phrase Ranking Loss | 使正確匹配 Query 的 token 分數高於未匹配 Query |
| Auxiliary Branch Loss | 提供 H-DETR 輔助分支監督 |
| Duplicate Suppression Loss | 抑制同一物件的重複高分預測 |
| Hard Negative Loss | 處理高分但低 IoU 的錯誤候選框 |
| Text Negative Loss | 抑制負文字片段與物件 Query 的錯誤對齊 |

Hungarian Matcher 使用以下成本：

```yaml
loss:
  matcher:
    cost_bbox: 5.0
    cost_giou: 2.0
    cost_score: 2.0
    score_cost_type: focal
    cost_alignment: 2.0
    alignment_negative_weight: 0.25
```

Loss 權重可在前期動態調整：

```yaml
loss:
  weight:
    dynamic: true
    bbox_start: 5.0
    bbox_end: 3.5
    giou_start: 2.0
    giou_end: 2.0
    score_start: 1.0
    score_end: 4.0
    start_epoch: 1
    end_epoch: 30
    schedule: cosine
```

---

## 啟動訓練

預設設定檔已由 `train.py` 載入時，可直接執行：

```bash
python3 units/model/train.py
```

也可以先切換到模型目錄：

```bash
cd units/model
python3 train.py
```

開始訓練前，請至少確認：

```text
1. datasets 路徑正確
2. train / val 影像與標記檔相互對應
3. BERT 模型或文字快取可被讀取
4. train.yaml 的 device 對應實際 GPU
5. image_cache_dir 與 image_size 一致
6. 輸出目錄具有寫入權限
```

---

## Checkpoint 與續訓

訓練輸出位置由下列設定控制：

```yaml
log:
  save_dir: units/model/runs/train/lightdet_ODVG_token_alignment
  weights_path: null
  resume_path: null
  prefer_ema: true
  save_latest_interval: 1
  save_epoch_interval: 50
```

### 從頭訓練

```yaml
log:
  weights_path: null
  resume_path: null
```

### Weights-only warm start

只載入模型權重，不恢復 optimizer、scheduler、epoch 與最佳指標。

```yaml
log:
  weights_path: /path/to/checkpoint.pt
  resume_path: null
  prefer_ema: true
```

適合以下情況：

```text
模型結構有小幅調整
新增 token-alignment projection head
修改 Loss 或資料策略
希望重新建立 optimizer 與 learning-rate schedule
```

### 完整接續訓練

恢復模型、EMA、optimizer、scheduler、epoch 與可用的 RNG 狀態。

```yaml
log:
  weights_path: null
  resume_path: /path/to/latest.pt
```

`weights_path` 與 `resume_path` 不應同時設定。

---

## 驗證與後處理

```yaml
eval:
  val_loss_interval: 5
  eval_interval: 1
  score_thr: 0.001
  top_k: 20
  nms_iou_thr: 0.5
  use_nms: false
  use_topk_fallback: false
  compute_raw_oracle: true
  raw_oracle_iou_thresholds: [0.25, 0.50, 0.75]
  best_metric: map50_95
```

目前預設 `use_nms: false`，主要依賴 DETR-style Query assignment、文字對齊與 duplicate suppression 學習降低重複預測。

常見驗證指標包含：

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

最佳 checkpoint 依 `best_metric: map50_95` 選擇。

---

## 訓練輸出

預設輸出目錄：

```text
units/model/runs/train/lightdet_ODVG_token_alignment/
```

可能包含：

```text
latest.pt
best_map50_95.pt
epoch_050.pt
metrics_epoch.jsonl
metrics_step.jsonl
metrics_epoch.csv
latest_metrics.json
```

| 檔案 | 說明 |
|---|---|
| `latest.pt` | 最近一次保存的完整 checkpoint |
| `best_map50_95.pt` | 驗證集 mAP50-95 最佳 checkpoint |
| `epoch_XXX.pt` | 依固定週期保存的 checkpoint |
| `metrics_epoch.jsonl` | Epoch-level 指標 |
| `metrics_step.jsonl` | Step-level 指標，需啟用 step metrics |
| `metrics_epoch.csv` | CSV 格式的 epoch 指標 |
| `latest_metrics.json` | 最近一次訓練與驗證摘要 |

---





