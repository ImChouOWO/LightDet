# LightDet

![Framework](https://img.shields.io/badge/framework-PyTorch-EE4C2C)
![Architecture](https://img.shields.io/badge/architecture-DETR%20%2B%20FPN%20%2B%20Transformer-8A2BE2)
![Modality](https://img.shields.io/badge/input-Text%20%2B%20Image-2F80ED)
![Task](https://img.shields.io/badge/task-Multimodal%20Object%20Detection-4CAF50)
![Precision](https://img.shields.io/badge/precision-BF16%20AMP-FF9800)
![Status](https://img.shields.io/badge/status-active%20research-yellow)

LightDet 是文字引導的多模態物件偵測模型。模型輸入影像與文字描述，輸出與描述相符的邊界框、定位品質分數與文字對齊分數。

```text
Image + Text Phrases -> Object Queries -> Boxes + Quality + Phrase Alignment
```

目前版本使用 FPN、多模態 Decoder-only Object Queries、兩階段 Query Refinement，以及 Main/Auxiliary matching。資料標註使用 `lightdet.relations.v1`，同一個物件可以對應多段描述。

<details>
<summary><em>當前限制 / Current Limitations</em></summary>

目前仍需要在完整 100 epoch 訓練後重新評估 mAP、Recall@K 與實際推論延遲。現有整合測試已驗證資料載入、抽樣、forward、loss 與 backward，但不代表完整訓練結果已完成。

---

The full 100-epoch training run and end-to-end latency evaluation are still required for final performance reporting. The integration test currently validates data loading, sampling, forward, loss, and backward execution.

</details>

---

## 簡介

### 模型能力

![模型能力](https://github.com/ImChouOWO/LightDet/blob/main/units/runs/predict/lightdet_odvg/prediction.jpg)

模型可使用多種文字條件查詢同一張影像，例如「單色紅色的船」與「含紅色的船」。同一個物件只保留一份定位框，描述關係由物件 ID 建立。

目前資料集包含 10,733 張訓練影像與 2,683 張驗證影像。完整訓練結果應以新的 `lightdet_relations_100ep` 實驗輸出為準。

### 模型資料流

![模型資料流](https://github.com/ImChouOWO/LightDet/blob/main/img/dataFlow.png)

```text
Image -> Backbone -> FPN -> Image Tokens
Text  -> Frozen BERT -> Text Tokens
Image Tokens + Text Tokens -> Object Query Transformer
                           -> Main / Aux Boxes and Scores
                           -> Phrase Alignment and Ranking
```

Main branch 使用 Hungarian one-to-one matching；Auxiliary branch 使用受 IoU 與成本限制的 one-to-many matching，僅於訓練時提供額外定位監督。

### 模型結構

![模型架構](https://github.com/ImChouOWO/LightDet/blob/main/img/LightDet.png)

| 模組 | 說明 |
|---|---|
| `FPN` | 將多尺度 CNN 特徵轉為影像特徵 |
| `Frozen BERT` | 編碼文字描述，並使用預計算快取降低成本 |
| `Object Query Transformer` | 從影像與文字 context 產生候選物件 |
| `Main Branch` | 使用一對一 Hungarian matching 產生主要定位監督 |
| `Auxiliary Branch` | 使用 Top-K、IoU 閾值與成本產生 one-to-many 輔助監督 |
| `Relation Alignment` | 根據描述與物件關係矩陣計算正、負與忽略監督 |
| `Bidirectional Ranking` | 同時比較同一描述的不同物件，以及同一物件的不同描述 |
| `Duplicate Suppression` | 降低同一物件的重複預測 |

---

## 專案結構

```text
LightDet/
├── datasets/
│   ├── images/{train,val}/
│   ├── labels/{train,val}/
│   └── .cache/
├── units/
│   ├── model/
│   │   ├── cards/
│   │   │   ├── config/{model.yaml,train.yaml}
│   │   │   ├── loss.py
│   │   │   └── relation_loss.py
│   │   ├── pipeline/{data.py,relations.py}
│   │   ├── tool/
│   │   └── train.py
│   └── tool/card.py
├── migrate_annotations.py
└── requirements.txt
```

---

## 安裝

### 1. 下載專案

```bash
git clone https://github.com/ImChouOWO/LightDet.git
cd LightDet
```

### 2. 建立虛擬環境

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 已包含目前環境使用的 PyTorch、TorchVision、Transformers、SciPy、Pillow 與 PyYAML。CUDA 版本應與目標 GPU 的 PyTorch 安裝相容。

---

## 資料集格式

資料集目錄如下：

```text
datasets/
├── images/train/
├── images/val/
├── labels/train/
└── labels/val/
```

影像與標記檔使用相同檔名：

```text
datasets/images/train/000001.jpg
datasets/labels/train/000001.json
```

標記檔使用 `lightdet.relations.v1`：

```json
{
  "schema_version": "lightdet.relations.v1",
  "filename": "2021_10_20_13_28_41_00000.jpg",
  "height": 1296,
  "width": 2304,
  "grounding": {
    "caption": "黃綠白相間的船。含綠色的船。",
    "phrases": [
      {
        "id": "phrase_0000",
        "semantic_key": "colors_exact:白色|黃色|綠色",
        "phrase": "黃綠白相間的船",
        "tokens_positive": [[0, 7]],
        "positive_object_ids": ["object_0000"],
        "negative_object_ids": ["object_0001"],
        "ignore_object_ids": []
      },
      {
        "id": "phrase_0001",
        "semantic_key": "contains_color:綠色",
        "phrase": "含綠色的船",
        "tokens_positive": [[8, 13]],
        "positive_object_ids": ["object_0000", "object_0001"],
        "negative_object_ids": [],
        "ignore_object_ids": []
      }
    ]
  },
  "objects": [
    {
      "id": "object_0000",
      "bbox": [1858, 493, 2017, 558],
      "attributes": {
        "colors": ["白色", "綠色", "黃色"],
        "colors_complete": true
      }
    },
    {
      "id": "object_0001",
      "bbox": [1590, 454, 1653, 501],
      "attributes": {
        "colors": ["綠色"],
        "colors_complete": true
      }
    }
  ]
}
```

| 欄位 | 必要 | 說明 |
|---|---:|---|
| `schema_version` | 是 | 必須為 `lightdet.relations.v1` |
| `filename` | 是 | 對應影像檔名 |
| `grounding.caption` | 是 | 包含所有文字描述的完整字串 |
| `grounding.phrases` | 是 | 描述及其物件關係 |
| `phrases[].tokens_positive` | 是 | 描述在 caption 中的字元區間 |
| `positive_object_ids` | 是 | 確認符合描述的物件 |
| `negative_object_ids` | 是 | 確認不符合描述的物件 |
| `ignore_object_ids` | 是 | 資訊不足、不得計入該描述 loss 的物件 |
| `objects[].id` | 是 | 同一張影像內唯一的物件 ID |
| `objects[].bbox` | 是 | 原始影像像素座標 `[x1,y1,x2,y2]` |
| `objects[].attributes` | 建議 | 用於保存顏色等可解釋屬性 |

同一個物件可以對應多個 phrase，但 `objects` 中只保留一份 bbox。資料載入時會將 bbox 轉為模型輸入尺寸並正規化。

---

## Query Budget Batch

`query_budget: true` 時，`batch_size` 代表一個 batch 的 region 預算。資料集會依影像的描述數量組合 batch，並在每個 epoch 重新排序與組批，避免固定分組造成樣本分布偏差。

```yaml
data:
  query_budget: true

train:
  batch_size: 48
```

訓練資料會輪替可靠的短同義描述；過長或不符合既定語意規則的 `phrase_variants` 不會被抽樣。外部負文字池目前停用，負例由標記內的關係矩陣提供。

---

## 文字編碼與快取

文字分支使用 `hfl/chinese-macbert-base` 相容的 BERT 權重，並預設凍結 BERT：

```yaml
model:
  freeze_bert: true
  precomputed_bert_path: units/model/cards/cache/bert_raw_cache.pt
```

訓練啟動時會建立或補充文字特徵快取。`text_max_length` 為 256；若 phrase 超出 tokenizer 長度，資料驗證會拒絕該樣本。

---

## 模型設定

設定檔：`units/model/cards/config/model.yaml`

目前關鍵設定：

```yaml
model:
  hidden_dim: 256
  num_object_queries: 100
  num_heads: 8
  num_layers: 2
  staged_query_refinement: true
  score_num_heads: 8
  score_num_layers: 2
  text_max_length: 256
  use_auxiliary_head: true
```

影像 token grids 為 `24×24`、`12×12` 與 `10×10`。第二階段使用 bbox conditioning，並停止 bbox 梯度回傳至 score refinement。

---

## 訓練設定

設定檔：`units/model/cards/config/train.yaml`

目前使用 100 epoch：

```yaml
train:
  epochs: 100
  batch_size: 48
  warmup_epochs: 5
  use_amp: true
  amp_dtype: bf16

loss:
  score_sampling:
    max_positive_per_gt: 3
    min_extra_positive_iou: 0.30

log:
  save_dir: units/model/runs/train/lightdet_relations_100ep
```

---

## Loss 與 Matching

### Main branch

每個唯一物件只進行一次 Hungarian one-to-one matching：

```text
BBox L1 + GIoU + Quality cost
              ↓
       Query ↔ unique object
```

### Auxiliary branch

每個 GT 從成本矩陣選擇最多 3 個候選 query，且必須滿足 IoU ≥ 0.30。候選由 cost 排序並解決 query 衝突，僅用於訓練輔助監督。

### Relation loss

Matching 後，對每個 matched query 計算描述關係：

```text
query × phrase → positive / negative / ignore
```

使用 masked alignment loss，並加入雙向 ranking：

```text
同一描述：正確物件 > 錯誤物件
同一物件：正確描述 > 錯誤描述
```

為避免文字分數污染早期定位，alignment matcher cost 在第 10 epoch 開始，於第 25 epoch 完成 warmup。

---

## 啟動訓練

```bash
cd /home/soic/Desktop/LightDet
source venv/bin/activate
python3 units/model/train.py
```

啟動前會執行 real-batch smoke test，確認資料、模型輸出與 loss 形狀一致。

---

## Checkpoint 與續訓

輸出目錄：

```text
units/model/runs/train/lightdet_relations_100ep/
```

從頭訓練：

```yaml
log:
  weights_path: null
  resume_path: null
```

模型結構、資料格式或 loss 改變後，建議使用 `weights_path` 進行 weights-only warm start；只有要恢復同一實驗狀態時才使用 `resume_path`。

---

## 驗證與後處理

```yaml
eval:
  eval_interval: 1
  score_thr: 0.001
  top_k: 20
  use_nms: false
  best_metric: map50
```

建議同時觀察 `map50`、`quality_recall50_at_1`、`text_recall50_at_1`、`final_recall50_at_1` 與 `raw_oracle_recall50`，區分定位候選品質和文字排序品質。

---

## 訓練輸出

| 檔案 | 說明 |
|---|---|
| `latest.pt` | 最近一次完整 checkpoint |
| `best_map50.pt` | `map50` 最佳 checkpoint |
| `epoch_XXX.pt` | 每 10 epoch 保存的 checkpoint |
| `metrics_epoch.jsonl` | Epoch-level 訓練與驗證指標 |
| `latest_metrics.json` | 最近一次摘要 |

---

## 參考資料

- [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
- [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)

## License

本專案採用 [MIT License](./LICENSE)。使用、修改、發布或再散布本專案及其衍生作品時，請保留 `LICENSE` 中的原作者版權聲明與授權條款，並標記原作者 **Cheng-Han Chou**。
