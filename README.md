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

目前 ResNet 版本使用 ImageNet V2 預訓練 ResNet-50、FPN、多模態 Decoder-only Object Queries、FPN ROI refinement 與 Main/Auxiliary matching。Quality 與描述匹配分開監督，推論再融合分數。資料標註使用 `lightdet.relations.v1`，同一個物件可以對應多段描述；原 BottleNet 設定保留供比較。

<details>
<summary><em>當前限制 / Current Limitations</em></summary>

ResNet 實驗已完成 100 epoch，最佳 Hit@1 為 52.28%（epoch 55）；後期訓練 loss 下降而驗證語意 loss 上升，排序泛化仍有改善空間。Hit@10 與 Oracle50 是候選診斷，不能當成保證可達的 Hit@1。實際端到端推論延遲尚未量測。

---

The ResNet experiment completed 100 epochs, with the best Hit@1 of 52.28% at epoch 55. Validation semantic loss rises during late training despite decreasing training loss. Hit@10 and Oracle50 diagnose candidate coverage and do not guarantee achievable Hit@1. End-to-end inference latency has not been measured.

</details>

---

## 簡介

### 模型能力

![模型能力](https://github.com/ImChouOWO/LightDet/blob/main/units/runs/predict/lightdet_odvg/prediction.jpg)

模型可使用多種文字條件查詢同一張影像，例如「單色紅色的船」與「含紅色的船」。同一個物件只保留一份定位框，描述關係由物件 ID 建立。

目前資料集包含 10,733 張訓練影像與 2,683 張驗證影像。ResNet 實驗使用 `lightdet_grounding_resnet50_roi8_hardneg_log_100ep`，最佳 EMA checkpoint 位於 epoch 55：Hit@1 **52.28%**、Hit@5 **61.72%**、Hit@10 **68.50%**、Object Recall@10 **46.08%**、Set Success50 **3.61%**。

**消融與版本比較**

以下採各實驗最後一次訓練序列中、Hit@1 最佳 epoch 的同次驗證數值；均使用 `lightdet.grounding.v2`。這些是累積版本實驗，初始化、排程與訓練長度並未完全控制，不能將差異歸因於單一模組。

| 實驗 | 已完成 epoch | 最佳 epoch | Hit@1 | TextHit@1 | Oracle50 |
|---|---:|---:|---:|---:|---:|
| 背景監督（BottleNet） | 51 | 45 | 45.88% | 42.06% | 73.99% |
| 對比項（BottleNet） | 25 | 8 | 45.73% | 41.30% | 73.96% |
| GT ROI 暖機（BottleNet） | 39 | 7 | 45.47% | 40.18% | 73.98% |
| 描述匹配（BottleNet） | 69 | 8 | 45.38% | 39.51% | 73.97% |
| 描述配對 MLP（BottleNet） | 17 | 7 | 44.98% | 38.70% | 73.96% |
| 困難負例＋log（BottleNet，未完成） | 5 | 1 | 44.50% | 34.77% | 73.95% |
| 困難負例＋log（ResNet-50） | 100 | 55 | 52.28% | 38.43% | 75.62% |

> [!NOTE]
> 實驗目錄名稱不是配置證據；例如名稱含 `roi16` 的描述匹配實驗，其後續報告實際使用 ROI8。BottleNet 的困難負例＋log 實驗在 epoch 6 中斷，只完成 5 個 epoch，不能用來判定新 loss 的最終效益。
> ResNet 除 Backbone 外，也加入 ImageNet 正規化、固定 BN 統計、重建 FPN／影像投影，並取消前 10 epoch 定位凍結；其餘任務模組沿用相同的 ROI8 phrase-pair E7 起始權重。

ResNet 在 epoch 45 已達 Hit@1 52.14%，epoch 55 最佳為 52.28%，epoch 100 則為 50.96%。Oracle50 從 epoch 50 的 74.74% 升至 76.63%，但 TextHit@1 從 38.89% 降至 36.76%；較好的候選定位沒有持續轉成較好的語意排序。使用最佳 checkpoint，不以最後一個 epoch 代表最佳表現。

### 模型資料流

![模型資料流](./img/dataFlow.png)

```text
Image + Text -> LightDet -> Main / Aux BBox + Quality + Alignment [Q,P]
推論：sigmoid 轉為機率 -> sqrt(Quality × Alignment) -> 每個描述排序 -> threshold + NMS -> Result
評估：未經 threshold / NMS 的原始候選 -> Hit@K / Object Recall@K / Oracle50
訓練：唯一 GT BBox + Object–Phrase Relation -> 分別建立定位、Quality 與語意監督
```

Main 使用幾何成本的 Hungarian one-to-one matching；Aux 使用受 IoU 與成本限制的 one-to-many matching，只在訓練啟用。Hungarian 決定定位配對，不會替 GT 選出唯一描述。多描述關係由標記矩陣提供，Quality、Alignment 與 Ranking 分別計算 loss，再加權合併。GT ROI 暖機直接以 GT 框擷取 FPN ROI，提供額外語意監督。

### 模型結構

![模型架構](./img/LightDet.png)

| 模組 | 說明 |
|---|---|
| `BackBone` | ResNet-50 layer2／3／4 輸出 stride 8／16／32、512／1024／2048 通道；影像先做 ImageNet 正規化，固定 BN 統計但保留 affine 訓練；原 BottleNet 路徑保留 |
| `FPN` | 融合三個尺度，統一至 256 通道；同時供定位 token 與原始 FPN ROI 擷取使用 |
| `Image Projector` | 各尺度使用獨立投影與 refinement，輸出 24×24、12×12、10×10，共 820 個 256 維 image tokens；ROI 不從這些縮小的 tokens 裁切 |
| `Bert` | 凍結 MacBERT 編碼完整 caption；可訓練的 Linear → LayerNorm → GELU 將文字由 768 投影至 256 維，使用 raw BERT 快取 |
| `Main object query / Aux object query` | 各 100 個候選 query；Main 使用 one-to-one，Aux 最多 3 個 query 對應一個 GT，僅訓練使用 |
| `Learnable Fusion Token` | 提供多模態融合 query，與物件 queries 一起進入定位階段 |
| `Localization Transformer` | 使用影像與文字 context，產生 Location Query |
| `BBox Head` | 從 Location Query 預測正規化 xyxy BBox |
| `BBox Query Conditioning` | 將 detach 的 BBox 幾何資訊編碼為評分 query；不直接沿用 Location Query 特徵 |
| `ROI Align` | 根據各候選 BBox，從 FPN feature maps 擷取 8×8、256 通道的局部特徵 |
| `Text-Alignment Transformer` | ROI pooled feature＋BBox embedding 經每個 ROI 自己的 64 個 tokens 做局部 refinement，產生 Score Query；不讀取文字，也不讓不同 bbox queries 互相 attention |
| `Alignment Head` | ROI／文字各自 LayerNorm＋Linear 256→256；文字依描述 span 加權池化，與 ROI 做正規化相似度及配對 MLP，輸出 `[Q,P]` logits |
| `Quality Head` | ROI 平均池化後走獨立視覺投影及 Quality Head，輸出 `[Q,1]` logits，經 sigmoid 預測最大 GT IoU；不接收文字或 Score Query |
| `Phrase Ranking` | 融合 `sqrt(Quality × Alignment)`，依描述分別排序；輸出使用閾值與 NMS，原始候選評估則不使用後處理 |

前段 FPN、BERT 投影與 ROI refinement 由 Main／Aux 共用，各自有獨立的 BBox、Quality 與 Alignment Head。描述匹配採 `cosine(v,t) + 0.25 × tanh(MLP([v*t, |v-t|]))`，MLP 為 512→64→1，再用可學習 scale／bias 轉成 logits。每個 ROI 只做一次 refinement，所有描述共用其視覺摘要；描述不會重新引導 ROI 空間注意力。

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
│   │   │   ├── config/{model.yaml,train.yaml,model_resnet50.yaml,train_resnet50.yaml}
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

`query_budget: true` 時，`batch_size` 代表一個 batch 的 region 預算，每批另設最多 6 張影像。資料集會依影像的描述數量組合 batch，並在每個 epoch 重新排序與組批，避免固定分組造成樣本分布偏差。

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

原 BottleNet 設定：`units/model/cards/config/model.yaml`。ResNet 獨立設定：`units/model/cards/config/model_resnet50.yaml`。

目前 ResNet 關鍵設定：

```yaml
model:
  hidden_dim: 256
  backbone:
    name: resnet50
    in_channels: 3
    weights: IMAGENET1K_V2
    norm_eval: true
  fpn:
    out_channels: 256
  num_object_queries: 100
  num_heads: 8
  num_layers: 2
  staged_query_refinement: true
  score_num_heads: 8
  score_num_layers: 2
  score_roi_size: 8
  score_bbox_detach: true
  score_fusion: geometric_mean
  text_max_length: 256
  freeze_bert: true
  use_auxiliary_head: true
```

影像 token grids 為 `24×24`、`12×12` 與 `10×10`。BBox conditioning 與 ROI 使用 detach 的框座標；語意 loss 不直接更新框座標，但定位與語意仍共用 FPN。ImageNet 正規化只在 ResNet 路徑執行，不重複套用至資料載入或 BottleNet。

---

## 訓練設定

原實驗設定：`units/model/cards/config/train.yaml`。ResNet 獨立設定：`units/model/cards/config/train_resnet50.yaml`。

ResNet 使用 100 epoch，從第一個 epoch 聯合訓練新影像分支與定位模組：

```yaml
train:
  epochs: 100
  localization_freeze_epochs: 0
  batch_size: 48
  warmup_epochs: 5
  use_amp: true
  amp_dtype: bf16

loss:
  score_sampling:
    max_positive_per_gt: 3
    min_extra_positive_iou: 0.30
  text_alignment:
    ranking_weight: 0.25
    ranking_contrast_weight: 0.01
    ranking_contrast_temperature: 0.1
    ranking_margin: 0.6931471805599453 # ln(2)
    ranking_negative_top_k: 5
    ranking_hard_negative_quota: 3
  gt_roi:
    weight: 0.5
    hold_epochs: 5
    end_epoch: 15

log:
  save_dir: units/model/runs/train/lightdet_grounding_resnet50_roi8_hardneg_log_100ep
  weights_path: units/model/runs/train/lightdet_grounding_roi8_phrase_pair_100ep/best_hit_at_1.pt
  weights_exclude_prefixes: ["img_model."]
  resume_path: null
```

排除舊影像分支後，ResNet 使用預訓練權重，FPN／Image Projector 重新初始化，其餘任務模組載入原 EMA 權重；optimizer 與排程重設。原 BottleNet 實驗保留於原目錄，不以 ResNet 輸出覆寫。

---

## Loss 與 Matching

### Main branch

每個唯一物件只進行一次 Hungarian one-to-one matching：

```text
BBox L1 + GIoU（目前 Quality／Alignment matching cost 均為 0）
              ↓
       Query ↔ unique object
```

### Auxiliary branch

每個 GT 從成本矩陣選擇最多 3 個候選 query，且必須滿足 IoU ≥ 0.30。候選由 cost 排序並解決 query 衝突，僅用於訓練輔助監督。

### Relation loss

從全部 queries 篩選唯一對應 GT 且 IoU ≥ 0.5 的候選，再依物件 ID 查詢關係矩陣：

```text
query × phrase → positive / negative / ignore
```

整體 loss 為 Main 定位＋Quality、Alignment、Ranking、加權 Aux，以及 GT ROI 暖機：

| 項目 | 監督與目前權重 |
|---|---|
| BBox／GIoU | Hungarian 配對後回歸 GT；L1 權重 5→3.5，GIoU 權重 2 |
| Quality | IoU ≥ 0.5 候選的最大 GT IoU；可靠背景目標為 0；權重 1→4 |
| Alignment | 關係矩陣的 masked focal BCE；正例 1、負例 0、未知忽略；權重 1 |
| Ranking | 0.25×雙向 log-margin 排序＋0.01×雙向對比項；語意暖機 7 epoch |
| Aux | 定位、Quality 與 Alignment 整組乘 0.25，另有暖機與後期衰減；Aux Ranking 權重目前為 0 |
| GT ROI | GT 框擷取乾淨 ROI，提供 Alignment、排序與對比項；前 5 epoch 權重 0.5，第 15 epoch 降至 0 |

融合與排序中的 Quality／Alignment 使用 sigmoid 後機率；Focal BCE 使用 logits。雙向排序使用 hinge loss `ReLU(margin − log(s_positive) + log(s_negative))`：

```text
同一描述／不同物件：s = sqrt(detach(Quality) × Alignment)
同一物件／不同描述：s = Alignment
margin = ln(2)：要求正例分數至少是負例的 2 倍
```

兩方向各取最多 5 個已知負例，優先保留最多 3 個與描述共享顏色但關係為負的候選，再補入其餘高分負例。同一物件的多個正確描述／同一描述的多個正確物件不互相競爭；未知關係不被取樣。重複 query 以每物件候選數倒數加權，物件方向每個負例 GT 只保留最高分候選。

Quality 在 Ranking 中 detach。同樣的 log margin 在 Quality 相同時，物件方向需要 Alignment 正／負比 ≥4，描述方向則 ≥2；GT ROI 的兩方向都使用 Alignment。對比項使用同一匹配 affinity，temperature=0.1，不另外建立一套推論分數。

背景監督由全部 GT 與預測框的幾何關係建立：最大 IoU ≤0.2、中心不在 GT 內，且交集／候選面積與交集／GT 面積皆 ≤0.2。目前訓練依使用者確認的完整標記對所有影像啟用，不需修改 JSON；分數只用來選高分背景，不決定正負標籤。前 5 epoch 停用，第 6–10 epoch 逐步加入 Quality、Alignment 與 Ranking 背景 loss。IoU 中間區間及模糊對應不強制標為背景。

推論依描述融合分數，使用 `score_thr=0.5` 與 NMS IoU=0.5；Quality 與 Alignment 的定位／語意任務分開。Matching cost 沒有加入文字分數，不啟用舊 duplicate-suppression loss；重複輸出由描述內 NMS 處理。困難負例候選數與實際選入數另記錄於訓練指標。

---

## 啟動訓練

```bash
cd /PATH/TO/LightDet
source venv/bin/activate
python3 units/model/train.py \
  --model-config units/model/cards/config/model_resnet50.yaml \
  --train-config units/model/cards/config/train_resnet50.yaml
```

伺服器可將相同命令的入口改為 `units/model/_train.py`。原 BottleNet 使用 `model.yaml`／`train.yaml`；兩種模型由設定選擇，不覆寫原模型設定。啟動前執行 real-batch smoke test，確認資料、輸出與 loss 形狀；ResNet 路徑與權重轉移已通過 79 項測試及 6 張 1024 圖像的 BF16 forward／backward／optimizer 檢查。

---

## Checkpoint 與續訓

ResNet 輸出目錄：

```text
units/model/runs/train/lightdet_grounding_resnet50_roi8_hardneg_log_100ep/
```

不載入任務 checkpoint：

```yaml
log:
  weights_path: null
  weights_exclude_prefixes: []
  resume_path: null
```

此模式的 ResNet 仍依 `backbone.weights` 使用 ImageNet 預訓練；其他模組重新初始化。`weights_path` 用於新實驗的 weights-only warm start，只有恢復同一架構與實驗狀態時才使用 `resume_path`。

`weights_exclude_prefixes: ["img_model."]` 僅用於從 BottleNet 切換至 ResNet 的轉移實驗。若載入已訓練的 ResNet checkpoint，應改為 `[]`，避免重建已學好的影像分支。Hit@1 任務使用 `best_hit_at_1.pt`，本輪最佳為 epoch 55；不使用 epoch 100 取代最佳 checkpoint。

---

## 驗證與後處理

```yaml
eval:
  eval_interval: 1
  score_thr: 0.5
  nms_iou_thr: 0.5
  use_nms: true
  best_metric: hit_at_1
```

IoU 命中條件為 ≥0.5，指標以 0–1 記錄，表格以百分比呈現：

| 指標 | 分母與定義 |
|---|---|
| `hit_at_1/5/10` | 有正例的描述查詢數；該描述前 K 個候選是否至少命中一個符合描述的 GT 物件 |
| `object_recall_at_5/10` | 正確描述–物件配對總數；前 K 候選經一對一最大匹配後召回的 GT 比例，重複框不重複召回 |
| `set_success50` | 有效描述查詢數；後處理輸出必須找齊全部正例且沒有多報，透過一對一匹配確認；重複框算多報 |
| `no_target_accuracy` | 已確認沒有目標的描述查詢數；是否輸出零個框，沒有此類查詢時為 `null`／N/A |
| `oracle50` | 正確描述–物件配對總數；任一原始候選能否覆蓋每個 GT，忽略分數；是物件層級定位覆蓋，並非 Hit@1 上限 |
| `quality_hit_at_1` / `text_hit_at_1` | 有正例的描述查詢數；分別只用 Quality／Alignment 排序的 Top-1 命中率，用於分析融合前的分支能力 |
| `avg_selected_per_query` | 有效描述查詢數；閾值與 NMS 後平均輸出框數，用於觀察輸出校準 |

Hit@K、Object Recall@K、Oracle50 與單分支 Hit@1 使用未經分數閾值及 NMS 的原始 Main 候選。Set Success50 與 N-acc 使用設定的閾值與 NMS，不強制保留 Top-20、不在空輸出時補框。ignore 物件重疊候選若未同時命中正例，會排除；沒有正例但含未知關係的查詢不計為確認無目標。

> [!NOTE]
> Hit@10=68.50% 表示前十候選包含正確答案的查詢比例；若能將它們完美重排序，才可轉為至少相同的 Hit@1。這是候選提供的可達空間，不是模型經過訓練就保證達到的準確率。Oracle50 分母不同，不能直接拿 75.62% 當作 Hit@1 的理論上限。

本驗證集沒有確認無目標的描述，因此 N-acc 為 N/A，不以 0 取代。公平比較固定相同閾值、NMS、資料與 `metric_schema: lightdet.grounding.v2`。舊 mAP、P/R、`recall50_at_*` 不能換算成新版 Hit@K，歷史權重需要重新驗證。

---

## 訓練輸出

| 檔案 | 說明 |
|---|---|
| `latest.pt` | 最近一次完整 checkpoint |
| `best_hit_at_1.pt` | `hit_at_1` 最佳 checkpoint |
| `epoch_XXX.pt` | 每 10 epoch 保存的 checkpoint |
| `metrics_epoch.jsonl` | Epoch-level 訓練與驗證指標 |
| `latest_metrics.json` | 最近一次摘要 |

---

## 參考資料

- [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
- [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)

## License

本專案採用 [MIT License](./LICENSE)。使用、修改、發布或再散布本專案及其衍生作品時，請保留 `LICENSE` 中的原作者版權聲明與授權條款，並標記原作者 **Cheng-Han Chou**。
