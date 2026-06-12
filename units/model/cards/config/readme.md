# LightDet 訓練設定檔說明

此設定檔用於控制目前的 `VisionTextModel` 訓練流程。模型輸入為整張影像與文字查詢，輸出為符合文字查詢的多目標物件偵測結果。

目前模型不進行類別分類，不需要 `label prediction`。訓練目標為：

```text
image + query_text -> bbox + score_logit
```

其中 bbox 格式為 normalized `xyxy`：

```text
x1, y1, x2, y2 ∈ [0, 1]
x1 <= x2
y1 <= y2
```

---

## YAML 範例

```yaml
model:
  img_in_channels: 1024
  hidden_dim: 512
  num_heads: 8
  num_layers: 4
  mlp_ratio: 4.0
  image_grid_size: 10
  text_max_length: 32
  fusion_token_num: 16
  dropout: 0.1
  freeze_bert: true

data:
  dataset_dir: /home/soic/Desktop/LightDet/datasets
  image_size: 640
  max_text_aug_per_image: 1

train:
  epochs: 300
  batch_size: 32
  warmup_epochs: 5
  num_workers: 18
  device: cuda:1
  seed: 42
  use_amp: true
  use_ema: true
  ema_decay: 0.999
  grad_clip_norm: 1.0

optim:
  lr_vision: 0.0001
  lr_text: 0.00001
  lr_transformer: 0.0001
  lr_head: 0.0001
  weight_decay: 0.0001
  min_lr_ratio: 0.05
  max_warmup_steps: 3000

loss:
  cost_bbox: 5.0
  cost_giou: 2.0
  cost_score: 1.0
  min_pos_weight: 1.0
  max_pos_weight: 5.0

eval:
  val_loss_interval: 1
  eval_interval: 1
  max_val_batches: 320
  score_thr: 0.25
  top_k: 20
  nms_iou_thr: 0.5
  best_metric: map50

log:
  save_dir: null
  resume_path: null
  save_epoch_interval: 50
  emit_step_metrics: true
  log_interval: 10
```

---

# 1. model

`model` 區塊控制模型結構與文字編碼設定。

| 參數                 |    預設值 | 型別    | 說明                                                                        |
| ------------------ | -----: | ----- | ------------------------------------------------------------------------- |
| `img_in_channels`  | `1024` | int   | `BottleNet` 輸出的影像特徵 channel 數，也是後續 `BackBone / ImgProjector` 的輸入 channel。 |
| `hidden_dim`       |  `512` | int   | 模型內部 token 維度。影像 token、文字 token、fusion token 都會投影到此維度。                    |
| `num_heads`        |    `8` | int   | Transformer multi-head attention 的 head 數。需能整除 `hidden_dim`。              |
| `num_layers`       |    `4` | int   | Transformer encoder layer 數量。越大表示融合能力越強，但訓練與推論成本也越高。                      |
| `mlp_ratio`        |  `4.0` | float | Transformer feed-forward hidden dim 放大倍率。實際維度為 `hidden_dim * mlp_ratio`。  |
| `image_grid_size`  |   `10` | int   | 影像 token grid 大小。若為 `10`，代表輸出 `10 × 10 = 100` 個 image tokens。             |
| `text_max_length`  |   `32` | int   | BERT tokenizer 的最大文字長度。超過會截斷，不足會 padding。                                 |
| `fusion_token_num` |   `16` | int   | learnable fusion tokens 數量，用於整合全域影像與文字資訊。                                 |
| `dropout`          |  `0.1` | float | Transformer layer 中的 dropout rate。                                        |
| `freeze_bert`      | `true` | bool  | 是否凍結 BERT。若為 `true`，BERT 不更新參數，只訓練 projection 與後續模組。                      |

## image_grid_size 對輸出數量的影響

模型每個 image token 會預測一個 bbox 與 score，因此：

```text
num_predictions = image_grid_size × image_grid_size
```

例如：

| `image_grid_size` | image token 數 | bbox 輸出 shape  |
| ----------------: | ------------: | -------------- |
|              `10` |         `100` | `[B, 100, 4]`  |
|              `20` |         `400` | `[B, 400, 4]`  |
|              `40` |        `1600` | `[B, 1600, 4]` |

建議初期使用 `10` 或 `20`。若資料中同張圖的目標數很多，可以提高到 `20`，但會增加 Transformer 與 head 的計算量。

---

# 2. data

`data` 區塊控制資料集路徑與 dataloader 輸入尺寸。

| 參數                       |                                    預設值 | 型別  | 說明                                                      |
| ------------------------ | -------------------------------------: | --- | ------------------------------------------------------- |
| `dataset_dir`            | `/home/soic/Desktop/LightDet/datasets` | str | 資料集根目錄。                                                 |
| `image_size`             |                                  `640` | int | 輸入影像 resize 尺寸。影像會被 resize 成 `image_size × image_size`。 |
| `max_text_aug_per_image` |                                    `1` | int | 每張圖最多抽樣幾組 `query_texts_aug`。用於控制文字增強樣本數量。               |

## 資料夾結構

建議資料夾結構如下：

```text
datasets/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

每個 JSON 標記檔應對應一張影像，且 bbox 應使用 `bbox_xyxy` 格式。

---

# 3. train

`train` 區塊控制訓練週期、batch、GPU、AMP、EMA 等設定。

| 參數               |      預設值 | 型別    | 說明                                     |
| ---------------- | -------: | ----- | -------------------------------------- |
| `epochs`         |    `300` | int   | 總訓練 epoch 數。                           |
| `batch_size`     |     `32` | int   | 每次訓練 batch 大小。                         |
| `warmup_epochs`  |      `5` | int   | learning rate warmup 的 epoch 數。        |
| `num_workers`    |     `18` | int   | dataloader worker 數量。                  |
| `device`         | `cuda:1` | str   | 指定訓練裝置，例如 `cuda:0`、`cuda:1`、`cpu`。     |
| `seed`           |     `42` | int   | 隨機種子，用於提高實驗可重現性。                       |
| `use_amp`        |   `true` | bool  | 是否使用 AMP mixed precision。CUDA 訓練時建議開啟。 |
| `use_ema`        |   `true` | bool  | 是否使用 EMA model 做驗證與保存最佳權重。             |
| `ema_decay`      |  `0.999` | float | EMA 更新係數。越接近 `1.0`，EMA 模型變化越慢。         |
| `grad_clip_norm` |    `1.0` | float | gradient clipping 上限。可降低訓練初期梯度爆炸風險。    |

## batch_size 建議

| GPU VRAM | 建議 batch_size |
| -------: | ------------: |
|     8 GB |       `4 ~ 8` |
|    12 GB |      `8 ~ 16` |
|    24 GB |     `16 ~ 32` |
|    32 GB |    `32` 以上可嘗試 |

如果出現 CUDA out of memory，優先降低：

```yaml
train:
  batch_size: 16
```

其次降低：

```yaml
model:
  image_grid_size: 10
  num_layers: 2
```

---

# 4. optim

`optim` 區塊控制 optimizer 與 learning rate schedule。

| 參數                 |       預設值 | 型別    | 說明                                                                              |
| ------------------ | --------: | ----- | ------------------------------------------------------------------------------- |
| `lr_vision`        |  `0.0001` | float | 影像端模組 learning rate，例如 `BottleNet`、`ImgProjector`。                              |
| `lr_text`          | `0.00001` | float | BERT 相關模組 learning rate。若 `freeze_bert=true`，主要影響文字 projection。                 |
| `lr_transformer`   |  `0.0001` | float | Transformer fusion block learning rate。                                         |
| `lr_head`          |  `0.0001` | float | DenseHead learning rate。                                                        |
| `weight_decay`     |  `0.0001` | float | AdamW weight decay，用於正則化。                                                       |
| `min_lr_ratio`     |    `0.05` | float | cosine scheduler 最低 learning rate 比例。                                           |
| `max_warmup_steps` |    `3000` | int   | warmup 最大 step 數。實際 warmup step 會取 `warmup_epochs × len(train_loader)` 與此值的較小值。 |

## learning rate 分組邏輯

目前 optimizer 會依照模組名稱分配 learning rate：

| 模組                        | 使用參數             |
| ------------------------- | ---------------- |
| `bottle_net`, `img_model` | `lr_vision`      |
| `text_model.model`        | `lr_text`        |
| `transformer`             | `lr_transformer` |
| `head`                    | `lr_head`        |

若 BERT 凍結，BERT 本體不會更新，但文字 projection 仍會訓練。

---

# 5. loss

`loss` 區塊控制 Hungarian matching 與 loss 權重相關設定。

目前 loss 組成為：

```text
loss = λ_bbox × L1 + λ_giou × GIoU + λ_score × BCE
```

其中：

| Loss   | 說明                                       |
| ------ | ---------------------------------------- |
| `L1`   | 預測 bbox 與 GT bbox 的座標距離                  |
| `GIoU` | bbox 幾何重疊品質                              |
| `BCE`  | 每個 image token 是否對應 query 目標的 score loss |

| 參數               |   預設值 | 型別    | 說明                                    |
| ---------------- | ----: | ----- | ------------------------------------- |
| `cost_bbox`      | `5.0` | float | Hungarian matching 時 bbox L1 cost 權重。 |
| `cost_giou`      | `2.0` | float | Hungarian matching 時 GIoU cost 權重。    |
| `cost_score`     | `1.0` | float | Hungarian matching 時 score cost 權重。   |
| `min_pos_weight` | `1.0` | float | BCE positive sample 最小權重。             |
| `max_pos_weight` | `5.0` | float | BCE positive sample 最大權重。             |

## cost 與 loss 的差異

`cost_*` 用於 Hungarian matching，決定哪個 prediction 對應哪個 GT。

真正反向傳播的 loss 權重則由訓練腳本中的動態 schedule 控制，例如：

```text
lambda_bbox
lambda_giou
lambda_score
```

`cost_bbox`、`cost_giou`、`cost_score` 不直接等於最終 loss 權重，但會影響匹配結果。

## pos_weight

因為 image token 很多，但正樣本通常很少，例如：

```text
100 個 image tokens 中，可能只有 1~5 個是正樣本
```

所以 `pos_weight` 用於提高正樣本在 BCE 中的重要性。

若模型初期完全不出框或 score 很低，可以提高：

```yaml
loss:
  max_pos_weight: 8.0
```

若 false positive 太多，可以降低：

```yaml
loss:
  max_pos_weight: 3.0
```

---

# 6. eval

`eval` 區塊控制驗證頻率與物件偵測指標計算方式。

| 參數                  |     預設值 | 型別    | 說明                                          |
| ------------------- | ------: | ----- | ------------------------------------------- |
| `val_loss_interval` |     `1` | int   | 每幾個 epoch 計算一次 validation loss。             |
| `eval_interval`     |     `1` | int   | 每幾個 epoch 計算一次 mAP / precision / recall。    |
| `max_val_batches`   |   `320` | int   | 每次驗證最多使用幾個 validation batch。若資料很大，可降低以加快驗證。 |
| `score_thr`         |  `0.25` | float | 推論時保留 prediction 的 score threshold。         |
| `top_k`             |    `20` | int   | 每張圖最多保留前 K 個 prediction。                    |
| `nms_iou_thr`       |   `0.5` | float | NMS IoU threshold。                          |
| `best_metric`       | `map50` | str   | 用於保存最佳 checkpoint 的指標。                      |

## 評估指標

目前評估是 query-conditioned binary detection，不是多類別 COCO mAP。

每筆樣本的語意是：

```text
image + query_text -> 找出符合 query 的所有 bbox
```

因此指標代表：

| 指標          | 說明                                       |
| ----------- | ---------------------------------------- |
| `map50`     | IoU threshold = 0.5 時的 Average Precision |
| `map50_95`  | IoU threshold 從 0.5 到 0.95 的平均 AP        |
| `precision` | 被模型選出的框中，有多少比例是正確命中                      |
| `recall`    | GT 目標中，有多少比例被模型找出                        |
| `tp`        | true positive 數量                         |
| `fp`        | false positive 數量                        |
| `num_gt`    | GT bbox 總數                               |
| `num_pred`  | 預測 bbox 總數                               |

## score_thr 調整建議

| 現象                | 調整方式                         |
| ----------------- | ---------------------------- |
| false positive 太多 | 提高 `score_thr`               |
| recall 太低         | 降低 `score_thr`               |
| 預測框太多             | 降低 `top_k` 或降低 `nms_iou_thr` |
| 預測框被 NMS 過度移除     | 提高 `nms_iou_thr`             |

---

# 7. log

`log` 區塊控制 checkpoint 與訓練紀錄輸出。

| 參數                    |    預設值 | 型別         | 說明                                     |
| --------------------- | -----: | ---------- | -------------------------------------- |
| `save_dir`            | `null` | str 或 null | checkpoint 儲存位置。若為 `null`，會自動建立時間戳資料夾。 |
| `resume_path`         | `null` | str 或 null | 要恢復訓練的 checkpoint 路徑。                  |
| `save_epoch_interval` |   `50` | int        | 每幾個 epoch 額外保存一次 checkpoint。           |
| `emit_step_metrics`   | `true` | bool       | 是否輸出 step-level metrics。適合外部監聽訓練曲線。    |
| `log_interval`        |   `10` | int        | 每幾個 step 寫入一次 step-level metrics。      |

## 輸出檔案

訓練時會在 `save_dir` 下輸出：

```text
latest.pt
best_map50.pt
metrics_step.jsonl
metrics_epoch.jsonl
metrics_epoch.csv
latest_metrics.json
```

| 檔案                    | 說明                                                |
| --------------------- | ------------------------------------------------- |
| `latest.pt`           | 最新 checkpoint                                     |
| `best_map50.pt`       | 根據 `best_metric` 保存的最佳 checkpoint                 |
| `metrics_step.jsonl`  | step-level loss 紀錄                                |
| `metrics_epoch.jsonl` | epoch-level loss / mAP / precision / recall       |
| `metrics_epoch.csv`   | 與 `metrics_epoch.jsonl` 相同，但方便用 pandas / Excel 讀取 |
| `latest_metrics.json` | 最新一筆 epoch 指標，適合外部監聽程式讀取                          |

---

# 8. 常用設定建議

## 低 VRAM 設定

```yaml
model:
  num_layers: 2
  image_grid_size: 10

train:
  batch_size: 8
  use_amp: true
```

## 提高多目標能力

```yaml
model:
  image_grid_size: 20

eval:
  top_k: 50
```

## 降低 false positive

```yaml
eval:
  score_thr: 0.4
  top_k: 10

loss:
  max_pos_weight: 3.0
```

## 提高 recall

```yaml
eval:
  score_thr: 0.15
  top_k: 50

loss:
  max_pos_weight: 5.0
```

## 訓練不穩定時

```yaml
train:
  grad_clip_norm: 1.0

optim:
  lr_vision: 0.00005
  lr_transformer: 0.00005
  lr_head: 0.00005
```

---

# 9. 執行方式

使用 YAML 訓練：

```bash
python3 train_card.py --config cards/config/model.yaml
```

臨時覆蓋 batch size：

```bash
python3 train_card.py \
  --config cards/config/model.yaml \
  --batch-size 16
```

指定 GPU：

```bash
python3 train_card.py \
  --config cards/config/model.yaml \
  --device cuda:0
```

恢復訓練：

```bash
python3 train_card.py \
  --config cards/config/model.yaml \
  --resume-path checkpoints/results_xxxx/latest.pt
```

---

# 10. 注意事項

1. GT bbox 必須是 normalized `xyxy` 或在 dataloader 中轉成 normalized `xyxy`。
2. 模型輸出的 bbox 必須是 normalized `xyxy`。
3. `score_logit` 應直接丟給 `BCEWithLogitsLoss`，不要先做 sigmoid。
4. 推論與評估時才對 `score_logit` 做 sigmoid。
5. 目前不需要 class label prediction。
6. 目前 mAP / precision / recall 是文字查詢條件下的 binary detection 指標，不是多類別分類指標。
