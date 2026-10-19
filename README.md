# LightDet-gRef

![Framework](https://img.shields.io/badge/framework-PyTorch-EE4C2C)
![Architecture](https://img.shields.io/badge/architecture-Transformer%20Queries%20%2B%20FPN%20%2B%20ROI-8A2BE2)
![Modality](https://img.shields.io/badge/input-Text%20%2B%20Image-2F80ED)
![Backbone](https://img.shields.io/badge/backbone-ResNet--50-2F80ED)
![Dataset](https://img.shields.io/badge/dataset-gRefCOCO-4CAF50)
![Text](https://img.shields.io/badge/text-Multilingual%20MiniLM-8A2BE2)
![Precision](https://img.shields.io/badge/precision-BF16%20AMP-FF9800)
![Status](https://img.shields.io/badge/status-active%20research-yellow)

文字引導的零／單／多目標物件定位。此分支使用 gRefCOCO、Multilingual MiniLM 與 Transformer object queries；Main 採 one-to-one 集合預測，Aux 可選擇以 one-to-many 提供訓練輔助。

```text
Image + Expression -> Transformer Object Queries -> BBox + ROI Scores -> Target Set
```

<details>
<summary><em>當前限制 / Current Limitations</em></summary>

此版本正在訓練，尚未完成完整訓練與 testA／testB 評估。gRefCOCO 的官方 val 標記沒有單目標樣本，因此該切分的 Acc@0.5 為 N/A；testA／testB 保留單目標評估。

---

Training is ongoing; full training and testA/testB evaluation are not yet complete. The downloaded official gRefCOCO validation annotations contain no single-target references, so single-target Acc@0.5 is N/A on val and is reported on testA/testB.

</details>

---

## 簡介

### 模型能力

輸入英文或中文描述，預測符合描述的物件集合；無目標描述允許輸出空集合。公開訓練描述為英文，中英 grounding 的實際泛化仍需另外驗證。



| Model | Encoder | val | testA | testB |
|---|---|---:|---:|---:|
| ReLA | ResNet-50＋BERT | 59.36% | 48.09% | 42.85% |
| SSDE | BEiT-3 ViT-B | 74.17% | 71.19% | 60.62% |
| LightDet | ResNet-50＋MiniLM | 62.79% | 44.69% | 41.03% |

> 驗證指標：GREC Pr@(F1=1)  
> 參考資料：[GREx／ReLA，Table 14](https://arxiv.org/html/2601.05244v1#S5.SS4)、[SSDE，Table 8](https://arxiv.org/html/2608.30233#S4.SS2)。



### 模型資料流

![模型能力](https://github.com/ImChouOWO/LightDet/blob/LightDet-gRef/dataFlow.png)




>Main／Aux 分別使用 10／100 個獨立 query 與輸出 Head，共用文字 Cross-Attention 與 Decoder；兩組 query 的 self-attention 隔離。初始化為 `CrossAttention(Zero + Learnable Query, Text, Text) + Learnable Query + Projection(Mean-pooled Text)`

### 模型結構
![模型能力](https://github.com/ImChouOWO/LightDet/blob/LightDet-gRef/LightDet.png)

| 模組 | 說明 |
|---|---|
| `ResNet-50 / FPN` | ImageNet 骨幹與三尺度 256 通道特徵，固定 BN 統計；同圖描述共用影像計算 |
| `Multilingual MiniLM` | 凍結文字編碼器，投影 384→256；保留 token 特徵與描述池化特徵 |
| `Object Query Transformer` | 先以 8-head 文字 Cross-Attention 引導 Main 10／Aux 100 個 queries，加回 learnable query 與整句文字投影；再以 FPN memory grids 20／10／5 與文字 tokens，經兩層 8-head Decoder 更新 |
| `BBox Head` | 每組 query 經獨立 MLP 預測中心與寬高，限制於有效影像範圍 |
| `ROI Align / Refinement` | 由原始 FPN 擷取 8×8 ROI，ROI mean＋BBox embedding 對自身 tokens 執行局部 refinement |
| `Quality Head` | 原始 ROI mean 預測與已標記 COCO 物件的最大 IoU |
| `Alignment Head` | Refined ROI 與文字投影後的正規化相似度 |
| `Selection Head` | Refined ROI 加上 Decoder query conditioning，再與文字拼接，產生保留分數的校準殘差；讓不同 query 可區分重複 ROI |


## TGQ(Text-Guided Query)

![TGQ](https://github.com/ImChouOWO/LightDet/blob/LightDet-gRef/TQG.png)

**評分輸出與融合**

| 項目 | 功用 |
|---|---|
| **Quality Logits** | 預測框的定位品質；經 sigmoid 轉成 Quality 機率，學習與已標記 COCO 物件的最大 IoU。 |
| **Alignment Logits** | 評估 ROI 與輸入描述的語意匹配程度；經 sigmoid 轉成 Alignment 機率。 |
| **Selection Residual** | 根據 ROI 與文字的聯合特徵，學習修正基礎融合分數；它本身不是機率。 |
| **Score Fusion** | 先計算 `base = √(Quality × Alignment)`，再計算 `logit(base) + Selection Residual`。 |
| **logit** | 將機率轉為實數分數：`log(p / (1-p))`，讓 Residual 能以加法修正。 |
| **Selection Logits** | 融合後的最終分數，供 Selection Loss 使用；經 sigmoid 得到 Final Score，Ranking Loss 比較其 log 分數。 |



---

## 專案結構

```text
LightDet/
├── datasets/
│   ├── images/                 # COCO JPEG，直接存於既有目錄
│   └── labels/                 # grefs(unc).json、instances.json
├── units/
│   ├── infer.py
│   ├── model/
│   │   ├── bert/               # MiniLM 本機資源
│   │   ├── resnet/checkpoints/
│   │   ├── cards/{loss.py,config/}
│   │   ├── pipeline/data.py
│   │   ├── tool/{runtime.py,evaluation.py,grounding_metrics.py,config.py}
│   │   ├── runs/train/         # 以檔名前綴隔離實驗，不建立新資料夾
│   │   ├── train.py
│   │   └── val.py
│   └── tool/card.py
└── prepare_grefcoco.py
```

---

## 安裝

### 1. 下載專案

```bash
git clone --branch LightDet-gRef https://github.com/ImChouOWO/LightDet.git
cd LightDet
```

### 2. 建立虛擬環境

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

伺服器可直接使用既有 venv。PyTorch CUDA 套件需與執行環境相容。

### 3. 下載資料與編碼器

```bash
python3 prepare_grefcoco.py --workers 32
python3 units/tool/getResNet.py
```

下載官方 gRefCOCO JSON、其引用的 19,994 張 COCO train2014 影像與 MiniLM。已下載檔案會重用，不新增資料夾；圖片從 COCO 官方 S3 儲存桶取得。

---

## 資料集格式

直接讀取原生 gRefCOCO 標記，不轉成舊 relation JSON：

```json
{
  "ref_id": 0,
  "image_id": 72,
  "ann_id": [600276],
  "no_target": false,
  "split": "train",
  "sentences": [
    {"sent_id": 0, "sent": "giraffe on left"}
  ]
}
```

| 欄位 | 功能 |
|---|---|
| `ref_id`／`sent_id` | 保留官方 reference／sentence 識別 |
| `ann_id` | 從 COCO instances 查詢對應的零到多個 GT 框 |
| `no_target` | 明確的無目標監督；不以未標記物件推導 |
| `split` | 使用官方 train／val／testA／testB，禁止跨切分抽樣 |
| `sentences` | 每個 reference 的自然語言描述 |

影像等比例縮放至 640 並向右／下補邊；BBox 同步轉為正規化 xyxy。關係描述不做水平翻轉或顏色擾動。

---

## 資料取樣

訓練每 epoch 訪問每個 reference 一次，依 epoch 輪替句子。每張圖最多四段描述先組為區塊，再隨機組批；所有目標類型保留原始分布，不重複抽樣、不丟棄最後一批。同圖描述在 batch 中共用一次 ResNet／FPN 計算；合併 tensor 傳輸並快取近期縮放影像。Main／Aux 匹配各批次僅集中傳輸一次成本矩陣。驗證使用全部 sentences。

---

## 文字編碼

使用 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` 的 token backbone，搭配原模型 tokenizer；不沿用 MacBERT 快取。輸入超過設定 token 長度會明確報錯，不靜默截斷。

---

## 模型設定

設定檔：`units/model/cards/config/model.yaml`。

```yaml
model:
  hidden_dim: 256
  freeze_text: true
  num_object_queries: 10
  num_aux_queries: 100
  query_heads: 8
  query_layers: 2
  memory_grids: [20, 10, 5]
  roi_size: 8
```

---

## 訓練設定

設定檔：`units/model/cards/config/train.yaml`。本次實驗訓練 30 epoch、batch size 64、640 影像，使用 BF16、梯度裁切及 最多 250 step LR 暖機（維持約 16,000 組描述的暖機量）。預訓練 ResNet 使用較低學習率；新 FPN／Query Decoder／評分 heads 使用較高學習率。

GT ROI 語意輔助權重前 5 epoch 為 0.5，第 15 epoch 歸零。GT 引導僅用於訓練，驗證與推論完全使用模型候選。

---

## Loss 與 Matching

Main 使用 Hungarian one-to-one。Aux 先保留符合 IoU ≥0.3 的 Hungarian 配對，再依成本補入最多 3 個候選；每個候選僅屬於一個 GT，不強制湊滿。

| Loss | 監督 |
|---|---|
| BBox／GIoU | 各分支的已匹配候選回歸 GT；Aux 同物件正例以候選數倒數加權 |
| Quality | 預測框對全部已標記 COCO 物件的最大 IoU |
| Alignment | 已匹配候選與 IoU ≥0.5 的語意候選為正例 |
| Selection | Main 只有一對一代表候選為正例，所有未配對候選為 no-object；Aux 使用自己的 one-to-many 標籤 |
| Ranking | 正例與高分未配對候選比較最終 log 分數 |
| Threshold margin | 以 `logit(0.7)` 為邊界；前 5 epoch 停用，第 6–15 epoch 將 margin／權重增加至 0.5／0.1 |
| GT ROI 暖機 | GT ROI 提供 Main 早期 Alignment 輔助監督，第 15 epoch 歸零 |

`Total Loss = 0.8 × Main Loss + 0.2 × schedule × Aux Loss`。Aux 前 5 epoch 停用，第 6–15 epoch 暖機，第 24–30 epoch 衰減至 0；關閉 Aux 時不計算 Aux queries／Head，Main 權重仍為 0.8。Aux 的 NMS 僅用於輸出檢查，不參與 matching 或 Loss。


---

## 啟動訓練

```bash
python3 units/model/train.py --no-aux --smoke-test
python3 units/model/train.py --no-aux
```

伺服器的 `_train.py` 使用相同程式。`train.aux_enabled` 預設為 false，可用 `--aux`／`--no-aux` 選擇；另開實驗需使用新的 `log.prefix`。

---

## Checkpoint 與續訓

```bash
python3 units/model/train.py \
  --resume units/model/runs/train/lightdet_gref_tqg_main10_30ep_latest.pt
```

此分支只接受 `lightdet.grefcoco.queries.v3` checkpoint。舊 Query／Dense CNN／原版權重不相容；新模型僅從 ImageNet ResNet 與預訓練 MiniLM 初始化。另開實驗時修改 `log.prefix`，避免覆寫既有訓練。

---

## 驗證與後處理

```bash
python3 units/model/val.py \
  --weights units/model/runs/train/lightdet_gref_tqg_main10_30ep_best.pt --split val
python3 units/model/val.py \
  --weights units/model/runs/train/lightdet_gref_tqg_main10_30ep_best.pt --split testA
```

| 指標 | 定義 |
|---|---|
| GREC Pr@(F1=1) | 描述的完整輸出集合與全部 GT 一對一命中，沒有漏報或多報；作為最佳 checkpoint 指標 |
| 單目標 Acc@0.5 | 僅對單目標描述，最高分有效候選與 GT 的 IoU >0.5 |
| N-acc | 明確無目標描述是否輸出空集合 |
| T-acc | 有目標描述是否輸出非空集合，不代表定位正確 |

Main 集合評估與推論統一使用分數 ≥0.7、不使用 NMS、最多 100 框。集合匹配與官方參考程式一致，採 GIoU ≥0.5 的最大重疊貪婪一對一配對；不是最大配對數 Hungarian。Acc@0.5 額外評估單目標最高分候選，val 沒有單目標描述時為 N/A。`Positive-Pr` 為有目標描述的完整集合成功率，僅供拆分分析。

官方驗證輸出清楚的設定摘要與分切分百分比表：

```bash
python3 units/model/val_official.py \
  --weights units/model/runs/train/lightdet_gref_tqg_main10_30ep_best.pt \
  --output units/model/runs/train/lightdet_gref_tqg_main10_30ep_official.json
```

多描述推論：

```bash
python3 units/infer.py --weights units/model/runs/train/lightdet_gref_tqg_main10_30ep_best.pt \
  --image /PATH/TO/image.jpg --text "the red boat" "紅色的船"
```

---

## 訓練輸出

| 檔案 | 功能 |
|---|---|
| `lightdet_gref_tqg_main10_30ep_latest.pt` | 完整續訓狀態 |
| `lightdet_gref_tqg_main10_30ep_best.pt` | 最佳 GREC 集合成功率權重 |
| `lightdet_gref_tqg_main10_30ep_epoch_XXX.pt` | 每 10 epoch 保存 |
| `lightdet_gref_tqg_main10_30ep_metrics.jsonl` | 每個 epoch 的 loss 與統一指標 |

全部輸出位於既有 `units/model/runs/train`。資料、編碼器與模型權重不提交 Git。

---

## 參考資料

- [gRefCOCO 官方資料與 API](https://github.com/henghuiding/gRefCOCO)
- [GREC 論文](https://arxiv.org/abs/2308.16182)
- [Multilingual MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

## License

本專案採用 [MIT License](./LICENSE)。使用或再散布時保留原作者 **Cheng-Han Chou** 的版權聲明及授權條款。外部資料集與預訓練模型依其各自授權使用。
