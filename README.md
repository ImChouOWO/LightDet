# LightDet-gRef

![Framework](https://img.shields.io/badge/framework-PyTorch-EE4C2C)
![Backbone](https://img.shields.io/badge/backbone-ResNet--50-2F80ED)
![Dataset](https://img.shields.io/badge/dataset-gRefCOCO-4CAF50)
![Text](https://img.shields.io/badge/text-Multilingual%20MiniLM-8A2BE2)

文字引導的零／單／多目標物件定位。此分支使用 gRefCOCO、Multilingual MiniLM 與文字條件化的密集 CNN，移除原 Localization Transformer 與舊船舶資料流程。

```text
Image + Expression -> Text-conditioned Dense CNN -> BBox + ROI Scores -> Target Set
```

<details>
<summary><em>當前限制 / Current Limitations</em></summary>

此版本尚未完成正式訓練，沒有可報告的模型績效。gRefCOCO 的官方 val 標記沒有單目標樣本，因此該切分的 Acc@0.5 為 N/A；testA／testB 保留單目標評估。

---

This branch has not completed training. The downloaded official gRefCOCO validation annotations contain no single-target references, so single-target Acc@0.5 is N/A on val and is reported on testA/testB.

</details>

---

## 簡介

### 模型能力

輸入英文或中文描述，預測符合描述的物件集合；無目標描述允許輸出空集合。公開訓練描述為英文，中英 grounding 的實際泛化仍需另外驗證。

| 官方切分 | 訓練／評估單位 | 單目標 | 多目標 | 無目標 |
|---|---:|---:|---:|---:|
| train | 78,930 references，每 epoch 輪替一句 | 42,404 | 25,060 | 11,466 |
| val | 14,229 sentences，全部評估 | 0 | 5,324 | 8,905 |
| testA | 19,200 sentences，全部評估 | 5,917 | 8,835 | 4,448 |
| testB | 16,063 sentences，全部評估 | 5,646 | 5,744 | 4,673 |

### 模型資料流

```text
Image -> ResNet-50 -> FPN Maps ---------------------------> ROI Align
                         |                                    ^
Expression -> MiniLM -> Text Tokens + Phrase Feature           |
                         |                                    |
                  Spatial Text/Image Fusion                   |
                         |                                    |
                    Dense CNN -> Dense Boxes -> Top-100 Boxes -+
                                                    |
                                    Local ROI Refinement
                                                    |
                           Quality / Alignment / Selection
                                                    |
                              Score Threshold -> NMS -> Target Set
```

### 模型結構

| 模組 | 功能 |
|---|---|
| ResNet-50／FPN | ImageNet 預訓練骨幹；保留 stride 8／16／32 空間特徵，固定 BN 統計 |
| Multilingual MiniLM | 凍結多語編碼器；token 特徵 384→256，保留可訓練文字投影 |
| Spatial Fusion | 影像位置編碼、token 相似度加權文字與 FiLM 條件融合 |
| Dense Localization CNN | 三層密集連接卷積，在每個 FPN 位置預測框中心、寬高及描述目標分數；沒有 learnable localization queries |
| ROI Align／Refinement | 每個候選取 8×8 FPN ROI，再執行 ROI 自身的局部注意力 |
| Quality | 預測候選框與已標記 COCO 物件的最大 IoU |
| Alignment | ROI 與描述的正規化相似度，判斷是否符合描述 |
| Selection | 描述條件下的保留／丟棄分數，對一對一代表框監督，抑制重複輸出 |

先融合 `sqrt(sigmoid(Quality) × sigmoid(Alignment))`，轉成 logit 後加上 Selection 的校準殘差，最後用 sigmoid 得到保留分數。Quality 在這條監督路徑停止梯度，Selection loss 與 Ranking loss 都使用最終分數。多段描述推論會重用同一份 FPN 特徵。

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

訓練每 epoch 訪問每個 reference 一次，依 epoch 輪替句子。每張圖最多四段描述先組為區塊，再隨機組批；所有目標類型保留原始分布，不重複抽樣、不丟棄最後一批。同圖描述在 batch 中共用一次 ResNet／FPN 計算；合併 tensor 傳輸並快取近期縮放影像。GT 的密集幾何標籤在 CPU workers 建立，不於 loss 中逐樣本同步 GPU。驗證使用全部 sentences。

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
  dense_growth: 64
  dense_depth: 3
  candidate_count: 100
  roi_size: 8
```

---

## 訓練設定

設定檔：`units/model/cards/config/train.yaml`。訓練 100 epoch、batch size 64、640 影像，使用 BF16、梯度裁切及 最多 250 step LR 暖機（維持約 16,000 組描述的暖機量）。預訓練 ResNet 使用較低學習率；新 FPN／定位／評分 heads 使用較高學習率。

初期 5 epoch 保留靠近 GT 中心的預測候選；GT ROI 語意輔助權重前 5 epoch 為 0.5，第 15 epoch 歸零。GT 引導僅用於訓練，驗證與推論完全使用模型候選。

---

## Loss 與 Matching

| Loss | 監督 |
|---|---|
| Dense classification | GT 內的中心區域為描述目標正例，無目標描述全部為負例；排除補邊區域 |
| BBox／GIoU | 密集正例及 Hungarian 一對一配對的定位回歸 |
| Quality | 預測框對已標記 COCO 物件的最大 IoU |
| Alignment | 符合描述的候選為正例，同物件重複候選仍可保留語意正例 |
| Selection | 每個 GT 的一對一代表候選為正例，其餘有效候選為丟棄目標；同物件且會被 NMS 移除的重複候選忽略，以免相同 ROI 接收矛盾標籤 |
| Ranking | 比較保留候選與高分負例的最終 log 分數；Quality 停止梯度 |
| GT ROI 暖機 | 直接從 GT 框取得 ROI，輔助早期語意學習 |

Matching 成本整批回傳 CPU，逐筆求解後一次傳回配對；逐框 GIoU 直接計算，不建立完整成對矩陣。GT ROI 與一般候選合併評分。Matching 只在 loss 中配對候選與 GT，沒有 Localization Transformer、Main／Aux queries 或舊顏色關係矩陣。

---

## 啟動訓練

```bash
python3 units/model/train.py --smoke-test
python3 units/model/train.py
```

伺服器的 `_train.py` 使用相同程式；不自動啟動完整訓練。

---

## Checkpoint 與續訓

```bash
python3 units/model/train.py \
  --resume units/model/runs/train/lightdet_gref_latest.pt
```

此分支只接受 `lightdet.grefcoco.v1` checkpoint。原版權重的任務模組不相容；新模型僅從 ImageNet ResNet 與預訓練 MiniLM 初始化。另開實驗時修改 `log.prefix`，避免覆寫既有訓練。

---

## 驗證與後處理

```bash
python3 units/model/val.py \
  --weights units/model/runs/train/lightdet_gref_best.pt --split val
python3 units/model/val.py \
  --weights units/model/runs/train/lightdet_gref_best.pt --split testA
```

| 指標 | 定義 |
|---|---|
| GREC Pr@(F1=1) | 描述的完整輸出集合與全部 GT 一對一命中，沒有漏報或多報；作為最佳 checkpoint 指標 |
| 單目標 Acc@0.5 | 僅對單目標描述，最高分有效候選與 GT 的 IoU >0.5 |
| N-acc | 明確無目標描述是否輸出空集合 |
| T-acc | 有目標描述是否輸出非空集合，不代表定位正確 |

集合評估與推論統一使用分數 ≥0.5、NMS IoU=0.5、最多 100 框；集合匹配使用 IoU ≥0.5 的最大一對一配對。gRefCOCO 每段描述最多有 18 個 GT，不使用固定 Top-10 截斷。Acc@0.5 取最高分有效候選，不以 GT 在 Top-10 中挑選答案，也不受集合分數閾值限制。

此流程使用 GREC 指標定義，但後處理與官方參考程式的 0.7／無 NMS 設定不同；不能直接比較不同協定的論文數值。val 的 Acc@0.5 為 N/A，不以 0 取代。不再輸出 Hit@K、TextHit、Oracle 或舊 mAP。

多描述推論：

```bash
python3 units/infer.py --weights units/model/runs/train/lightdet_gref_best.pt \
  --image /PATH/TO/image.jpg --text "the red boat" "紅色的船"
```

---

## 訓練輸出

| 檔案 | 功能 |
|---|---|
| `lightdet_gref_latest.pt` | 完整續訓狀態 |
| `lightdet_gref_best.pt` | 最佳 GREC 集合成功率權重 |
| `lightdet_gref_epoch_XXX.pt` | 每 10 epoch 保存 |
| `lightdet_gref_metrics.jsonl` | 每個 epoch 的 loss 與統一指標 |

全部輸出位於既有 `units/model/runs/train`。資料、編碼器與模型權重不提交 Git。

---

## 參考資料

- [gRefCOCO 官方資料與 API](https://github.com/henghuiding/gRefCOCO)
- [GREC 論文](https://arxiv.org/abs/2308.16182)
- [Multilingual MiniLM](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)

## License

本專案採用 [MIT License](./LICENSE)。使用或再散布時保留原作者 **Cheng-Han Chou** 的版權聲明及授權條款。外部資料集與預訓練模型依其各自授權使用。
