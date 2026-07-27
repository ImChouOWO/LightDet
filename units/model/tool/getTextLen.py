#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
from transformers import BertTokenizerFast


# =========================
# 直接設定路徑
# =========================

LABEL_DIRS = {
    "train": Path(
        "/home/soic/Desktop/LightDet/datasets/labels/train"
    ),
    "val": Path(
        "/home/soic/Desktop/LightDet/datasets/labels/val"
    ),
}

BERT_MODEL_DIR = Path(
    "/home/soic/Desktop/LightDet/units/model/bert"
)

# Train 會附加 negative phrases。
# 目前此數值是估計預留量，不是精確 sampling 後的長度。
TRAIN_NEGATIVE_TOKEN_RESERVE = 32

SHOW_LONGEST_COUNT = 20
SHOW_INVALID_COUNT = 20


def round_up_model_length(
    token_length: int,
) -> int:
    candidates = [
        32,
        48,
        64,
        80,
        96,
        112,
        128,
        160,
        192,
        224,
        256,
        320,
        384,
        512,
    ]

    for value in candidates:
        if value >= token_length:
            return value

    return int(
        math.ceil(token_length / 64)
        * 64
    )


def extract_caption(
    data: Dict[str, Any],
) -> Tuple[str, str]:
    """
    支援兩種 JSON 結構：

    1. caption 位於根節點
       {
           "caption": "..."
       }

    2. caption 位於 grounding
       {
           "grounding": {
               "caption": "..."
           }
       }

    回傳：
        caption
        caption_path
    """
    if not isinstance(data, dict):
        raise TypeError(
            "JSON 根節點必須是 object"
        )

    root_caption = data.get(
        "caption"
    )

    if root_caption is not None:
        caption = str(
            root_caption
        ).strip()

        if caption:
            return (
                caption,
                "caption",
            )

    grounding = data.get(
        "grounding"
    )

    if isinstance(
        grounding,
        dict,
    ):
        grounding_caption = grounding.get(
            "caption"
        )

        if grounding_caption is not None:
            caption = str(
                grounding_caption
            ).strip()

            if caption:
                return (
                    caption,
                    "grounding.caption",
                )

    raise ValueError(
        "缺少 caption 或 grounding.caption"
    )


def read_caption_lengths(
    split_name: str,
    label_dir: Path,
    tokenizer: BertTokenizerFast,
) -> Tuple[
    List[int],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    if not label_dir.is_dir():
        raise FileNotFoundError(
            f"{split_name} 標註資料夾不存在："
            f"{label_dir}"
        )

    json_files = sorted(
        label_dir.rglob("*.json")
    )

    if not json_files:
        raise RuntimeError(
            f"{split_name} 資料夾內沒有 JSON："
            f"{label_dir}"
        )

    lengths: List[int] = []
    records: List[Dict[str, Any]] = []
    invalid_files: List[Dict[str, Any]] = []

    root_caption_count = 0
    grounding_caption_count = 0

    for json_path in json_files:
        try:
            with open(
                json_path,
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            caption, caption_path = (
                extract_caption(data)
            )

            if caption_path == "caption":
                root_caption_count += 1
            elif caption_path == "grounding.caption":
                grounding_caption_count += 1

            encoded = tokenizer(
                caption,
                add_special_tokens=True,
                truncation=False,
            )

            token_length = len(
                encoded["input_ids"]
            )

            lengths.append(
                token_length
            )

            records.append({
                "split": split_name,
                "token_length": token_length,
                "character_length": len(
                    caption
                ),
                "path": str(
                    json_path
                ),
                "caption": caption,
                "caption_path": caption_path,
            })

        except Exception as error:
            invalid_files.append({
                "split": split_name,
                "path": str(
                    json_path
                ),
                "error": str(
                    error
                ),
            })

    if not lengths:
        print(
            f"\n[{split_name}] "
            f"沒有讀取到有效 caption。"
        )

        print(
            f"找到 JSON：{len(json_files)}"
        )

        print(
            f"\n前 {SHOW_INVALID_COUNT} 筆錯誤："
        )

        for item in invalid_files[
            :SHOW_INVALID_COUNT
        ]:
            print(
                f"{item['path']} | "
                f"{item['error']}"
            )

        raise RuntimeError(
            f"{split_name} 沒有有效 caption"
        )

    print(
        f"\n[{split_name}] Caption 來源："
    )
    print(
        f"  caption             : "
        f"{root_caption_count}"
    )
    print(
        f"  grounding.caption   : "
        f"{grounding_caption_count}"
    )
    print(
        f"  invalid             : "
        f"{len(invalid_files)}"
    )

    return (
        lengths,
        records,
        invalid_files,
    )


def print_split_statistics(
    split_name: str,
    lengths: List[int],
) -> None:
    values = np.asarray(
        lengths,
        dtype=np.int32,
    )

    print(
        f"\n========== {split_name} =========="
    )
    print(
        f"有效 caption：{len(values)}"
    )
    print(
        f"平均：{values.mean():.2f}"
    )
    print(
        f"P50："
        f"{np.percentile(values, 50):.0f}"
    )
    print(
        f"P90："
        f"{np.percentile(values, 90):.0f}"
    )
    print(
        f"P95："
        f"{np.percentile(values, 95):.0f}"
    )
    print(
        f"P99："
        f"{np.percentile(values, 99):.0f}"
    )
    print(
        f"最大：{values.max()}"
    )


def main() -> None:
    if not BERT_MODEL_DIR.is_dir():
        raise FileNotFoundError(
            "BERT 模型資料夾不存在："
            f"{BERT_MODEL_DIR}"
        )

    tokenizer = (
        BertTokenizerFast.from_pretrained(
            str(BERT_MODEL_DIR),
            local_files_only=True,
        )
    )

    all_records: List[
        Dict[str, Any]
    ] = []

    all_invalid_files: List[
        Dict[str, Any]
    ] = []

    split_lengths: Dict[
        str,
        List[int],
    ] = {}

    for split_name, label_dir in (
        LABEL_DIRS.items()
    ):
        (
            lengths,
            records,
            invalid_files,
        ) = read_caption_lengths(
            split_name=split_name,
            label_dir=label_dir,
            tokenizer=tokenizer,
        )

        split_lengths[
            split_name
        ] = lengths

        all_records.extend(
            records
        )

        all_invalid_files.extend(
            invalid_files
        )

        print_split_statistics(
            split_name=split_name,
            lengths=lengths,
        )

    if "train" not in split_lengths:
        raise KeyError(
            "LABEL_DIRS 必須包含 train"
        )

    if "val" not in split_lengths:
        raise KeyError(
            "LABEL_DIRS 必須包含 val"
        )

    train_max = max(
        split_lengths["train"]
    )

    val_max = max(
        split_lengths["val"]
    )

    # Train 需要額外保留 negative phrase token。
    train_required = (
        train_max
        + int(
            TRAIN_NEGATIVE_TOKEN_RESERVE
        )
    )

    # Validation 目前不加入 negative phrase。
    val_required = val_max

    exact_required = max(
        train_required,
        val_required,
    )

    recommended = round_up_model_length(
        exact_required
    )

    print(
        "\n========== 最終需求 =========="
    )

    print(
        f"Train 原始最大："
        f"{train_max}"
    )

    print(
        f"Train negative 預留："
        f"{TRAIN_NEGATIVE_TOKEN_RESERVE}"
    )

    print(
        f"Train 最低估計："
        f"{train_required}"
    )

    print(
        f"Val 最低需求："
        f"{val_required}"
    )

    print(
        f"整體最低需求："
        f"{exact_required}"
    )

    print(
        f"建議 text_max_length："
        f"{recommended}"
    )

    print(
        "\nmodel.yaml："
    )

    print(
        "model:"
    )

    print(
        f"  text_max_length: "
        f"{recommended}"
    )

    print(
        f"\n最長的 "
        f"{SHOW_LONGEST_COUNT} 筆："
    )

    longest_records = sorted(
        all_records,
        key=lambda item: item[
            "token_length"
        ],
        reverse=True,
    )[:SHOW_LONGEST_COUNT]

    for index, item in enumerate(
        longest_records,
        start=1,
    ):
        print(
            f"\n[{index:02d}] "
            f"split={item['split']}, "
            f"tokens={item['token_length']}, "
            f"chars={item['character_length']}, "
            f"source={item['caption_path']}"
        )

        print(
            f"path：{item['path']}"
        )

        print(
            f"caption：{item['caption']}"
        )

    if all_invalid_files:
        print(
            f"\n========== 無效檔案 =========="
        )

        print(
            f"總數："
            f"{len(all_invalid_files)}"
        )

        for item in all_invalid_files[
            :SHOW_INVALID_COUNT
        ]:
            print(
                f"{item['split']} | "
                f"{item['path']} | "
                f"{item['error']}"
            )


if __name__ == "__main__":
    main()