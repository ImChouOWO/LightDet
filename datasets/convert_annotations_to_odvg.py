#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
將 LightDet 目前的 object-centric JSON 標記轉換為
ODVG Phrase Grounding 風格的 image-centric JSON / JSONL。

支援：
1. 單一 JSON 檔案轉換
2. 資料夾遞迴批次轉換
3. tqdm 進度顯示
4. 顏色名稱正規化
5. 相同語意 phrase 合併所有合格 bbox
6. 自動建立 caption 與 tokens_positive
7. 輸出前完整驗證
8. 可選擇輸出逐檔 JSON 或整合 JSONL

安裝：
    pip install tqdm

單檔測試：
    python convert_annotations_to_odvg.py \
        --input /path/to/label.json \
        --output /path/to/output.json

單檔只預覽、不寫檔：
    python convert_annotations_to_odvg.py \
        --input /path/to/label.json \
        --dry-run

資料夾遞迴轉換：
    python convert_annotations_to_odvg.py \
        --input /path/to/labels/train \
        --output /path/to/labels_odvg/train

資料夾整合成單一 JSONL：
    python convert_annotations_to_odvg.py \
        --input /path/to/labels/train \
        --output /path/to/train_odvg.jsonl \
        --format jsonl

搭配影像資料夾解析真實影像檔名：
    python convert_annotations_to_odvg.py \
        --input /path/to/labels/train \
        --output /path/to/labels_odvg/train \
        --image-dir /path/to/images/train
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm


BBox = Tuple[float, float, float, float]


COLOR_ALIASES: Dict[str, str] = {
    "白": "白色",
    "白色": "白色",
    "黑": "黑色",
    "黑色": "黑色",
    "灰": "灰色",
    "灰色": "灰色",
    "銀灰": "灰色",
    "紅": "紅色",
    "紅色": "紅色",
    "橙": "橘色",
    "橙色": "橘色",
    "橘": "橘色",
    "橘色": "橘色",
    "黃": "黃色",
    "黃色": "黃色",
    "綠": "綠色",
    "綠色": "綠色",
    "藍": "藍色",
    "藍色": "藍色",
    "紫": "紫色",
    "紫色": "紫色",
    "棕": "棕色",
    "棕色": "棕色",
    "褐": "棕色",
    "褐色": "棕色",
    "銀": "銀色",
    "銀色": "銀色",
    "金": "金色",
    "金色": "金色",
}

COLOR_ORDER: Dict[str, int] = {
    color: index
    for index, color in enumerate(
        [
            "白色",
            "黑色",
            "灰色",
            "紅色",
            "橘色",
            "黃色",
            "綠色",
            "藍色",
            "紫色",
            "棕色",
            "銀色",
            "金色",
        ]
    )
}

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
)


@dataclass
class PhraseGroup:
    semantic_key: str
    phrase: str
    group_type: str
    bboxes: List[BBox] = field(default_factory=list)
    variants: List[str] = field(default_factory=list)

    def add_bbox(self, bbox: BBox) -> None:
        if bbox not in self.bboxes:
            self.bboxes.append(bbox)

    def add_variants(self, values: Iterable[str]) -> None:
        seen = set(self.variants)
        for value in values:
            text = clean_text(value)
            if not text or text == self.phrase or text in seen:
                continue
            self.variants.append(text)
            seen.add(text)


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def normalize_color(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return COLOR_ALIASES.get(text, text)


def normalize_colors(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []

    result: List[str] = []
    seen = set()

    for value in values:
        color = normalize_color(value)
        if not color or color in seen:
            continue
        result.append(color)
        seen.add(color)

    return result


def sort_colors(colors: Sequence[str]) -> List[str]:
    return sorted(
        colors,
        key=lambda color: (
            COLOR_ORDER.get(color, len(COLOR_ORDER)),
            color,
        ),
    )


def colors_key(colors: Sequence[str]) -> str:
    return "|".join(sort_colors(colors))


def short_color_name(color: str) -> str:
    return color[:-1] if color.endswith("色") else color


def build_multicolor_phrase(
    colors: Sequence[str],
    original_query: str = "",
) -> str:
    """
    多色物件優先保留原始明確描述，例如「黃綠白相間的船」。
    若原始描述不存在，才依顏色組合自動產生。
    """
    original_query = clean_text(original_query)
    if original_query:
        return original_query

    compact = "".join(short_color_name(color) for color in colors)
    return f"{compact}相間的船"


def build_exact_variants(obj: Dict[str, Any]) -> List[str]:
    variants: List[str] = []

    query_text = clean_text(obj.get("query_text", ""))
    if query_text:
        variants.append(query_text)

    query_texts_aug = obj.get("query_texts_aug", [])
    if isinstance(query_texts_aug, list):
        variants.extend(
            clean_text(text)
            for text in query_texts_aug
            if clean_text(text)
        )

    return list(dict.fromkeys(variants))


def build_contains_variants(color: str) -> List[str]:
    return [
        f"帶有{color}的船",
        f"船體包含{color}的船",
        f"有{color}部分的船",
    ]


def build_single_color_variants(color: str) -> List[str]:
    return [
        f"純{color}的船",
        f"船體僅為{color}的船",
        f"只有{color}的船",
    ]


def parse_bbox(value: Any) -> BBox:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"bbox_xyxy 必須為四個數值，收到：{value!r}")

    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"bbox_xyxy 包含非數值內容：{value!r}") from error

    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox 面積必須大於 0：{value!r}")

    return x1, y1, x2, y2


def number_for_json(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def bbox_for_json(bbox: BBox) -> List[int | float]:
    return [number_for_json(value) for value in bbox]


def infer_source_name(objects: Sequence[Dict[str, Any]]) -> str:
    source_names = {
        clean_text(obj.get("source_name", ""))
        for obj in objects
        if clean_text(obj.get("source_name", ""))
    }

    if not source_names:
        raise ValueError("標記中找不到 source_name")

    if len(source_names) != 1:
        raise ValueError(
            "同一標記檔包含多個 source_name："
            f"{sorted(source_names)}"
        )

    return next(iter(source_names))


def infer_source_size(
    objects: Sequence[Dict[str, Any]],
) -> Tuple[int, int]:
    sizes = set()

    for obj in objects:
        source_size = obj.get("source_size", {})
        if not isinstance(source_size, dict):
            continue

        width = source_size.get("width")
        height = source_size.get("height")

        if width is None or height is None:
            continue

        try:
            sizes.add((int(width), int(height)))
        except (TypeError, ValueError):
            continue

    if not sizes:
        raise ValueError("標記中找不到有效的 source_size.width/height")

    if len(sizes) != 1:
        raise ValueError(
            "同一標記檔包含多組 source_size："
            f"{sorted(sizes)}"
        )

    width, height = next(iter(sizes))

    if width <= 0 or height <= 0:
        raise ValueError(f"影像尺寸無效：width={width}, height={height}")

    return width, height


def build_image_index(image_dir: Optional[Path]) -> Dict[str, str]:
    if image_dir is None:
        return {}

    image_dir = image_dir.resolve()
    if not image_dir.is_dir():
        raise NotADirectoryError(f"影像資料夾不存在：{image_dir}")

    index: Dict[str, str] = {}

    image_paths = [
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for path in tqdm(
        image_paths,
        desc="建立影像索引",
        unit="image",
        dynamic_ncols=True,
    ):
        index.setdefault(path.stem, path.name)

    return index


def resolve_filename(
    source_name: str,
    image_index: Dict[str, str],
    default_image_ext: str,
) -> str:
    if source_name in image_index:
        return image_index[source_name]

    suffix = Path(source_name).suffix
    if suffix.lower() in IMAGE_EXTENSIONS:
        return source_name

    extension = default_image_ext.strip()
    if not extension.startswith("."):
        extension = "." + extension

    return source_name + extension


def ensure_group(
    groups: "OrderedDict[str, PhraseGroup]",
    *,
    semantic_key: str,
    phrase: str,
    group_type: str,
) -> PhraseGroup:
    if semantic_key not in groups:
        groups[semantic_key] = PhraseGroup(
            semantic_key=semantic_key,
            phrase=clean_text(phrase),
            group_type=group_type,
        )
    return groups[semantic_key]


def add_object_to_groups(
    groups: "OrderedDict[str, PhraseGroup]",
    obj: Dict[str, Any],
    *,
    include_contains_color: bool,
    include_single_color: bool,
    include_exact_color: bool,
) -> None:
    bbox = parse_bbox(obj.get("bbox_xyxy"))

    attributes = obj.get("attributes", {})
    if not isinstance(attributes, dict):
        attributes = {}

    colors = normalize_colors(attributes.get("main_colors", []))
    is_multicolor = bool(attributes.get("is_multicolor", len(colors) > 1))

    if not colors:
        # 沒有顏色屬性時，保留原始 query_text 作為 fallback。
        original_query = clean_text(obj.get("query_text", ""))
        if original_query:
            semantic_key = f"raw_query:{original_query}"
            group = ensure_group(
                groups,
                semantic_key=semantic_key,
                phrase=original_query,
                group_type="raw_query",
            )
            group.add_bbox(bbox)
            group.add_variants(build_exact_variants(obj))
        return

    sorted_color_values = sort_colors(colors)

    # 1. 精確多色組合，例如「黃綠白相間的船」。
    # 單色物件不建立 colors_exact，避免與「含綠色」語意混淆。
    if include_exact_color and (is_multicolor or len(colors) > 1):
        semantic_key = f"colors_exact:{colors_key(colors)}"
        phrase = build_multicolor_phrase(
            colors=colors,
            original_query=clean_text(obj.get("query_text", "")),
        )
        group = ensure_group(
            groups,
            semantic_key=semantic_key,
            phrase=phrase,
            group_type="colors_exact",
        )
        group.add_bbox(bbox)
        group.add_variants(build_exact_variants(obj))

    # 2. 單色查詢，例如「單色綠色的船」。
    if (
        include_single_color
        and len(sorted_color_values) == 1
        and not is_multicolor
    ):
        color = sorted_color_values[0]
        semantic_key = f"single_color:{color}"
        phrase = f"單色{color}的船"

        group = ensure_group(
            groups,
            semantic_key=semantic_key,
            phrase=phrase,
            group_type="single_color",
        )
        group.add_bbox(bbox)
        group.add_variants(build_single_color_variants(color))

    # 3. 包含顏色查詢。
    # 多色與單色物件都加入，因此「含綠色的船」可合併全部合格 bbox。
    if include_contains_color:
        for color in sorted_color_values:
            semantic_key = f"contains_color:{color}"
            phrase = f"含{color}的船"

            group = ensure_group(
                groups,
                semantic_key=semantic_key,
                phrase=phrase,
                group_type="contains_color",
            )
            group.add_bbox(bbox)
            group.add_variants(build_contains_variants(color))


def sort_phrase_groups(
    groups: Iterable[PhraseGroup],
) -> List[PhraseGroup]:
    type_priority = {
        "colors_exact": 0,
        "single_color": 1,
        "contains_color": 2,
        "raw_query": 3,
    }

    return sorted(
        groups,
        key=lambda group: (
            type_priority.get(group.group_type, 99),
            group.semantic_key,
        ),
    )


def build_caption_and_regions(
    groups: Sequence[PhraseGroup],
) -> Tuple[str, List[Dict[str, Any]], Dict[str, List[str]]]:
    caption_parts: List[str] = []
    regions: List[Dict[str, Any]] = []
    phrase_variants: Dict[str, List[str]] = {}

    cursor = 0

    for group in groups:
        phrase = clean_text(group.phrase)
        if not phrase:
            continue

        start = cursor
        end = start + len(phrase)

        caption_parts.append(phrase)

        regions.append(
            {
                "semantic_key": group.semantic_key,
                "phrase": phrase,
                "tokens_positive": [[start, end]],
                "bbox": [
                    bbox_for_json(bbox)
                    for bbox in group.bboxes
                ],
            }
        )

        if group.variants:
            phrase_variants[phrase] = list(group.variants)

        # 每個 phrase 後方加上一個全形句號。
        cursor = end + 1

    caption = "。".join(caption_parts)
    if caption:
        caption += "。"

    return caption, regions, phrase_variants


def validate_odvg_record(record: Dict[str, Any]) -> None:
    filename = record.get("filename")
    width = record.get("width")
    height = record.get("height")
    grounding = record.get("grounding")

    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("輸出 record 缺少 filename")

    if not isinstance(width, int) or width <= 0:
        raise ValueError(f"輸出 width 無效：{width!r}")

    if not isinstance(height, int) or height <= 0:
        raise ValueError(f"輸出 height 無效：{height!r}")

    if not isinstance(grounding, dict):
        raise ValueError("輸出 record 缺少 grounding")

    caption = grounding.get("caption")
    regions = grounding.get("regions")

    if not isinstance(caption, str):
        raise ValueError("grounding.caption 必須為字串")

    if not isinstance(regions, list):
        raise ValueError("grounding.regions 必須為陣列")

    seen_semantic_keys = set()

    for index, region in enumerate(regions):
        if not isinstance(region, dict):
            raise ValueError(f"regions[{index}] 必須為物件")

        semantic_key = region.get("semantic_key")
        phrase = region.get("phrase")
        spans = region.get("tokens_positive")
        bboxes = region.get("bbox")

        if not isinstance(semantic_key, str) or not semantic_key:
            raise ValueError(f"regions[{index}] 缺少 semantic_key")

        if semantic_key in seen_semantic_keys:
            raise ValueError(f"semantic_key 重複：{semantic_key}")
        seen_semantic_keys.add(semantic_key)

        if not isinstance(phrase, str) or not phrase:
            raise ValueError(f"regions[{index}] 缺少 phrase")

        if (
            not isinstance(spans, list)
            or len(spans) == 0
            or not isinstance(spans[0], list)
            or len(spans[0]) != 2
        ):
            raise ValueError(
                f"regions[{index}].tokens_positive 格式錯誤：{spans!r}"
            )

        start, end = spans[0]
        if not isinstance(start, int) or not isinstance(end, int):
            raise ValueError(
                f"regions[{index}] token span 必須是整數"
            )

        if not (0 <= start < end <= len(caption)):
            raise ValueError(
                f"regions[{index}] token span 超出 caption："
                f"[{start}, {end}], caption_length={len(caption)}"
            )

        actual_phrase = caption[start:end]
        if actual_phrase != phrase:
            raise ValueError(
                f"regions[{index}] token span 不符："
                f"caption[{start}:{end}]={actual_phrase!r}, "
                f"phrase={phrase!r}"
            )

        if not isinstance(bboxes, list) or len(bboxes) == 0:
            raise ValueError(f"regions[{index}] 沒有 bbox")

        for bbox_index, bbox in enumerate(bboxes):
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(
                    f"regions[{index}].bbox[{bbox_index}] 格式錯誤"
                )

            x1, y1, x2, y2 = bbox
            if not (x2 > x1 and y2 > y1):
                raise ValueError(
                    f"regions[{index}].bbox[{bbox_index}] 面積無效：{bbox}"
                )

            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                raise ValueError(
                    f"regions[{index}].bbox[{bbox_index}] 超出影像範圍："
                    f"bbox={bbox}, size=({width}, {height})"
                )


def convert_annotation(
    objects: Any,
    *,
    input_path: Path,
    image_index: Dict[str, str],
    default_image_ext: str,
    include_contains_color: bool,
    include_single_color: bool,
    include_exact_color: bool,
) -> Dict[str, Any]:
    if not isinstance(objects, list):
        raise TypeError(
            f"標記根節點必須是 list，收到：{type(objects).__name__}"
        )

    if len(objects) == 0:
        raise ValueError("標記陣列為空")

    if not all(isinstance(obj, dict) for obj in objects):
        raise TypeError("標記陣列中的每個元素都必須是 JSON object")

    source_name = infer_source_name(objects)
    width, height = infer_source_size(objects)
    filename = resolve_filename(
        source_name=source_name,
        image_index=image_index,
        default_image_ext=default_image_ext,
    )

    groups: "OrderedDict[str, PhraseGroup]" = OrderedDict()

    for obj in objects:
        add_object_to_groups(
            groups,
            obj,
            include_contains_color=include_contains_color,
            include_single_color=include_single_color,
            include_exact_color=include_exact_color,
        )

    sorted_groups = sort_phrase_groups(groups.values())
    caption, regions, phrase_variants = build_caption_and_regions(
        sorted_groups
    )

    if not regions:
        raise ValueError(
            "沒有建立任何 grounding region；請檢查 query_text "
            "與 attributes.main_colors"
        )

    original_queries = list(
        dict.fromkeys(
            clean_text(obj.get("query_text", ""))
            for obj in objects
            if clean_text(obj.get("query_text", ""))
        )
    )

    record: Dict[str, Any] = {
        "filename": filename,
        "height": height,
        "width": width,
        "grounding": {
            "caption": caption,
            "regions": regions,
        },
        "metadata": {
            "source_name": source_name,
            "source_label": input_path.name,
            "object_count": len(objects),
            "region_count": len(regions),
            "phrase_variants": phrase_variants,
            "original_query_texts": original_queries,
        },
    }

    validate_odvg_record(record)
    return record


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(
    path: Path,
    record: Dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"輸出檔案已存在，請使用 --overwrite：{path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            record,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    temporary_path.replace(path)


def write_jsonl(
    path: Path,
    records: Sequence[Dict[str, Any]],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"輸出檔案已存在，請使用 --overwrite：{path}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")

    temporary_path.replace(path)


def collect_json_files(
    input_dir: Path,
    *,
    recursive: bool,
) -> List[Path]:
    iterator = input_dir.rglob("*.json") if recursive else input_dir.glob("*.json")
    return sorted(path for path in iterator if path.is_file())


def resolve_single_output_path(
    input_path: Path,
    output_path: Optional[Path],
) -> Path:
    if output_path is None:
        return input_path.with_name(input_path.stem + "_odvg.json")

    if output_path.exists() and output_path.is_dir():
        return output_path / (input_path.stem + "_odvg.json")

    if output_path.suffix.lower() == ".json":
        return output_path

    return output_path / (input_path.stem + "_odvg.json")


def convert_single_file(
    input_path: Path,
    *,
    output_path: Optional[Path],
    image_index: Dict[str, str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    objects = load_json(input_path)

    record = convert_annotation(
        objects,
        input_path=input_path,
        image_index=image_index,
        default_image_ext=args.default_image_ext,
        include_contains_color=not args.no_contains_color,
        include_single_color=not args.no_single_color,
        include_exact_color=not args.no_exact_color,
    )

    if args.dry_run:
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return record

    if args.format == "jsonl":
        destination = output_path or input_path.with_name(
            input_path.stem + "_odvg.jsonl"
        )
        write_jsonl(
            destination,
            [record],
            overwrite=args.overwrite,
        )
    else:
        destination = resolve_single_output_path(
            input_path,
            output_path,
        )
        write_json(
            destination,
            record,
            overwrite=args.overwrite,
        )

    print(f"[完成] {input_path} -> {destination}")
    return record


def convert_directory(
    input_dir: Path,
    *,
    output_path: Path,
    image_index: Dict[str, str],
    args: argparse.Namespace,
) -> None:
    input_files = collect_json_files(
        input_dir,
        recursive=not args.no_recursive,
    )

    if not input_files:
        raise FileNotFoundError(
            f"找不到 JSON 標記檔：{input_dir}"
        )

    records: List[Dict[str, Any]] = []
    errors: List[Tuple[Path, str]] = []
    converted_count = 0

    for input_file in tqdm(
        input_files,
        desc="轉換標記",
        unit="file",
        dynamic_ncols=True,
    ):
        try:
            objects = load_json(input_file)

            record = convert_annotation(
                objects,
                input_path=input_file,
                image_index=image_index,
                default_image_ext=args.default_image_ext,
                include_contains_color=not args.no_contains_color,
                include_single_color=not args.no_single_color,
                include_exact_color=not args.no_exact_color,
            )

            if args.format == "jsonl":
                records.append(record)
            elif not args.dry_run:
                relative_path = input_file.relative_to(input_dir)
                destination = output_path / relative_path
                write_json(
                    destination,
                    record,
                    overwrite=args.overwrite,
                )

            converted_count += 1

        except Exception as error:
            errors.append((input_file, str(error)))
            if args.fail_fast:
                raise

    if args.format == "jsonl" and not args.dry_run:
        write_jsonl(
            output_path,
            records,
            overwrite=args.overwrite,
        )

    print()
    print(f"輸入檔案數：{len(input_files)}")
    print(f"成功轉換：{converted_count}")
    print(f"失敗檔案：{len(errors)}")

    if args.dry_run:
        print("模式：dry-run，未寫入任何輸出檔案")

    if errors:
        print("\n失敗清單：")
        for path, message in errors[:20]:
            print(f"- {path}: {message}")

        if len(errors) > 20:
            print(f"- 另有 {len(errors) - 20} 筆錯誤未顯示")

        raise RuntimeError(
            f"有 {len(errors)} 個標記檔轉換失敗"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "將 LightDet object-centric JSON 轉換為 "
            "ODVG Phrase Grounding JSON / JSONL"
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="輸入 JSON 檔案或標記資料夾",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "輸出 JSON、JSONL 或資料夾。"
            "單檔未指定時會輸出 *_odvg.json"
        ),
    )
    parser.add_argument(
        "--format",
        choices=("json", "jsonl"),
        default="json",
        help=(
            "json：逐檔輸出；"
            "jsonl：將所有 record 整合為一個檔案"
        ),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="可選，用於依 source_name 尋找真實影像檔名",
    )
    parser.add_argument(
        "--default-image-ext",
        default=".jpg",
        help="找不到實際影像時使用的預設副檔名，預設 .jpg",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允許覆寫既有輸出檔案",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只驗證與預覽，不寫入輸出",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="批次處理遇到第一個錯誤立即停止",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="資料夾模式只處理第一層 JSON，不遞迴子資料夾",
    )
    parser.add_argument(
        "--no-exact-color",
        action="store_true",
        help="不產生 colors_exact phrase",
    )
    parser.add_argument(
        "--no-contains-color",
        action="store_true",
        help="不產生 contains_color phrase",
    )
    parser.add_argument(
        "--no-single-color",
        action="store_true",
        help="不產生 single_color phrase",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path: Path = args.input.expanduser().resolve()
    output_path: Optional[Path] = (
        args.output.expanduser().resolve()
        if args.output is not None
        else None
    )
    image_dir: Optional[Path] = (
        args.image_dir.expanduser().resolve()
        if args.image_dir is not None
        else None
    )

    if not input_path.exists():
        parser.error(f"輸入路徑不存在：{input_path}")

    image_index = build_image_index(image_dir)

    try:
        if input_path.is_file():
            if input_path.suffix.lower() != ".json":
                parser.error("單檔模式目前只接受 .json")

            convert_single_file(
                input_path,
                output_path=output_path,
                image_index=image_index,
                args=args,
            )
            return 0

        if not input_path.is_dir():
            parser.error(f"輸入不是檔案或資料夾：{input_path}")

        if output_path is None and not args.dry_run:
            parser.error("資料夾模式必須指定 --output")

        if args.format == "jsonl":
            if output_path is None:
                parser.error("JSONL 模式必須指定 --output")
            if output_path.suffix.lower() != ".jsonl":
                parser.error("JSONL 輸出路徑必須使用 .jsonl 副檔名")
        else:
            if output_path is None:
                output_path = input_path.parent / (
                    input_path.name + "_odvg"
                )

        convert_directory(
            input_path,
            output_path=output_path,
            image_index=image_index,
            args=args,
        )
        return 0

    except Exception as error:
        print(f"[錯誤] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
