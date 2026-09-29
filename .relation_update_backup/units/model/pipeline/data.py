from __future__ import annotations

import hashlib
import json
import math
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


IMAGE_EXTS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
]

CACHE_VERSION = "lightdet_uint8_image_cache_v4_odvg"


class ODVGFormatError(ValueError):
    """Raised when an annotation does not follow the required ODVG schema."""


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _as_int(value: Any, *, name: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ODVGFormatError(f"{name} must be an integer, got {value!r}") from error
    return result


def _parse_bbox_list(value: Any, *, context: str) -> List[List[float]]:
    """
    Accept both ODVG bbox forms:
      [x1, y1, x2, y2]
      [[x1, y1, x2, y2], ...]
    """
    if not isinstance(value, list) or not value:
        raise ODVGFormatError(f"{context}.bbox must be a non-empty list")

    if len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        candidates = [value]
    else:
        candidates = value

    boxes: List[List[float]] = []
    for box_index, box in enumerate(candidates):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ODVGFormatError(
                f"{context}.bbox[{box_index}] must have four values, got {box!r}"
            )

        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError) as error:
            raise ODVGFormatError(
                f"{context}.bbox[{box_index}] contains a non-numeric value: {box!r}"
            ) from error

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 <= x1 or y2 <= y1:
            raise ODVGFormatError(
                f"{context}.bbox[{box_index}] has zero/negative area: {box!r}"
            )

        boxes.append([x1, y1, x2, y2])

    return boxes


def _parse_char_spans(
    value: Any,
    *,
    caption: str,
    phrase: str,
    context: str,
) -> List[List[int]]:
    if not isinstance(value, list) or not value:
        raise ODVGFormatError(
            f"{context}.tokens_positive must be a non-empty list"
        )

    spans: List[List[int]] = []
    seen = set()

    for span_index, span in enumerate(value):
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ODVGFormatError(
                f"{context}.tokens_positive[{span_index}] must be [start, end]"
            )

        start = _as_int(span[0], name=f"{context}.tokens_positive[{span_index}][0]")
        end = _as_int(span[1], name=f"{context}.tokens_positive[{span_index}][1]")

        if not (0 <= start < end <= len(caption)):
            raise ODVGFormatError(
                f"{context}.tokens_positive[{span_index}] is outside caption: "
                f"[{start}, {end}], caption_length={len(caption)}"
            )

        key = (start, end)
        if key not in seen:
            spans.append([start, end])
            seen.add(key)

    # The converter currently emits one contiguous span per phrase. Enforce the
    # exact mapping in that common case and still allow standard multi-span ODVG.
    if len(spans) == 1:
        start, end = spans[0]
        actual = caption[start:end]
        if actual != phrase:
            raise ODVGFormatError(
                f"{context}: caption[{start}:{end}]={actual!r} does not match "
                f"phrase={phrase!r}"
            )

    return spans


def validate_odvg_record(record: Any, *, anno_path: str = "") -> Dict[str, Any]:
    prefix = f"{anno_path}: " if anno_path else ""

    if not isinstance(record, dict):
        raise ODVGFormatError(
            prefix + "annotation root must be an ODVG JSON object, not a list"
        )

    filename = _clean_text(record.get("filename"))
    if not filename:
        raise ODVGFormatError(prefix + "missing filename")

    width = _as_int(record.get("width"), name=prefix + "width")
    height = _as_int(record.get("height"), name=prefix + "height")
    if width <= 0 or height <= 0:
        raise ODVGFormatError(
            prefix + f"invalid image size: width={width}, height={height}"
        )

    grounding = record.get("grounding")
    if not isinstance(grounding, dict):
        raise ODVGFormatError(prefix + "missing grounding object")

    caption = grounding.get("caption")
    if not isinstance(caption, str) or not caption.strip():
        raise ODVGFormatError(prefix + "grounding.caption must be a non-empty string")

    regions = grounding.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ODVGFormatError(prefix + "grounding.regions must be a non-empty list")

    normalized_regions: List[Dict[str, Any]] = []
    for region_index, region in enumerate(regions):
        context = prefix + f"grounding.regions[{region_index}]"
        if not isinstance(region, dict):
            raise ODVGFormatError(context + " must be an object")

        phrase = _clean_text(region.get("phrase"))
        if not phrase:
            raise ODVGFormatError(context + ".phrase must be non-empty")

        semantic_key = _clean_text(region.get("semantic_key"))
        if not semantic_key:
            # Standard ODVG does not require this field, but the LightDet
            # converter emits it. Keep a deterministic fallback.
            semantic_key = f"region:{region_index}:{phrase}"

        spans = _parse_char_spans(
            region.get("tokens_positive"),
            caption=caption,
            phrase=phrase,
            context=context,
        )
        boxes = _parse_bbox_list(region.get("bbox"), context=context)

        for box_index, (x1, y1, x2, y2) in enumerate(boxes):
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                raise ODVGFormatError(
                    f"{context}.bbox[{box_index}] exceeds annotation size "
                    f"({width}, {height}): {[x1, y1, x2, y2]}"
                )

        normalized_regions.append(
            {
                "region_index": region_index,
                "semantic_key": semantic_key,
                "phrase": phrase,
                "tokens_positive": spans,
                "bbox": boxes,
            }
        )

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "caption": caption,
        "regions": normalized_regions,
        "metadata": metadata,
    }


def _bbox_identity(box: Sequence[float], precision: int = 4) -> Tuple[float, ...]:
    return tuple(round(float(value), precision) for value in box)


def merge_regions_to_unique_targets(
    regions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Merge repeated physical boxes across phrases.

    Example:
      exact-color phrase -> box A
      contains-green phrase -> box A, box B

    The output has unique targets [A, B]. Target A receives both positive
    character-span groups rather than becoming two duplicate Hungarian GT boxes.
    """
    targets: List[Dict[str, Any]] = []
    box_to_target: Dict[Tuple[float, ...], int] = {}
    region_to_target_indices: List[List[int]] = []

    for region in regions:
        target_indices: List[int] = []

        for box in region["bbox"]:
            box_key = _bbox_identity(box)
            target_index = box_to_target.get(box_key)

            if target_index is None:
                target_index = len(targets)
                box_to_target[box_key] = target_index
                targets.append(
                    {
                        "bbox": list(box),
                        "positive_char_spans": [],
                        "phrases": [],
                        "semantic_keys": [],
                        "region_indices": [],
                    }
                )

            target = targets[target_index]

            for span in region["tokens_positive"]:
                normalized_span = [int(span[0]), int(span[1])]
                if normalized_span not in target["positive_char_spans"]:
                    target["positive_char_spans"].append(normalized_span)

            phrase = region["phrase"]
            if phrase not in target["phrases"]:
                target["phrases"].append(phrase)

            semantic_key = region["semantic_key"]
            if semantic_key not in target["semantic_keys"]:
                target["semantic_keys"].append(semantic_key)

            region_index = int(region["region_index"])
            if region_index not in target["region_indices"]:
                target["region_indices"].append(region_index)

            target_indices.append(target_index)

        region_to_target_indices.append(list(dict.fromkeys(target_indices)))

    if not targets:
        raise ODVGFormatError("ODVG record produced no target boxes")

    return {
        "targets": targets,
        "region_to_target_indices": region_to_target_indices,
    }


def _flatten_negative_phrase_pool(value: Any) -> List[str]:
    """
    Accept common JSON pool layouts:

      ["黃色的船", "大型貨輪"]
      {"phrases": [...]}
      {"negative_phrases": [...]}
      {"color": [...], "type": [...]}
    """
    collected: List[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, str):
            text = item.strip()
            if text:
                collected.append(text)
            return

        if isinstance(item, dict):
            preferred_keys = (
                "phrases",
                "negative_phrases",
                "queries",
                "items",
                "data",
            )
            preferred_found = False

            for key in preferred_keys:
                if key in item:
                    visit(item[key])
                    preferred_found = True

            if not preferred_found:
                for child in item.values():
                    visit(child)
            return

        if isinstance(item, (list, tuple, set)):
            for child in item:
                visit(child)

    visit(value)

    unique: List[str] = []
    seen = set()

    for phrase in collected:
        if phrase in seen:
            continue
        unique.append(phrase)
        seen.add(phrase)

    return unique


def load_negative_phrase_pool(
    path: Optional[str],
) -> List[str]:
    if path is None:
        return []

    resolved = os.path.abspath(str(path))

    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"ODVG negative phrase pool not found: {resolved}"
        )

    with open(
        resolved,
        "r",
        encoding="utf-8",
    ) as file:
        value = json.load(file)

    phrases = _flatten_negative_phrase_pool(value)

    if not phrases:
        raise ValueError(
            "ODVG negative phrase pool contains no usable strings: "
            f"{resolved}"
        )

    return phrases


def append_negative_phrases_to_caption(
    caption: str,
    negative_phrases: Sequence[str],
    separator: str = "；負向描述：",
) -> Tuple[str, List[List[int]]]:
    """
    Append unmatched phrases to a caption and return their character spans.

    Positive ODVG spans remain unchanged because text is appended only at the
    end. These phrases receive no target box and therefore become token-level
    negatives in GroundingLoss.
    """
    caption = str(caption)
    phrases = [
        str(value).strip()
        for value in negative_phrases
        if str(value).strip()
    ]

    if not phrases:
        return caption, []

    suffix = str(separator) + "、".join(phrases)
    extended = caption + suffix

    spans: List[List[int]] = []
    search_start = len(caption) + len(str(separator))

    for phrase in phrases:
        position = extended.find(
            phrase,
            search_start,
        )

        if position < 0:
            raise RuntimeError(
                f"Failed to locate appended negative phrase: {phrase!r}"
            )

        end = position + len(phrase)
        spans.append([
            int(position),
            int(end),
        ])
        search_start = end

    return extended, spans


class ShipGroundingDataset(Dataset):
    """
    ODVG-only LightDet dataset.

    Dataset unit:
      one image + one complete caption + all grounding regions.

    Legacy object-list annotations, query_text grouping, main-color grouping and
    query_texts_aug sampling are intentionally unsupported.
    """

    def __init__(
        self,
        image_dir: str,
        anno_paths: List[str],
        image_size: Tuple[int, int] = (640, 640),
        random_seed: Optional[int] = None,
        normalize_boxes: bool = True,
        clip_boxes: bool = True,
        min_box_size: float = 1.0,
        image_mean: Optional[Tuple[float, float, float]] = None,
        image_std: Optional[Tuple[float, float, float]] = None,
        strict_image: bool = True,
        strict_size: bool = True,
        cache_images: bool = False,
        image_cache_dir: Optional[str] = None,
        prebuild_image_cache: bool = False,
        negative_phrase_pool_path: Optional[str] = None,
        negative_phrase_ratio: float = 0.0,
        negative_phrase_max_per_image: int = 3,
        enable_negative_phrases: bool = False,
        negative_phrase_separator: str = "；負向描述：",
    ) -> None:
        self.image_dir = os.path.abspath(str(image_dir))
        self.anno_paths = [os.path.abspath(str(path)) for path in anno_paths]
        self.image_size = tuple(int(v) for v in image_size)
        self.normalize_boxes = bool(normalize_boxes)
        self.clip_boxes = bool(clip_boxes)
        self.min_box_size = float(min_box_size)
        self.image_mean = image_mean
        self.image_std = image_std
        self.strict_image = bool(strict_image)
        self.strict_size = bool(strict_size)
        self.random_seed = int(random_seed or 0)
        self.rng = random.Random(self.random_seed)

        self.negative_phrase_pool_path = negative_phrase_pool_path
        self.negative_phrase_ratio = max(
            0.0,
            float(negative_phrase_ratio),
        )
        self.negative_phrase_max_per_image = max(
            0,
            int(negative_phrase_max_per_image),
        )
        self.enable_negative_phrases = bool(
            enable_negative_phrases
            and self.negative_phrase_ratio > 0.0
            and self.negative_phrase_max_per_image > 0
        )
        self.negative_phrase_separator = str(
            negative_phrase_separator
        )
        self.negative_phrase_pool = (
            load_negative_phrase_pool(
                negative_phrase_pool_path
            )
            if self.enable_negative_phrases
            else []
        )

        self.cache_images = bool(cache_images)
        self.image_cache_dir = image_cache_dir
        self.prebuild_image_cache = bool(prebuild_image_cache)
        self.require_cache_ready = bool(self.cache_images and self.prebuild_image_cache)

        # ODVG is always image-level. QueryBudgetBatchSampler uses region count as
        # an annotation budget so the existing YAML batch_size remains practical.
        self.image_level_batching = True
        self.image_cache_map: Dict[str, str] = {}

        if self.cache_images:
            if self.image_cache_dir is None:
                target_h, target_w = self.image_size
                self.image_cache_dir = os.path.join(
                    self.image_dir,
                    f".lightdet_cache_uint8_{target_h}x{target_w}",
                )
            self.image_cache_dir = os.path.abspath(str(self.image_cache_dir))
            os.makedirs(self.image_cache_dir, exist_ok=True)

        self.records: List[Dict[str, Any]] = []
        self.annos = self.records  # retained name for existing diagnostics
        self.image_paths: List[str] = []
        self.samples: List[Dict[str, Any]] = []

        for anno_path in self.anno_paths:
            with open(anno_path, "r", encoding="utf-8") as file:
                raw_record = json.load(file)

            record = validate_odvg_record(raw_record, anno_path=anno_path)
            merged = merge_regions_to_unique_targets(record["regions"])
            record["unique_targets"] = merged["targets"]
            record["region_to_target_indices"] = merged["region_to_target_indices"]

            image_path = self.find_image_path(record["filename"])
            if image_path is None:
                if self.strict_image:
                    raise FileNotFoundError(
                        f"Image not found: filename={record['filename']}, "
                        f"image_dir={self.image_dir}, anno={anno_path}"
                    )
                continue

            record_index = len(self.records)
            source_name = _clean_text(record["metadata"].get("source_name"))
            if not source_name:
                source_name = os.path.splitext(record["filename"])[0]

            self.records.append(record)
            self.image_paths.append(image_path)
            self.samples.append(
                {
                    "anno_idx": record_index,
                    "caption": record["caption"],
                    # Temporary runtime alias: existing BERT precompute code reads
                    # sample["query_text"]. The annotation source remains ODVG-only.
                    "query_text": record["caption"],
                    "region_count": len(record["regions"]),
                    "target_count": len(record["unique_targets"]),
                    "source_name": source_name,
                }
            )

        if not self.records:
            raise RuntimeError(
                "No valid ODVG annotations were loaded. "
                "Expected JSON objects with grounding.caption and grounding.regions."
            )

        self.image_level_indices = list(range(len(self.records)))
        self.queries_per_image = max(
            1,
            int(
                math.ceil(
                    sum(max(1, len(record["regions"])) for record in self.records)
                    / len(self.records)
                )
            ),
        )

    def __len__(self) -> int:
        return len(self.records)

    def get_query_count_for_dataset_index(self, idx: int) -> int:
        record = self.records[int(idx)]
        return max(1, len(record["regions"]))

    def find_image_path(self, filename: str) -> Optional[str]:
        filename = _clean_text(filename)
        if not filename:
            return None

        direct_path = os.path.join(self.image_dir, filename)
        if os.path.isfile(direct_path):
            return direct_path

        stem, ext = os.path.splitext(filename)
        candidates: List[str] = []

        if ext:
            candidates.extend(
                os.path.join(self.image_dir, stem + image_ext)
                for image_ext in IMAGE_EXTS
            )
        else:
            candidates.extend(
                os.path.join(self.image_dir, filename + image_ext)
                for image_ext in IMAGE_EXTS
            )

        for path in candidates:
            if os.path.isfile(path):
                return path

        return None

    @staticmethod
    def reorder_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes.reshape(-1, 4)
        x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
        y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
        x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
        y2 = torch.maximum(boxes[:, 1], boxes[:, 3])
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def resize_boxes_xyxy(
        boxes: torch.Tensor,
        orig_size: Tuple[int, int],
        new_size: Tuple[int, int],
    ) -> torch.Tensor:
        orig_h, orig_w = orig_size
        new_h, new_w = new_size
        if boxes.numel() == 0:
            return boxes.reshape(-1, 4)
        boxes = boxes.clone()
        boxes[:, [0, 2]] *= float(new_w) / float(orig_w)
        boxes[:, [1, 3]] *= float(new_h) / float(orig_h)
        return boxes

    @staticmethod
    def clip_boxes_xyxy(
        boxes: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        h, w = image_size
        if boxes.numel() == 0:
            return boxes.reshape(-1, 4)
        boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
        return boxes

    @staticmethod
    def normalize_xyxy(
        boxes: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        h, w = image_size
        if boxes.numel() == 0:
            return boxes.reshape(-1, 4)
        boxes = boxes.clone()
        boxes[:, [0, 2]] /= float(w)
        boxes[:, [1, 3]] /= float(h)
        return boxes.clamp(0.0, 1.0)

    def valid_box_mask_pixel(self, boxes: torch.Tensor) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool)
        wh = boxes[:, 2:] - boxes[:, :2]
        return (wh[:, 0] >= self.min_box_size) & (wh[:, 1] >= self.min_box_size)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        target_h, target_w = self.image_size
        image = TF.resize(
            image,
            [target_h, target_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        image_tensor = TF.to_tensor(image)
        if self.image_mean is not None and self.image_std is not None:
            image_tensor = TF.normalize(
                image_tensor,
                mean=list(self.image_mean),
                std=list(self.image_std),
            )
        return image_tensor

    def preprocess_image_u8(self, image: Image.Image) -> torch.Tensor:
        target_h, target_w = self.image_size
        image = TF.resize(
            image,
            [target_h, target_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        return TF.pil_to_tensor(image).contiguous()

    def get_image_cache_key(self, image_path: str) -> str:
        stat = os.stat(image_path)
        payload = {
            "version": CACHE_VERSION,
            "path": os.path.abspath(image_path),
            "mtime_ns": int(stat.st_mtime_ns),
            "file_size": int(stat.st_size),
            "image_size": tuple(self.image_size),
        }
        payload_text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(payload_text.encode("utf-8")).hexdigest()

    def get_image_cache_path(self, image_path: str) -> str:
        if image_path in self.image_cache_map:
            return self.image_cache_map[image_path]
        key = self.get_image_cache_key(image_path)
        if self.image_cache_dir is None:
            raise RuntimeError("image_cache_dir is not configured")
        return os.path.join(self.image_cache_dir, f"{key}.pt")

    def load_image_uncached(self, image_path: str) -> Tuple[torch.Tensor, int, int]:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            orig_w, orig_h = image.size
            image_tensor = self.preprocess_image(image)
        return image_tensor, int(orig_w), int(orig_h)

    def load_image_u8_uncached(self, image_path: str) -> Tuple[torch.Tensor, int, int]:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            orig_w, orig_h = image.size
            image_u8 = self.preprocess_image_u8(image)
        return image_u8, int(orig_w), int(orig_h)

    def load_image_cached(
        self,
        image_path: str,
        allow_build: bool = False,
    ) -> Tuple[torch.Tensor, int, int]:
        if not self.cache_images:
            return self.load_image_uncached(image_path)

        cache_path = self.get_image_cache_path(image_path)
        if os.path.exists(cache_path):
            try:
                try:
                    obj = torch.load(cache_path, map_location="cpu", weights_only=False)
                except TypeError:
                    obj = torch.load(cache_path, map_location="cpu")

                image_tensor = obj.get("image_u8", obj.get("image"))
                if image_tensor is None:
                    raise KeyError(f"Invalid image cache keys={list(obj.keys())}")
                if image_tensor.dtype != torch.uint8:
                    image_tensor = (
                        image_tensor.clamp(0, 1).mul(255).to(torch.uint8)
                    )
                orig_h, orig_w = obj["orig_size_hw"]
                return image_tensor.contiguous(), int(orig_w), int(orig_h)
            except Exception:
                if self.require_cache_ready and not allow_build:
                    raise
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

        if self.require_cache_ready and not allow_build:
            raise FileNotFoundError(
                f"Image cache missing during training: {cache_path}, image={image_path}"
            )

        image_u8, orig_w, orig_h = self.load_image_u8_uncached(image_path)
        obj = {
            "type": CACHE_VERSION,
            "image_u8": image_u8.cpu(),
            "orig_size_hw": (int(orig_h), int(orig_w)),
            "image_path": image_path,
            "image_size": tuple(self.image_size),
        }
        temporary_path = cache_path + f".tmp.{os.getpid()}.{random.getrandbits(64)}"
        try:
            torch.save(obj, temporary_path)
            os.replace(temporary_path, cache_path)
        finally:
            if os.path.exists(temporary_path):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass

        return image_u8, int(orig_w), int(orig_h)

    def build_image_cache_index(self, verify: bool = False) -> None:
        if not self.cache_images:
            return

        self.image_cache_map = {
            image_path: self.get_image_cache_path(image_path)
            for image_path in dict.fromkeys(self.image_paths)
        }

        if verify:
            missing = [
                (image_path, cache_path)
                for image_path, cache_path in self.image_cache_map.items()
                if not os.path.exists(cache_path)
            ]
            if missing:
                image_path, cache_path = missing[0]
                raise FileNotFoundError(
                    f"Image cache missing: {cache_path}, image={image_path}, "
                    f"missing_count={len(missing)}"
                )

    def build_image_cache(
        self,
        num_workers: int = 8,
        desc: str = "[Image Cache]",
    ) -> None:
        if not self.cache_images:
            return

        self.build_image_cache_index(verify=False)
        image_paths = list(dict.fromkeys(self.image_paths))
        if not image_paths:
            return

        num_workers = max(1, int(num_workers))
        if num_workers == 1:
            for image_path in tqdm(
                image_paths,
                desc=desc,
                unit="image",
                dynamic_ncols=True,
            ):
                self.load_image_cached(image_path, allow_build=True)
            return

        def worker(path: str) -> str:
            self.load_image_cached(path, allow_build=True)
            return path

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, path) for path in image_paths]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc=desc,
                unit="image",
                dynamic_ncols=True,
            ):
                future.result()

    def build_target(
        self,
        record: Dict[str, Any],
        *,
        orig_w: int,
        orig_h: int,
    ) -> Dict[str, Any]:
        annotation_w = int(record["width"])
        annotation_h = int(record["height"])

        if self.strict_size and (orig_w != annotation_w or orig_h != annotation_h):
            raise ValueError(
                "Image/annotation size mismatch: "
                f"image=({orig_w}, {orig_h}), "
                f"annotation=({annotation_w}, {annotation_h}), "
                f"filename={record['filename']}"
            )

        target_h, target_w = self.image_size
        unique_targets = record["unique_targets"]

        boxes_orig = torch.tensor(
            [target["bbox"] for target in unique_targets],
            dtype=torch.float32,
        ).reshape(-1, 4)
        boxes_orig = self.reorder_xyxy(boxes_orig)

        boxes_pixel = self.resize_boxes_xyxy(
            boxes_orig,
            orig_size=(orig_h, orig_w),
            new_size=(target_h, target_w),
        )
        boxes_pixel = self.reorder_xyxy(boxes_pixel)

        if self.clip_boxes:
            boxes_pixel = self.clip_boxes_xyxy(
                boxes_pixel,
                image_size=(target_h, target_w),
            )

        valid_mask = self.valid_box_mask_pixel(boxes_pixel)
        boxes_pixel = boxes_pixel[valid_mask]
        boxes_orig = boxes_orig[valid_mask]

        if self.normalize_boxes:
            boxes = self.normalize_xyxy(
                boxes_pixel,
                image_size=(target_h, target_w),
            )
        else:
            boxes = boxes_pixel.clone()

        kept_indices = torch.nonzero(valid_mask, as_tuple=False).flatten().tolist()
        positive_char_spans = [
            unique_targets[index]["positive_char_spans"]
            for index in kept_indices
        ]
        phrases = [unique_targets[index]["phrases"] for index in kept_indices]
        semantic_keys = [
            unique_targets[index]["semantic_keys"]
            for index in kept_indices
        ]
        region_indices = [
            unique_targets[index]["region_indices"]
            for index in kept_indices
        ]

        labels = torch.zeros((boxes.shape[0],), dtype=torch.long)
        target_indices = torch.tensor(kept_indices, dtype=torch.long)

        return {
            "boxes": boxes,
            "boxes_pixel": boxes_pixel,
            "boxes_orig": boxes_orig,
            "labels": labels,
            "target_indices": target_indices,
            "positive_char_spans": positive_char_spans,
            "phrases": phrases,
            "semantic_keys": semantic_keys,
            "region_indices": region_indices,
            "image_size": torch.tensor([target_h, target_w], dtype=torch.long),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "annotation_size": torch.tensor(
                [annotation_h, annotation_w], dtype=torch.long
            ),
        }

    def sample_negative_phrases(
        self,
        record: Dict[str, Any],
        dataset_index: int,
    ) -> List[str]:
        if (
            not self.enable_negative_phrases
            or not self.negative_phrase_pool
        ):
            return []

        positive_phrases = {
            str(region.get("phrase", "")).strip()
            for region in record["regions"]
            if str(region.get("phrase", "")).strip()
        }
        caption = str(record["caption"])

        candidates = [
            phrase
            for phrase in self.negative_phrase_pool
            if phrase not in positive_phrases
            and phrase not in caption
        ]

        if not candidates:
            return []

        positive_count = max(
            1,
            len(record["regions"]),
        )
        requested = max(
            1,
            int(
                math.ceil(
                    positive_count
                    * self.negative_phrase_ratio
                )
            ),
        )
        requested = min(
            requested,
            self.negative_phrase_max_per_image,
            len(candidates),
        )

        # Deterministic per-image sampling remains stable across workers.
        rng = random.Random(
            self.random_seed
            + int(dataset_index) * 1_000_003
        )
        return rng.sample(
            candidates,
            requested,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
            record_index = int(idx)
            record = self.records[record_index]
            image_path = self.image_paths[record_index]
            anno_path = self.anno_paths[record_index]

            image, orig_w, orig_h = self.load_image_cached(
                image_path
            )
            target = self.build_target(
                record,
                orig_w=orig_w,
                orig_h=orig_h,
            )

            negative_phrases = self.sample_negative_phrases(
                record=record,
                dataset_index=record_index,
            )
            caption, negative_char_spans = (
                append_negative_phrases_to_caption(
                    caption=record["caption"],
                    negative_phrases=negative_phrases,
                    separator=self.negative_phrase_separator,
                )
            )

            source_name = _clean_text(
                record["metadata"].get("source_name")
            )
            if not source_name:
                source_name = os.path.splitext(
                    record["filename"]
                )[0]

            return {
                "image_level": True,
                "image": image,
                "caption": caption,
                "base_caption": record["caption"],
                "target": target,
                "negative_phrases": negative_phrases,
                "negative_char_spans": negative_char_spans,
                "regions": record["regions"],
                "region_to_target_indices": (
                    record["region_to_target_indices"]
                ),
                "image_path": image_path,
                "anno_path": anno_path,
                "filename": record["filename"],
                "source_name": source_name,
                "metadata": record["metadata"],
            }



class QueryBudgetBatchSampler(Sampler[List[int]]):
    """
    Fixed image batches constrained by an ODVG region budget.

    The class name is retained because train.py imports it directly. The budget
    now counts grounding regions, not legacy expanded query samples.
    """

    def __init__(
        self,
        dataset: ShipGroundingDataset,
        query_budget: int = 48,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.query_budget = max(1, int(query_budget))
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed or 0)
        self.epoch = 0
        self.indices = list(range(len(dataset)))
        self.query_counts = [
            max(1, int(dataset.get_query_count_for_dataset_index(index)))
            for index in self.indices
        ]
        self._base_batches = self._build_fixed_batches()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_fixed_batches(self) -> Tuple[Tuple[int, ...], ...]:
        order = list(range(len(self.indices)))
        if self.shuffle:
            random.Random(self.seed).shuffle(order)

        batches: List[List[int]] = []
        batch: List[int] = []
        budget_sum = 0

        for position in order:
            dataset_index = self.indices[position]
            item_cost = self.query_counts[position]

            if batch and budget_sum + item_cost > self.query_budget:
                batches.append(batch)
                batch = []
                budget_sum = 0

            batch.append(dataset_index)
            budget_sum += item_cost

        if batch and (not self.drop_last or budget_sum >= self.query_budget):
            batches.append(batch)

        if not batches and self.indices:
            batches.append(list(self.indices))

        return tuple(tuple(batch) for batch in batches)

    def __iter__(self) -> Iterable[List[int]]:
        batch_order = list(range(len(self._base_batches)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(batch_order)

        for batch_index in batch_order:
            batch = list(self._base_batches[batch_index])
            if self.shuffle and len(batch) > 1:
                rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return len(self._base_batches)


def grounding_collate_fn(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if not batch:
        raise ValueError(
            "grounding_collate_fn received an empty batch"
        )

    unique_images = torch.stack(
        [item["image"] for item in batch],
        dim=0,
    )
    captions = [
        item["caption"]
        for item in batch
    ]
    targets = [
        item["target"]
        for item in batch
    ]

    positive_char_spans = [
        target["positive_char_spans"]
        for target in targets
    ]
    target_phrases = [
        target["phrases"]
        for target in targets
    ]
    target_semantic_keys = [
        target["semantic_keys"]
        for target in targets
    ]
    target_region_indices = [
        target["region_indices"]
        for target in targets
    ]

    negative_phrases = [
        item.get("negative_phrases", [])
        for item in batch
    ]
    negative_char_spans = [
        item.get("negative_char_spans", [])
        for item in batch
    ]

    batch_size = len(batch)
    negative_phrase_count = sum(
        len(values)
        for values in negative_phrases
    )

    return {
        "unique_images": unique_images,
        "captions": captions,

        # Temporary alias retained for current runtime/model helpers.
        # Every entry is a complete image-level ODVG caption.
        "query_texts": captions,

        "image_indices": torch.arange(
            batch_size,
            dtype=torch.long,
        ),
        "targets": targets,
        "positive_char_spans": positive_char_spans,
        "target_phrases": target_phrases,
        "target_semantic_keys": target_semantic_keys,
        "target_region_indices": target_region_indices,

        # Negative phrases are appended to captions but have no bbox.
        "negative_phrases": negative_phrases,
        "negative_char_spans": negative_char_spans,
        "odvg_negative_phrase_count": (
            negative_phrase_count
        ),

        "regions": [
            item["regions"]
            for item in batch
        ],
        "region_to_target_indices": [
            item["region_to_target_indices"]
            for item in batch
        ],
        "image_paths": [
            item["image_path"]
            for item in batch
        ],
        "anno_paths": [
            item["anno_path"]
            for item in batch
        ],
        "filenames": [
            item["filename"]
            for item in batch
        ],
        "source_names": [
            item["source_name"]
            for item in batch
        ],
        "metadata": [
            item["metadata"]
            for item in batch
        ],
        "group_sources": [
            "odvg_caption"
        ] * batch_size,

        # Legacy whole-caption negative flags are always disabled.
        "negative_query_types": [
            ""
        ] * batch_size,
        "query_loss_weights": torch.ones(
            batch_size,
            dtype=torch.float32,
        ),
        "text_negative_mask": torch.zeros(
            batch_size,
            dtype=torch.bool,
        ),

        "num_unique_images": batch_size,
        "num_queries": batch_size,
        "num_regions": sum(
            len(item["regions"])
            for item in batch
        ),
        "num_unique_targets": sum(
            int(target["boxes"].shape[0])
            for target in targets
        ),
    }



def list_json_files(anno_dir: str) -> List[str]:
    anno_dir = os.path.abspath(str(anno_dir))
    if not os.path.isdir(anno_dir):
        raise NotADirectoryError(f"Annotation directory not found: {anno_dir}")

    results: List[str] = []
    for root, _, filenames in os.walk(anno_dir):
        for filename in filenames:
            if filename.lower().endswith(".json"):
                results.append(os.path.join(root, filename))
    return sorted(results)


def build_dataloader(
    dataset: ShipGroundingDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    prefetch_factor: int = 4,
    query_budget_batching: bool = True,
    random_seed: int = 0,
) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "num_workers": int(num_workers),
        "collate_fn": grounding_collate_fn,
        "pin_memory": bool(pin_memory),
        "drop_last": False,
    }

    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

    if query_budget_batching:
        kwargs["batch_sampler"] = QueryBudgetBatchSampler(
            dataset=dataset,
            query_budget=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=random_seed or 0,
        )
        return DataLoader(**kwargs)

    kwargs["batch_size"] = int(batch_size)
    kwargs["shuffle"] = bool(shuffle)
    kwargs["drop_last"] = bool(drop_last)
    return DataLoader(**kwargs)


def build_dataloaders(
    train_image_dir: str,
    train_anno_dir: str,
    val_image_dir: str,
    val_anno_dir: str,
    batch_size: int = 4,
    image_size: Tuple[int, int] = (640, 640),
    num_workers: int = 0,
    random_seed: Optional[int] = None,
    normalize_boxes: bool = True,
    clip_boxes: bool = True,
    min_box_size: float = 1.0,
    image_mean: Optional[
        Tuple[float, float, float]
    ] = None,
    image_std: Optional[
        Tuple[float, float, float]
    ] = None,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    cache_images: bool = False,
    image_cache_dir: Optional[str] = None,
    prebuild_image_cache: bool = False,
    cache_workers: int = 8,
    query_budget_batching: bool = True,

    # Existing configuration names are retained, but their semantics are now
    # ODVG phrase-level negatives rather than whole-caption negative samples.
    negative_query_path: Optional[str] = None,
    negative_sample_ratio: float = 0.0,
    use_negative_queries_in_val: bool = False,
    negative_phrase_max_per_image: int = 3,
    negative_phrase_separator: str = "；負向描述：",
) -> Tuple[DataLoader, DataLoader]:
    train_anno_paths = list_json_files(
        train_anno_dir
    )
    val_anno_paths = list_json_files(
        val_anno_dir
    )

    train_dataset = ShipGroundingDataset(
        image_dir=train_image_dir,
        anno_paths=train_anno_paths,
        image_size=image_size,
        random_seed=random_seed,
        normalize_boxes=normalize_boxes,
        clip_boxes=clip_boxes,
        min_box_size=min_box_size,
        image_mean=image_mean,
        image_std=image_std,
        strict_image=True,
        strict_size=True,
        cache_images=cache_images,
        image_cache_dir=image_cache_dir,
        prebuild_image_cache=prebuild_image_cache,
        negative_phrase_pool_path=negative_query_path,
        negative_phrase_ratio=negative_sample_ratio,
        negative_phrase_max_per_image=(
            negative_phrase_max_per_image
        ),
        enable_negative_phrases=(
            negative_query_path is not None
            and float(negative_sample_ratio) > 0.0
        ),
        negative_phrase_separator=(
            negative_phrase_separator
        ),
    )

    val_dataset = ShipGroundingDataset(
        image_dir=val_image_dir,
        anno_paths=val_anno_paths,
        image_size=image_size,
        random_seed=random_seed,
        normalize_boxes=normalize_boxes,
        clip_boxes=clip_boxes,
        min_box_size=min_box_size,
        image_mean=image_mean,
        image_std=image_std,
        strict_image=True,
        strict_size=True,
        cache_images=cache_images,
        image_cache_dir=image_cache_dir,
        prebuild_image_cache=prebuild_image_cache,
        negative_phrase_pool_path=negative_query_path,
        negative_phrase_ratio=negative_sample_ratio,
        negative_phrase_max_per_image=(
            negative_phrase_max_per_image
        ),
        enable_negative_phrases=bool(
            use_negative_queries_in_val
            and negative_query_path is not None
            and float(negative_sample_ratio) > 0.0
        ),
        negative_phrase_separator=(
            negative_phrase_separator
        ),
    )

    if prebuild_image_cache:
        train_dataset.build_image_cache(
            num_workers=cache_workers,
            desc="[Image Cache] Train",
        )
        val_dataset.build_image_cache(
            num_workers=cache_workers,
            desc="[Image Cache] Val",
        )

    train_dataset.build_image_cache_index(
        verify=train_dataset.require_cache_ready
    )
    val_dataset.build_image_cache_index(
        verify=val_dataset.require_cache_ready
    )

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=prefetch_factor,
        query_budget_batching=(
            query_budget_batching
        ),
        random_seed=random_seed or 0,
    )
    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        prefetch_factor=prefetch_factor,
        query_budget_batching=(
            query_budget_batching
        ),
        random_seed=random_seed or 0,
    )

    return train_loader, val_loader



if __name__ == "__main__":
    train_loader, val_loader = build_dataloaders(
        train_image_dir="path/to/datasets/images/train",
        train_anno_dir="path/to/datasets/labels/train",
        val_image_dir="path/to/datasets/images/val",
        val_anno_dir="path/to/datasets/labels/val",
        batch_size=48,
        image_size=(512, 512),
        num_workers=0,
        pin_memory=False,
        cache_images=False,
        prebuild_image_cache=False,
    )

    batch = next(iter(train_loader))
    print("unique_images:", tuple(batch["unique_images"].shape))
    print("captions:", len(batch["captions"]))
    print("regions:", batch["num_regions"])
    print("unique targets:", batch["num_unique_targets"])
    print("first caption:", batch["captions"][0])
    print("first positive spans:", batch["positive_char_spans"][0])
