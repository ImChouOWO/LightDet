import os
import json
import random
import hashlib
import math
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from tqdm import tqdm


IMAGE_EXTS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
]

CACHE_VERSION = "lightdet_uint8_image_cache_v2"


class ShipGroundingDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        anno_paths: List[str],
        image_size: Tuple[int, int] = (640, 640),
        use_main_colors: bool = True,
        use_text_aug: bool = True,
        max_text_aug_per_image: Optional[int] = 1,
        random_seed: Optional[int] = None,
        normalize_boxes: bool = True,
        clip_boxes: bool = True,
        min_box_size: float = 1.0,
        image_mean: Optional[Tuple[float, float, float]] = None,
        image_std: Optional[Tuple[float, float, float]] = None,
        strict_image: bool = True,
        cache_images: bool = False,
        image_cache_dir: Optional[str] = None,
        prebuild_image_cache: bool = False,
    ):
        self.image_dir = image_dir
        self.anno_paths = anno_paths
        self.image_size = image_size

        self.use_main_colors = use_main_colors
        self.use_text_aug = use_text_aug
        self.max_text_aug_per_image = max_text_aug_per_image

        self.normalize_boxes = normalize_boxes
        self.clip_boxes = clip_boxes
        self.min_box_size = min_box_size

        self.image_mean = image_mean
        self.image_std = image_std

        self.strict_image = strict_image
        self.rng = random.Random(random_seed)

        self.cache_images = bool(cache_images)
        self.image_cache_dir = image_cache_dir
        self.prebuild_image_cache = bool(prebuild_image_cache)
        self.require_cache_ready = bool(self.cache_images and self.prebuild_image_cache)
        self.image_level_batching = bool(self.cache_images and self.prebuild_image_cache)
        self.image_cache_map = {}

        if self.cache_images:
            if self.image_cache_dir is None:
                target_h, target_w = self.image_size
                self.image_cache_dir = os.path.join(
                    self.image_dir,
                    f".lightdet_cache_uint8_{target_h}x{target_w}",
                )
            os.makedirs(self.image_cache_dir, exist_ok=True)

        self.samples = []
        self.image_paths = []
        self.annos = []

        for anno_path in self.anno_paths:
            with open(anno_path, "r", encoding="utf-8") as f:
                anns = json.load(f)

            if not isinstance(anns, list) or len(anns) == 0:
                continue

            source_name = anns[0].get("source_name", "")
            image_path = self.find_image_path(source_name)

            if image_path is None:
                if self.strict_image:
                    raise FileNotFoundError(
                        f"Image not found for source_name={source_name}, anno={anno_path}"
                    )
                continue

            anno_idx = len(self.annos)
            self.image_paths.append(image_path)
            self.annos.append(anns)

            main_color_groups = self.build_main_color_groups(anns)
            text_aug_groups = self.build_text_aug_groups(anns)
            text_aug_groups = self.sample_text_aug_groups(
                text_aug_groups,
                max_samples=self.max_text_aug_per_image,
            )

            if self.use_main_colors:
                for query_text, obj_indices in main_color_groups.items():
                    if len(obj_indices) > 0:
                        self.samples.append({
                            "anno_idx": anno_idx,
                            "query_text": query_text,
                            "obj_indices": obj_indices,
                            "group_source": "main_colors",
                        })

            if self.use_text_aug:
                for query_text, obj_indices in text_aug_groups.items():
                    if len(obj_indices) > 0:
                        self.samples.append({
                            "anno_idx": anno_idx,
                            "query_text": query_text,
                            "obj_indices": obj_indices,
                            "group_source": "query_texts_aug",
                        })

        self.samples_by_anno = [[] for _ in range(len(self.annos))]
        for sample in self.samples:
            self.samples_by_anno[sample["anno_idx"]].append(sample)

        self.image_level_indices = [
            idx for idx, samples in enumerate(self.samples_by_anno)
            if len(samples) > 0
        ]

        self.queries_per_image = max(
            1,
            int(math.ceil(len(self.samples) / max(1, len(self.image_level_indices))))
        )

    def __len__(self):
        if self.image_level_batching:
            return len(self.image_level_indices)
        return len(self.samples)

    def find_image_path(self, source_name: str) -> Optional[str]:
        if not isinstance(source_name, str) or not source_name.strip():
            return None

        source_name = source_name.strip()
        base, ext = os.path.splitext(source_name)
        candidates = []

        if ext:
            candidates.append(os.path.join(self.image_dir, source_name))
        else:
            candidates.append(os.path.join(self.image_dir, source_name + ".jpg"))
            for image_ext in IMAGE_EXTS:
                candidates.append(os.path.join(self.image_dir, source_name + image_ext))

        for path in candidates:
            if os.path.exists(path):
                return path

        return None

    def sample_text_aug_groups(self, text_aug_groups, max_samples=1):
        if not isinstance(text_aug_groups, dict):
            return {}
        if len(text_aug_groups) == 0:
            return {}
        if max_samples is None:
            return text_aug_groups
        if max_samples <= 0:
            return {}

        keys = list(text_aug_groups.keys())
        sampled_keys = self.rng.sample(keys, k=min(max_samples, len(keys)))

        return {key: text_aug_groups[key] for key in sampled_keys}

    def normalize_color(self, color):
        if not isinstance(color, str):
            return ""

        color = color.strip()

        alias = {
            "白": "白色",
            "黑": "黑色",
            "紅": "紅色",
            "藍": "藍色",
            "綠": "綠色",
            "黃": "黃色",
            "灰": "灰色",
            "橘": "橘色",
            "棕": "棕色",
            "紫": "紫色",
            "銀": "銀色",
            "金": "金色",
            "白色": "白色",
            "黑色": "黑色",
            "紅色": "紅色",
            "藍色": "藍色",
            "綠色": "綠色",
            "黃色": "黃色",
            "灰色": "灰色",
            "橘色": "橘色",
            "棕色": "棕色",
            "紫色": "紫色",
            "銀色": "銀色",
            "金色": "金色",
        }

        return alias.get(color, color)

    def get_obj_colors(self, obj):
        attributes = obj.get("attributes", {})
        colors = attributes.get("main_colors", [])

        if not isinstance(colors, list):
            return []

        results = []
        for color in colors:
            color = self.normalize_color(color)
            if color:
                results.append(color)

        return list(dict.fromkeys(results))

    def is_valid_query_text(self, text):
        if not isinstance(text, str):
            return False

        text = text.strip()
        if not text:
            return False

        bad_keywords = [
            "否、未觀察到對應結構",
            "未觀察到對應結構",
            "無法判斷",
            "不確定",
            "未知",
            "none",
            "null",
        ]

        for kw in bad_keywords:
            if kw in text:
                return False

        return True

    def extract_colors_from_text(self, text):
        if not isinstance(text, str):
            return []

        color_keywords = [
            "白色", "黑色", "紅色", "藍色", "綠色", "黃色",
            "灰色", "橘色", "棕色", "紫色", "銀色", "金色",
            "白", "黑", "紅", "藍", "綠", "黃",
            "灰", "橘", "棕", "紫", "銀", "金",
        ]

        found = []
        for color in color_keywords:
            if color in text:
                norm_color = self.normalize_color(color)
                if norm_color:
                    found.append(norm_color)

        return list(dict.fromkeys(found))

    def build_main_color_queries(self, color):
        return [
            f"{color}的船",
            f"含{color}的船",
            f"{color}船隻",
            f"船體是{color}的船",
        ]

    def build_main_color_groups(self, anns):
        color_to_obj_indices = {}

        for obj_idx, obj in enumerate(anns):
            colors = self.get_obj_colors(obj)
            for color in colors:
                if color not in color_to_obj_indices:
                    color_to_obj_indices[color] = []
                color_to_obj_indices[color].append(obj_idx)

        query_groups = {}
        for color, obj_indices in color_to_obj_indices.items():
            queries = self.build_main_color_queries(color)
            for query in queries:
                query_groups[query] = list(dict.fromkeys(obj_indices))

        return query_groups

    def build_text_aug_groups(self, anns):
        aug_text_to_colors = {}

        for obj in anns:
            texts_aug = obj.get("query_texts_aug", [])
            if not isinstance(texts_aug, list):
                continue

            for text in texts_aug:
                if not self.is_valid_query_text(text):
                    continue

                text = text.strip()
                mentioned_colors = self.extract_colors_from_text(text)

                if len(mentioned_colors) == 0:
                    continue

                if text not in aug_text_to_colors:
                    aug_text_to_colors[text] = set()

                for color in mentioned_colors:
                    aug_text_to_colors[text].add(color)

        query_groups = {}

        for aug_text, mentioned_colors in aug_text_to_colors.items():
            obj_indices = []

            for obj_idx, obj in enumerate(anns):
                obj_colors = set(self.get_obj_colors(obj))
                if len(obj_colors.intersection(mentioned_colors)) > 0:
                    obj_indices.append(obj_idx)

            if len(obj_indices) > 0:
                query_groups[aug_text] = list(dict.fromkeys(obj_indices))

        return query_groups

    def load_boxes_and_labels(self, anns):
        boxes = []
        labels = []

        for obj in anns:
            if "bbox_xyxy" not in obj:
                raise KeyError(
                    f"Missing bbox_xyxy in object. obj keys={list(obj.keys())}"
                )

            bbox = obj["bbox_xyxy"]

            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError(f"Invalid bbox_xyxy: {bbox}")

            boxes.append(bbox)
            labels.append(obj.get("class_id", 0))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return boxes, labels

    def reorder_xyxy(self, boxes):
        if boxes.numel() == 0:
            return boxes

        x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
        y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
        x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
        y2 = torch.maximum(boxes[:, 1], boxes[:, 3])

        return torch.stack([x1, y1, x2, y2], dim=-1)

    def resize_boxes_xyxy(self, boxes, orig_size, new_size):
        orig_h, orig_w = orig_size
        new_h, new_w = new_size

        if boxes.numel() == 0:
            return boxes

        boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_w / orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_h / orig_h)

        return boxes

    def clip_boxes_xyxy(self, boxes, image_size):
        h, w = image_size

        if boxes.numel() == 0:
            return boxes

        boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)

        return boxes

    def normalize_xyxy(self, boxes, image_size):
        h, w = image_size

        if boxes.numel() == 0:
            return boxes

        boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / float(w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / float(h)

        return boxes.clamp(0.0, 1.0)

    def valid_box_mask_pixel(self, boxes):
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool)

        wh = boxes[:, 2:] - boxes[:, :2]
        valid = (wh[:, 0] >= self.min_box_size) & (wh[:, 1] >= self.min_box_size)

        return valid

    def preprocess_image(self, image):
        target_h, target_w = self.image_size

        image = TF.resize(
            image,
            [target_h, target_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        image = TF.to_tensor(image)

        if self.image_mean is not None and self.image_std is not None:
            image = TF.normalize(
                image,
                mean=list(self.image_mean),
                std=list(self.image_std),
            )

        return image

    def preprocess_image_u8(self, image):
        target_h, target_w = self.image_size

        image = TF.resize(
            image,
            [target_h, target_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        return TF.pil_to_tensor(image).contiguous()

    def decode_cached_image(self, image_u8):
        if image_u8.dtype == torch.uint8:
            image = image_u8.float().div_(255.0)
        else:
            image = image_u8.float()

        if self.image_mean is not None and self.image_std is not None:
            image = TF.normalize(
                image,
                mean=list(self.image_mean),
                std=list(self.image_std),
            )

        return image

    def get_image_cache_key(self, image_path: str) -> str:
        stat = os.stat(image_path)

        payload = {
            "version": CACHE_VERSION,
            "path": os.path.abspath(image_path),
            "mtime_ns": int(stat.st_mtime_ns),
            "file_size": int(stat.st_size),
            "image_size": tuple(self.image_size),
        }

        payload_text = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
        )

        return hashlib.sha1(payload_text.encode("utf-8")).hexdigest()

    def get_image_cache_path(self, image_path: str) -> str:
        if image_path in self.image_cache_map:
            return self.image_cache_map[image_path]

        key = self.get_image_cache_key(image_path)
        return os.path.join(self.image_cache_dir, f"{key}.pt")

    def load_image_uncached(self, image_path: str):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            orig_w, orig_h = image.size
            image_tensor = self.preprocess_image(image)

        return image_tensor, int(orig_w), int(orig_h)

    def load_image_u8_uncached(self, image_path: str):
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            orig_w, orig_h = image.size
            image_u8 = self.preprocess_image_u8(image)

        return image_u8, int(orig_w), int(orig_h)

    def load_image_cached(self, image_path: str, allow_build: bool = False):
        if not self.cache_images:
            return self.load_image_uncached(image_path)

        cache_path = self.get_image_cache_path(image_path)

        if os.path.exists(cache_path):
            try:
                try:
                    obj = torch.load(cache_path, map_location="cpu", weights_only=False)
                except TypeError:
                    obj = torch.load(cache_path, map_location="cpu")

                if "image_u8" in obj:
                    image_tensor = self.decode_cached_image(obj["image_u8"])
                else:
                    image_tensor = obj["image"].float()

                orig_h, orig_w = obj["orig_size_hw"]

                return image_tensor, int(orig_w), int(orig_h)

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

        tmp_path = cache_path + f".tmp.{os.getpid()}.{random.getrandbits(64)}"

        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, cache_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

        image_tensor = self.decode_cached_image(image_u8)

        return image_tensor, int(orig_w), int(orig_h)

    def build_image_cache_index(self, verify=False):
        if not self.cache_images:
            return

        self.image_cache_map = {}
        image_paths = list(dict.fromkeys(self.image_paths))

        for image_path in image_paths:
            cache_path = self.get_image_cache_path(image_path)
            self.image_cache_map[image_path] = cache_path

        if verify:
            missing = []
            for image_path, cache_path in self.image_cache_map.items():
                if not os.path.exists(cache_path):
                    missing.append((image_path, cache_path))

            if len(missing) > 0:
                image_path, cache_path = missing[0]
                raise FileNotFoundError(
                    f"Image cache missing: {cache_path}, image={image_path}, missing_count={len(missing)}"
                )

    def build_image_cache(self, num_workers=8, desc="[Image Cache]"):
        if not self.cache_images:
            return

        self.build_image_cache_index(verify=False)

        image_paths = list(dict.fromkeys(self.image_paths))
        total = len(image_paths)

        if total == 0:
            return

        num_workers = int(num_workers)

        if num_workers <= 1:
            for image_path in tqdm(
                image_paths,
                total=total,
                desc=desc,
                dynamic_ncols=True,
            ):
                _ = self.load_image_cached(image_path, allow_build=True)
            return

        def worker(image_path):
            _ = self.load_image_cached(image_path, allow_build=True)
            return image_path

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, image_path) for image_path in image_paths]

            for future in tqdm(
                as_completed(futures),
                total=total,
                desc=desc,
                dynamic_ncols=True,
            ):
                future.result()

    def build_common_image_data(self, anno_idx):
        image_path = self.image_paths[anno_idx]
        anno_path = self.anno_paths[anno_idx]
        anns = self.annos[anno_idx]

        image, orig_w, orig_h = self.load_image_cached(image_path)

        target_h, target_w = self.image_size

        boxes_orig_xyxy, labels = self.load_boxes_and_labels(anns)
        boxes_orig_xyxy = self.reorder_xyxy(boxes_orig_xyxy)

        boxes_resized_xyxy = self.resize_boxes_xyxy(
            boxes_orig_xyxy,
            orig_size=(orig_h, orig_w),
            new_size=(target_h, target_w),
        )
        boxes_resized_xyxy = self.reorder_xyxy(boxes_resized_xyxy)

        if self.clip_boxes:
            boxes_resized_xyxy = self.clip_boxes_xyxy(
                boxes_resized_xyxy,
                image_size=(target_h, target_w),
            )

        boxes_norm_xyxy = self.normalize_xyxy(
            boxes_resized_xyxy,
            image_size=(target_h, target_w),
        )

        return {
            "image": image,
            "image_path": image_path,
            "anno_path": anno_path,
            "anns": anns,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "target_h": target_h,
            "target_w": target_w,
            "boxes_norm_xyxy": boxes_norm_xyxy,
            "boxes_resized_xyxy": boxes_resized_xyxy,
            "labels": labels,
        }

    def build_query_record(self, common, sample_info):
        obj_indices = sample_info["obj_indices"]

        boxes_norm_xyxy = common["boxes_norm_xyxy"]
        boxes_resized_xyxy = common["boxes_resized_xyxy"]
        labels = common["labels"]
        target_h = common["target_h"]
        target_w = common["target_w"]
        orig_h = common["orig_h"]
        orig_w = common["orig_w"]

        obj_indices_tensor = torch.tensor(obj_indices, dtype=torch.long)

        target_boxes_pixel = boxes_resized_xyxy[obj_indices_tensor]
        target_boxes_norm = boxes_norm_xyxy[obj_indices_tensor]
        target_labels = labels[obj_indices_tensor]

        valid_target_mask = self.valid_box_mask_pixel(target_boxes_pixel)

        target_boxes_pixel = target_boxes_pixel[valid_target_mask]
        target_boxes_norm = target_boxes_norm[valid_target_mask]
        target_labels = target_labels[valid_target_mask]
        obj_indices_tensor = obj_indices_tensor[valid_target_mask]

        target = {
            "boxes": target_boxes_norm,
            "labels": target_labels,
            "boxes_pixel": target_boxes_pixel,
            "obj_indices": obj_indices_tensor,
            "image_size": torch.tensor([target_h, target_w], dtype=torch.long),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
        }

        return {
            "boxes": boxes_norm_xyxy,
            "boxes_pixel": boxes_resized_xyxy,
            "labels": labels,
            "target_boxes": target_boxes_norm,
            "target_boxes_pixel": target_boxes_pixel,
            "target_labels": target_labels,
            "target": target,
            "query_text": sample_info["query_text"],
            "group_source": sample_info["group_source"],
            "image_size": (target_h, target_w),
            "orig_size": (orig_h, orig_w),
            "image_path": common["image_path"],
            "anno_path": common["anno_path"],
            "obj_indices": obj_indices_tensor,
            "source_name": common["anns"][0].get("source_name", ""),
        }

    def get_query_item(self, idx):
        sample_info = self.samples[idx]
        common = self.build_common_image_data(sample_info["anno_idx"])
        query_record = self.build_query_record(common, sample_info)
        query_record["image"] = common["image"]
        return query_record

    def get_image_level_item(self, idx):
        anno_idx = self.image_level_indices[idx]
        common = self.build_common_image_data(anno_idx)
        sample_infos = self.samples_by_anno[anno_idx]

        if len(sample_infos) > self.queries_per_image:
            sample_infos = self.rng.sample(sample_infos, k=self.queries_per_image)

        queries = [self.build_query_record(common, sample_info) for sample_info in sample_infos]

        return {
            "image_level": True,
            "image": common["image"],
            "queries": queries,
            "image_path": common["image_path"],
            "anno_path": common["anno_path"],
            "source_name": common["anns"][0].get("source_name", ""),
        }

    def __getitem__(self, idx):
        if self.image_level_batching:
            return self.get_image_level_item(idx)
        return self.get_query_item(idx)


def _collate_query_items(items):
    images = torch.stack([item["image"] for item in items], dim=0)

    query_texts = [item["query_text"] for item in items]
    targets = [item["target"] for item in items]

    boxes_per_image = [item["boxes"] for item in items]
    boxes_pixel_per_image = [item["boxes_pixel"] for item in items]
    labels_per_image = [item["labels"] for item in items]

    target_boxes_per_image = [item["target_boxes"] for item in items]
    target_boxes_pixel_per_image = [item["target_boxes_pixel"] for item in items]
    target_labels_per_image = [item["target_labels"] for item in items]

    group_sources = [item["group_source"] for item in items]

    image_sizes = [item["image_size"] for item in items]
    orig_sizes = [item["orig_size"] for item in items]

    image_paths = [item["image_path"] for item in items]
    anno_paths = [item["anno_path"] for item in items]
    obj_indices = [item["obj_indices"] for item in items]
    source_names = [item["source_name"] for item in items]

    return {
        "images": images,
        "query_texts": query_texts,
        "targets": targets,
        "boxes_per_image": boxes_per_image,
        "boxes_pixel_per_image": boxes_pixel_per_image,
        "labels_per_image": labels_per_image,
        "target_boxes_per_image": target_boxes_per_image,
        "target_boxes_pixel_per_image": target_boxes_pixel_per_image,
        "target_labels_per_image": target_labels_per_image,
        "group_sources": group_sources,
        "image_sizes": image_sizes,
        "orig_sizes": orig_sizes,
        "image_paths": image_paths,
        "anno_paths": anno_paths,
        "obj_indices": obj_indices,
        "source_names": source_names,
    }


def grounding_collate_fn(batch):
    unique_images = []
    query_texts = []
    image_indices = []
    targets = []

    image_paths = []
    anno_paths = []
    source_names = []

    for image_idx, item in enumerate(batch):
        image = item["image"]
        unique_images.append(image)

        queries = item["queries"]

        for q in queries:
            query_texts.append(q["query_text"])
            targets.append(q["target"])
            image_indices.append(image_idx)

            image_paths.append(item["image_path"])
            anno_paths.append(item["anno_path"])
            source_names.append(item["source_name"])

    return {
        "unique_images": torch.stack(unique_images, dim=0),
        "image_indices": torch.tensor(image_indices, dtype=torch.long),
        "query_texts": query_texts,
        "targets": targets,
        "image_paths": image_paths,
        "anno_paths": anno_paths,
        "source_names": source_names,
    }


def list_json_files(anno_dir):
    return sorted([
        os.path.join(anno_dir, f)
        for f in os.listdir(anno_dir)
        if f.lower().endswith(".json")
    ])


def build_dataloader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=False,
    prefetch_factor=4,
):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": grounding_collate_fn,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**kwargs)


def _resolve_loader_batch_size(dataset, query_batch_size):
    if getattr(dataset, "image_level_batching", False):
        return max(1, int(query_batch_size) // max(1, int(dataset.queries_per_image)))
    return int(query_batch_size)


def build_dataloaders(
    train_image_dir,
    train_anno_dir,
    val_image_dir,
    val_anno_dir,
    batch_size=4,
    image_size=(640, 640),
    num_workers=0,
    max_text_aug_per_image=1,
    random_seed=None,
    use_main_colors=True,
    use_text_aug=True,
    normalize_boxes=True,
    clip_boxes=True,
    min_box_size=1.0,
    image_mean=None,
    image_std=None,
    pin_memory=True,
    prefetch_factor=4,
    cache_images=False,
    image_cache_dir=None,
    prebuild_image_cache=False,
    cache_workers=8,
):
    train_anno_paths = list_json_files(train_anno_dir)
    val_anno_paths = list_json_files(val_anno_dir)

    train_dataset = ShipGroundingDataset(
        image_dir=train_image_dir,
        anno_paths=train_anno_paths,
        image_size=image_size,
        use_main_colors=use_main_colors,
        use_text_aug=use_text_aug,
        max_text_aug_per_image=max_text_aug_per_image,
        random_seed=random_seed,
        normalize_boxes=normalize_boxes,
        clip_boxes=clip_boxes,
        min_box_size=min_box_size,
        image_mean=image_mean,
        image_std=image_std,
        strict_image=True,
        cache_images=cache_images,
        image_cache_dir=image_cache_dir,
        prebuild_image_cache=prebuild_image_cache,
    )

    val_dataset = ShipGroundingDataset(
        image_dir=val_image_dir,
        anno_paths=val_anno_paths,
        image_size=image_size,
        use_main_colors=use_main_colors,
        use_text_aug=use_text_aug,
        max_text_aug_per_image=max_text_aug_per_image,
        random_seed=random_seed,
        normalize_boxes=normalize_boxes,
        clip_boxes=clip_boxes,
        min_box_size=min_box_size,
        image_mean=image_mean,
        image_std=image_std,
        strict_image=True,
        cache_images=cache_images,
        image_cache_dir=image_cache_dir,
        prebuild_image_cache=prebuild_image_cache,
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

    train_dataset.build_image_cache_index(verify=train_dataset.require_cache_ready)
    val_dataset.build_image_cache_index(verify=val_dataset.require_cache_ready)

    train_batch_size = _resolve_loader_batch_size(train_dataset, batch_size)
    val_batch_size = _resolve_loader_batch_size(val_dataset, batch_size)

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        prefetch_factor=prefetch_factor,
    )

    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        prefetch_factor=prefetch_factor,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = build_dataloaders(
        train_image_dir="path/to/train/images",
        train_anno_dir="path/to/train/jsons",
        val_image_dir="path/to/val/images",
        val_anno_dir="path/to/val/jsons",
        batch_size=48,
        image_size=(512, 512),
        num_workers=0,
        max_text_aug_per_image=1,
        random_seed=42,
        pin_memory=False,
        cache_images=True,
        image_cache_dir="/tmp/lightdet_image_cache_512",
        prebuild_image_cache=True,
        cache_workers=8,
    )

    batch = next(iter(train_loader))

    print("images:", batch["images"].shape)
    print("query_texts:", len(batch["query_texts"]))
    print("num targets:", len(batch["targets"]))
