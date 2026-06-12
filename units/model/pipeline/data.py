import os
import json
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

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


class ShipGroundingDataset(Dataset):
    """
    Dataset output design:

    image:
        Tensor, shape [3, H, W], full image input.

    query_text:
        str, single grounding query.

    target_boxes:
        Tensor, shape [num_matched_objects, 4], normalized xyxy in [0, 1].

    target_labels:
        Tensor, shape [num_matched_objects].

    This matches your current model:

        out = model(images, query_texts)
        loss = criterion(out["bbox"], out["score"], targets)

    where targets is:

        [
            {
                "boxes": Tensor[num_gt, 4], normalized xyxy,
                "labels": Tensor[num_gt]
            },
            ...
        ]
    """

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
                else:
                    continue

            anno_idx = len(self.annos)
            self.image_paths.append(image_path)
            self.annos.append(anns)

            main_color_groups = self.build_main_color_groups(anns)
            text_aug_groups = self.build_text_aug_groups(anns)

            text_aug_groups = self.sample_text_aug_groups(
                text_aug_groups,
                max_samples=self.max_text_aug_per_image
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

    def __len__(self):
        return len(self.samples)

    def find_image_path(self, source_name: str) -> Optional[str]:
        """
        Old version assumes source_name + ".jpg".
        This version keeps that behavior, but also supports png/bmp/webp/etc.
        """

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

        return {
            key: text_aug_groups[key]
            for key in sampled_keys
        }

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
        """
        Keep your old aug text mechanism:

        1. Read query_texts_aug from each object.
        2. Extract color words from aug text.
        3. Match all objects that share those mentioned colors.
        4. One aug text can correspond to multiple GT boxes.
        """

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
        """
        Ensure x1 <= x2 and y1 <= y2.
        """

        if boxes.numel() == 0:
            return boxes

        x1 = torch.minimum(boxes[:, 0], boxes[:, 2])
        y1 = torch.minimum(boxes[:, 1], boxes[:, 3])
        x2 = torch.maximum(boxes[:, 0], boxes[:, 2])
        y2 = torch.maximum(boxes[:, 1], boxes[:, 3])

        return torch.stack([x1, y1, x2, y2], dim=-1)

    def resize_boxes_xyxy(self, boxes, orig_size, new_size):
        """
        Input:
            boxes: pixel xyxy under original image size.

        Output:
            boxes: pixel xyxy under resized image size.
        """

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
        """
        Convert pixel xyxy to normalized xyxy.

        x1, x2 are divided by image width.
        y1, y2 are divided by image height.
        """

        h, w = image_size

        if boxes.numel() == 0:
            return boxes

        boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / float(w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / float(h)

        return boxes.clamp(0.0, 1.0)

    def valid_box_mask_pixel(self, boxes):
        """
        Check box validity under pixel xyxy.
        """

        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool)

        wh = boxes[:, 2:] - boxes[:, :2]

        valid = (
            (wh[:, 0] >= self.min_box_size)
            &
            (wh[:, 1] >= self.min_box_size)
        )

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

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        anno_idx = sample_info["anno_idx"]
        query_text = sample_info["query_text"]
        obj_indices = sample_info["obj_indices"]
        group_source = sample_info["group_source"]

        image_path = self.image_paths[anno_idx]
        anno_path = self.anno_paths[anno_idx]
        anns = self.annos[anno_idx]

        image = Image.open(image_path).convert("RGB")
        orig_w, orig_h = image.size

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

        obj_indices_tensor = torch.tensor(obj_indices, dtype=torch.long)

        target_boxes_pixel = boxes_resized_xyxy[obj_indices_tensor]
        target_boxes_norm = boxes_norm_xyxy[obj_indices_tensor]
        target_labels = labels[obj_indices_tensor]

        valid_target_mask = self.valid_box_mask_pixel(target_boxes_pixel)

        target_boxes_pixel = target_boxes_pixel[valid_target_mask]
        target_boxes_norm = target_boxes_norm[valid_target_mask]
        target_labels = target_labels[valid_target_mask]
        obj_indices_tensor = obj_indices_tensor[valid_target_mask]

        image = self.preprocess_image(image)

        target = {
            "boxes": target_boxes_norm,          # normalized xyxy, for loss
            "labels": target_labels,
            "boxes_pixel": target_boxes_pixel,  # pixel xyxy, for debug / visualization
            "obj_indices": obj_indices_tensor,
            "image_size": torch.tensor([target_h, target_w], dtype=torch.long),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
        }

        return {
            "image": image,

            # All boxes of this image.
            "boxes": boxes_norm_xyxy,              # normalized xyxy
            "boxes_pixel": boxes_resized_xyxy,     # pixel xyxy
            "labels": labels,

            # Matched boxes for this query.
            "target_boxes": target_boxes_norm,     # normalized xyxy
            "target_boxes_pixel": target_boxes_pixel,
            "target_labels": target_labels,
            "target": target,

            "query_text": query_text,
            "group_source": group_source,

            "image_size": (target_h, target_w),
            "orig_size": (orig_h, orig_w),

            "image_path": image_path,
            "anno_path": anno_path,
            "obj_indices": obj_indices_tensor,
            "source_name": anns[0].get("source_name", ""),
        }


def grounding_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)

    query_texts = [item["query_text"] for item in batch]

    targets = [item["target"] for item in batch]

    boxes_per_image = [item["boxes"] for item in batch]
    boxes_pixel_per_image = [item["boxes_pixel"] for item in batch]
    labels_per_image = [item["labels"] for item in batch]

    target_boxes_per_image = [item["target_boxes"] for item in batch]
    target_boxes_pixel_per_image = [item["target_boxes_pixel"] for item in batch]
    target_labels_per_image = [item["target_labels"] for item in batch]

    group_sources = [item["group_source"] for item in batch]

    image_sizes = [item["image_size"] for item in batch]
    orig_sizes = [item["orig_size"] for item in batch]

    image_paths = [item["image_path"] for item in batch]
    anno_paths = [item["anno_path"] for item in batch]
    obj_indices = [item["obj_indices"] for item in batch]
    source_names = [item["source_name"] for item in batch]

    return {
        "images": images,

        # Directly feed to model(img, texts)
        "query_texts": query_texts,

        # Directly feed to GroundingLoss
        "targets": targets,

        # Debug / visualization / analysis
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
    )

    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = build_dataloaders(
        train_image_dir="path/to/train/images",
        train_anno_dir="path/to/train/jsons",
        val_image_dir="path/to/val/images",
        val_anno_dir="path/to/val/jsons",
        batch_size=2,
        image_size=(640, 640),
        num_workers=0,
        max_text_aug_per_image=1,
        random_seed=42,
        pin_memory=False,
    )

    batch = next(iter(train_loader))

    print("images:", batch["images"].shape)
    print("query_texts:", batch["query_texts"])
    print("num targets:", len(batch["targets"]))

    for i, target in enumerate(batch["targets"]):
        print(f"[{i}] boxes:", target["boxes"].shape)
        print(f"[{i}] labels:", target["labels"].shape)
        print(f"[{i}] boxes min/max:", target["boxes"].min().item(), target["boxes"].max().item())