import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF


class ShipGroundingDataset(Dataset):
    def __init__(
        self,
        image_dir,
        anno_paths,
        image_size=(640, 640),
        use_main_colors=True,
        use_text_aug=True,
        max_text_aug_per_image=1,
        random_seed=None
    ):
        self.image_dir = image_dir
        self.anno_paths = anno_paths
        self.image_size = image_size
        self.use_main_colors = use_main_colors
        self.use_text_aug = use_text_aug
        self.max_text_aug_per_image = max_text_aug_per_image

        if random_seed is not None:
            random.seed(random_seed)

        self.samples = []
        self.image_paths = []
        self.annos = []

        for anno_path in self.anno_paths:
            with open(anno_path, "r", encoding="utf-8") as f:
                anns = json.load(f)

            if len(anns) == 0:
                continue

            source_name = anns[0]["source_name"]
            image_path = os.path.join(self.image_dir, source_name + ".jpg")

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

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
                            "group_source": "main_colors"
                        })

            if self.use_text_aug:
                for query_text, obj_indices in text_aug_groups.items():
                    if len(obj_indices) > 0:
                        self.samples.append({
                            "anno_idx": anno_idx,
                            "query_text": query_text,
                            "obj_indices": obj_indices,
                            "group_source": "query_texts_aug"
                        })

    def __len__(self):
        return len(self.samples)

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
        sampled_keys = random.sample(keys, k=min(max_samples, len(keys)))

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
            "null"
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
            "灰", "橘", "棕", "紫", "銀", "金"
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
            f"船體是{color}的船"
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

    def resize_boxes(self, boxes, orig_size, new_size):
        orig_h, orig_w = orig_size
        new_h, new_w = new_size

        if boxes.numel() == 0:
            return boxes

        boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_w / orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_h / orig_h)

        return boxes

    def normalize_xyxy(self, boxes, image_size):
        h, w = image_size

        if boxes.numel() == 0:
            return boxes

        boxes = boxes.clone()
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / w
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / h

        return boxes

    def xyxy_to_cxcywh(self, boxes):
        if boxes.numel() == 0:
            return boxes

        x1, y1, x2, y2 = boxes.unbind(dim=-1)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        return torch.stack([cx, cy, w, h], dim=-1)

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

        boxes = []
        labels = []

        for obj in anns:
            boxes.append(obj["bbox_xyxy"])
            labels.append(obj["class_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        target_h, target_w = self.image_size

        image = TF.resize(image, [target_h, target_w])
        image = TF.to_tensor(image)

        boxes_xyxy = self.resize_boxes(
            boxes,
            orig_size=(orig_h, orig_w),
            new_size=(target_h, target_w)
        )

        boxes_norm_xyxy = self.normalize_xyxy(
            boxes_xyxy,
            image_size=(target_h, target_w)
        )

        boxes_cxcywh = self.xyxy_to_cxcywh(boxes_norm_xyxy)

        obj_indices_tensor = torch.tensor(obj_indices, dtype=torch.long)

        target_boxes = boxes_cxcywh[obj_indices_tensor]
        target_labels = labels[obj_indices_tensor]

        return {
            "image": image,
            "boxes": boxes_xyxy,
            "target_boxes": target_boxes,
            "target_bbox": target_boxes[0],
            "labels": labels,
            "target_labels": target_labels,
            "target_label": target_labels[0],
            "query_text": query_text,
            "group_source": group_source,
            "image_size": (target_h, target_w),
            "orig_size": (orig_h, orig_w),
            "image_path": image_path,
            "anno_path": anno_path,
            "obj_indices": obj_indices_tensor,
            "source_name": anns[0].get("source_name", "")
        }


def grounding_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch], dim=0)

    boxes_per_image = [item["boxes"] for item in batch]

    target_boxes_per_image = [item["target_boxes"] for item in batch]
    target_labels_per_image = [item["target_labels"] for item in batch]

    target_bboxes = torch.stack(
        [item["target_bbox"] for item in batch],
        dim=0
    )

    target_labels = torch.stack(
        [item["target_label"] for item in batch],
        dim=0
    )

    query_texts = [item["query_text"] for item in batch]
    group_sources = [item["group_source"] for item in batch]

    image_sizes = [item["image_size"] for item in batch]
    orig_sizes = [item["orig_size"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    anno_paths = [item["anno_path"] for item in batch]
    obj_indices = [item["obj_indices"] for item in batch]
    source_names = [item["source_name"] for item in batch]

    return {
        "images": images,
        "boxes_per_image": boxes_per_image,
        "target_bboxes": target_bboxes,
        "target_labels": target_labels,
        "target_boxes_per_image": target_boxes_per_image,
        "target_labels_per_image": target_labels_per_image,
        "query_texts": query_texts,
        "group_sources": group_sources,
        "image_sizes": image_sizes,
        "orig_sizes": orig_sizes,
        "image_paths": image_paths,
        "anno_paths": anno_paths,
        "obj_indices": obj_indices,
        "source_names": source_names
    }


def list_json_files(anno_dir):
    return sorted([
        os.path.join(anno_dir, f)
        for f in os.listdir(anno_dir)
        if f.lower().endswith(".json")
    ])


def build_dataloaders(
    train_image_dir,
    train_anno_dir,
    val_image_dir,
    val_anno_dir,
    batch_size=4,
    image_size=(640, 640),
    num_workers=0,
    max_text_aug_per_image=1,
    random_seed=None
):
    train_anno_paths = list_json_files(train_anno_dir)
    val_anno_paths = list_json_files(val_anno_dir)

    train_dataset = ShipGroundingDataset(
        image_dir=train_image_dir,
        anno_paths=train_anno_paths,
        image_size=image_size,
        use_main_colors=True,
        use_text_aug=True,
        max_text_aug_per_image=max_text_aug_per_image,
        random_seed=random_seed
    )

    val_dataset = ShipGroundingDataset(
        image_dir=val_image_dir,
        anno_paths=val_anno_paths,
        image_size=image_size,
        use_main_colors=True,
        use_text_aug=True,
        max_text_aug_per_image=max_text_aug_per_image,
        random_seed=random_seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=grounding_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=grounding_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False
    )

    return train_loader, val_loader