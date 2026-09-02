from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import yaml
from PIL import Image, ImageDraw
from torchvision.ops import nms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF


def _find_project_root() -> Path:
    """
    Locate the LightDet project root.

    Expected project structure:
        LightDet/
        ├── units/
        │   ├── infer.py
        │   └── tool/
        │       └── card.py
        └── ...

    This allows infer.py to be executed directly from the project root:

        python3 units/infer.py

    or imported from another working directory.
    """
    candidates: List[Path] = []

    # 1. Search upward from this file.
    file_path = Path(__file__).resolve()
    candidates.extend(file_path.parents)

    # 2. Search upward from the current working directory as a fallback.
    cwd = Path.cwd().resolve()
    candidates.append(cwd)
    candidates.extend(cwd.parents)

    checked: set[Path] = set()

    for candidate in candidates:
        if candidate in checked:
            continue
        checked.add(candidate)

        card_path = candidate / "units" / "tool" / "card.py"
        if card_path.is_file():
            return candidate

    raise RuntimeError(
        "Unable to locate the LightDet project root. "
        "Expected to find 'units/tool/card.py' in this file's parent "
        "directories or the current working directory."
    )


PROJECT_ROOT = _find_project_root()
UNITS_ROOT = PROJECT_ROOT / "units"

# Add the project root so absolute imports such as
# `from units.tool.card import VisionTextModel` work when infer.py is
# executed directly with `python3 units/infer.py`.
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

print(f"Project Root : {PROJECT_ROOT}")
print(f"Units Root   : {UNITS_ROOT}")

from units.tool.card import VisionTextModel
from units.model.tool.runtime import (
    forward_model_batch,
    score_queries_for_char_spans,
)


ImageInput = Union[str, Path, Image.Image]


@dataclass(frozen=True)
class PhraseRequest:
    phrase: str
    char_spans: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class InferenceSettings:
    image_size: Tuple[int, int]
    score_threshold: float = 0.05
    top_k: int = 20
    use_nms: bool = False
    nms_iou_threshold: float = 0.5
    token_reduction: str = "mean"
    score_fusion: str = "geometric_mean"


class LightDetODVGInferencer:
    """Phrase-level ODVG inference using the unified LightDet card interface."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        *,
        model_config_path: Optional[Union[str, Path]] = None,
        train_config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        prefer_ema: bool = True,
        strict_checkpoint: bool = True,
        image_size: Optional[Union[int, Sequence[int]]] = None,
        use_amp: bool = True,
    ) -> None:
        self.project_root = PROJECT_ROOT
        self.checkpoint_path = self._resolve_path(checkpoint_path)
        self.model_config_path = self._resolve_path(
            model_config_path
            or self.project_root / "units/model/cards/config/model.yaml"
        )
        self.train_config_path = self._resolve_path(
            train_config_path
            or self.project_root / "units/model/cards/config/train.yaml"
        )

        self.device = self._resolve_device(device)
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.amp_dtype = self._resolve_amp_dtype(self.device)

        self.model_config = self._read_yaml_section(
            self.model_config_path,
            "model",
        )
        self.train_config = self._read_yaml(self.train_config_path)

        if image_size is None:
            image_size = (
                self.train_config
                .get("data", {})
                .get("image_size", 1024)
            )
        self.image_size = self._normalize_image_size(image_size)

        self.model = self._build_model(self.model_config)
        self.checkpoint_info = self._load_checkpoint(
            self.model,
            self.checkpoint_path,
            prefer_ema=prefer_ema,
            strict=strict_checkpoint,
        )
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(value: Optional[str]) -> torch.device:
        if value is not None:
            device = torch.device(value)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            if device.type == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is not available")
            return device

        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _resolve_amp_dtype(device: torch.device) -> torch.dtype:
        if device.type != "cuda":
            return torch.float32
        if (
            hasattr(torch.cuda, "is_bf16_supported")
            and torch.cuda.is_bf16_supported()
        ):
            return torch.bfloat16
        return torch.float16

    def _resolve_path(self, value: Union[str, Path]) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"File not found: {path}\n"
                "Check the path for accidental trailing characters "
                "(for example: '.ptß' instead of '.pt')."
            )
        return path

    @staticmethod
    def _read_yaml(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            value = yaml.safe_load(file) or {}
        if not isinstance(value, dict):
            raise TypeError(f"YAML root must be a mapping: {path}")
        return value

    @classmethod
    def _read_yaml_section(
        cls,
        path: Path,
        section: str,
    ) -> Dict[str, Any]:
        root = cls._read_yaml(path)
        value = root.get(section, {})
        if not isinstance(value, dict):
            raise TypeError(
                f"YAML section {section!r} must be a mapping: {path}"
            )
        return dict(value)

    @staticmethod
    def _normalize_image_size(
        value: Union[int, Sequence[int]],
    ) -> Tuple[int, int]:
        if isinstance(value, int):
            height = width = int(value)
        else:
            values = list(value)
            if len(values) != 2:
                raise ValueError(
                    "image_size must be an integer or [height, width]"
                )
            height, width = int(values[0]), int(values[1])
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid image_size: {(height, width)}")
        return height, width

    def _resolve_optional_project_path(
        self,
        value: Optional[Union[str, Path]],
    ) -> Optional[str]:
        if value is None:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return str(path.resolve()) if path.exists() else None

    def _build_model(self, config: Mapping[str, Any]) -> VisionTextModel:
        """
        Build the current LightDet FPN + staged query refinement model.

        Expected model.yaml structure:
            model:
              hidden_dim: ...
              backbone: {...}
              fpn: {...}
              image_projector: {...}
              ...
        """
        backbone_config = config.get("backbone")
        fpn_config = config.get("fpn")
        image_projector_config = config.get("image_projector")

        if not isinstance(backbone_config, Mapping):
            raise TypeError(
                "model.backbone must be a mapping, "
                f"got {type(backbone_config).__name__}"
            )

        if not isinstance(fpn_config, Mapping):
            raise TypeError(
                "model.fpn must be a mapping, "
                f"got {type(fpn_config).__name__}"
            )

        if not isinstance(image_projector_config, Mapping):
            raise TypeError(
                "model.image_projector must be a mapping, "
                f"got {type(image_projector_config).__name__}"
            )

        precomputed_path = self._resolve_optional_project_path(
            config.get("precomputed_bert_path")
        )

        model = VisionTextModel(
            backbone_config=dict(backbone_config),
            fpn_config=dict(fpn_config),
            image_projector_config=dict(image_projector_config),

            hidden_dim=int(config.get("hidden_dim", 256)),
            text_max_length=int(config.get("text_max_length", 256)),
            fusion_token_num=int(config.get("fusion_token_num", 16)),
            num_object_queries=int(config.get("num_object_queries", 100)),

            num_heads=int(config.get("num_heads", 8)),
            num_layers=int(config.get("num_layers", 2)),
            mlp_ratio=float(config.get("mlp_ratio", 3.5)),
            dropout=float(config.get("dropout", 0.1)),

            freeze_bert=bool(config.get("freeze_bert", True)),
            precomputed_bert_path=precomputed_path,

            use_auxiliary_head=bool(
                config.get("use_auxiliary_head", True)
            ),
            # Inference always uses the Main one-to-one branch.
            auxiliary_in_eval=False,
            initialize_aux_from_main=bool(
                config.get("initialize_aux_from_main", True)
            ),

            query_init_std=float(
                config.get("query_init_std", 0.02)
            ),
            query_group_init_std=float(
                config.get("query_group_init_std", 0.02)
            ),
            freeze_img_projection=bool(
                config.get("freeze_img_projection", False)
            ),

            staged_query_refinement=bool(
                config.get("staged_query_refinement", True)
            ),
            score_num_heads=int(
                config.get("score_num_heads", 8)
            ),
            score_num_layers=int(
                config.get("score_num_layers", 2)
            ),
            score_mlp_ratio=float(
                config.get("score_mlp_ratio", 3.0)
            ),
            score_dropout=float(
                config.get("score_dropout", 0.1)
            ),
            score_bbox_conditioning=bool(
                config.get("score_bbox_conditioning", True)
            ),
            score_bbox_detach=bool(
                config.get("score_bbox_detach", True)
            ),
            score_fusion=str(
                config.get("score_fusion", "geometric_mean")
            ),
            score_fusion_eps=float(
                config.get("score_fusion_eps", 1e-6)
            ),
        )

        return model

    @staticmethod
    def _torch_load(path: Path) -> Any:
        try:
            return torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            return torch.load(path, map_location="cpu")

    @staticmethod
    def _strip_state_prefixes(
        state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        prefixes = ("module.", "_orig_mod.")
        result: Dict[str, torch.Tensor] = {}
        for raw_key, value in state_dict.items():
            key = str(raw_key)
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if key.startswith(prefix):
                        key = key[len(prefix):]
                        changed = True
            result[key] = value
        return result

    @classmethod
    def _select_state_dict(
        cls,
        checkpoint: Any,
        *,
        prefer_ema: bool,
    ) -> Tuple[Dict[str, torch.Tensor], str]:
        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                f"Checkpoint must be a mapping, got {type(checkpoint)}"
            )

        candidates: List[Tuple[str, Any]] = []
        if prefer_ema:
            candidates.append(("ema", checkpoint.get("ema")))
        candidates.extend([
            ("model", checkpoint.get("model")),
            ("state_dict", checkpoint.get("state_dict")),
        ])
        if not prefer_ema:
            candidates.append(("ema", checkpoint.get("ema")))

        for name, candidate in candidates:
            if isinstance(candidate, Mapping) and candidate:
                state = {
                    str(key): value
                    for key, value in candidate.items()
                    if torch.is_tensor(value)
                }
                if state:
                    return cls._strip_state_prefixes(state), name

        direct = {
            str(key): value
            for key, value in checkpoint.items()
            if torch.is_tensor(value)
        }
        if direct:
            return cls._strip_state_prefixes(direct), "direct"

        raise KeyError(
            "Checkpoint contains no model state under ema/model/state_dict"
        )

    @classmethod
    def _load_checkpoint(
        cls,
        model: torch.nn.Module,
        checkpoint_path: Path,
        *,
        prefer_ema: bool,
        strict: bool,
    ) -> Dict[str, Any]:
        checkpoint = cls._torch_load(checkpoint_path)
        state_dict, source = cls._select_state_dict(
            checkpoint,
            prefer_ema=prefer_ema,
        )

        incompatible = model.load_state_dict(
            state_dict,
            strict=strict,
        )
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))

        info = {
            "source": source,
            "strict": bool(strict),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        }
        if isinstance(checkpoint, Mapping):
            info["epoch"] = checkpoint.get("epoch")
            info["best_metric"] = checkpoint.get("best_metric")
            info["best_metric_name"] = checkpoint.get(
                "best_metric_name"
            )
        return info

    @staticmethod
    def _load_image(image: ImageInput) -> Tuple[Image.Image, Optional[str]]:
        if isinstance(image, Image.Image):
            return image.convert("RGB"), None
        path = Path(image).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as loaded:
            return loaded.convert("RGB"), str(path)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        height, width = self.image_size
        resized = TF.resize(
            image,
            [height, width],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensor = TF.to_tensor(resized).unsqueeze(0)
        return tensor.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=True,
        )

    @staticmethod
    def find_phrase_spans(
        caption: str,
        phrase: str,
    ) -> Tuple[Tuple[int, int], ...]:
        caption = str(caption)
        phrase = str(phrase)
        if not phrase:
            raise ValueError("phrase must not be empty")

        spans: List[Tuple[int, int]] = []
        search_start = 0
        while True:
            start = caption.find(phrase, search_start)
            if start < 0:
                break
            end = start + len(phrase)
            spans.append((start, end))
            search_start = end

        if not spans:
            raise ValueError(
                f"Phrase {phrase!r} does not occur in caption {caption!r}"
            )
        return tuple(spans)

    @classmethod
    def prepare_phrases(
        cls,
        caption: str,
        phrases: Optional[Sequence[str]],
    ) -> List[PhraseRequest]:
        if phrases is None or len(phrases) == 0:
            phrases = [caption]

        requests: List[PhraseRequest] = []
        seen = set()
        for raw_phrase in phrases:
            phrase = str(raw_phrase).strip()
            if not phrase or phrase in seen:
                continue
            requests.append(
                PhraseRequest(
                    phrase=phrase,
                    char_spans=cls.find_phrase_spans(caption, phrase),
                )
            )
            seen.add(phrase)

        if not requests:
            raise ValueError("No valid phrase was provided")
        return requests

    @staticmethod
    def _resolve_model_outputs(
        outputs: Mapping[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes = outputs.get("bbox", outputs.get("main_bbox"))
        quality = outputs.get(
            "quality_logit",
            outputs.get("main_quality_logit"),
        )
        token_logits = outputs.get(
            "token_alignment_logits",
            outputs.get("main_token_alignment_logits"),
        )
        token_offsets = outputs.get("token_offsets")
        text_mask = outputs.get("alignment_text_mask")

        values = {
            "bbox/main_bbox": boxes,
            "quality_logit": quality,
            "token_alignment_logits": token_logits,
            "token_offsets": token_offsets,
            "alignment_text_mask": text_mask,
        }
        missing = [
            name for name, value in values.items()
            if not torch.is_tensor(value)
        ]
        if missing:
            raise RuntimeError(
                "Model output is missing ODVG tensors: "
                + ", ".join(missing)
            )

        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError(f"bbox must be [B,Q,4], got {tuple(boxes.shape)}")
        if quality.ndim != 3 or quality.shape[-1] != 1:
            raise ValueError(
                f"quality_logit must be [B,Q,1], got {tuple(quality.shape)}"
            )
        if token_logits.ndim != 3:
            raise ValueError(
                "token_alignment_logits must be [B,Q,L], got "
                f"{tuple(token_logits.shape)}"
            )
        if token_offsets.ndim != 3 or token_offsets.shape[-1] != 2:
            raise ValueError(
                f"token_offsets must be [B,L,2], got {tuple(token_offsets.shape)}"
            )
        if text_mask.ndim != 2:
            raise ValueError(
                f"alignment_text_mask must be [B,L], got {tuple(text_mask.shape)}"
            )

        return boxes, quality, token_logits, token_offsets, text_mask

    @staticmethod
    def _select_indices(
        scores: torch.Tensor,
        boxes: torch.Tensor,
        settings: InferenceSettings,
    ) -> torch.Tensor:
        valid = torch.nonzero(
            scores >= float(settings.score_threshold),
            as_tuple=False,
        ).flatten()
        if valid.numel() == 0:
            return valid

        order = torch.argsort(
            scores[valid],
            descending=True,
            stable=True,
        )
        valid = valid[order]

        if settings.use_nms and valid.numel() > 1:
            kept = nms(
                boxes[valid].float(),
                scores[valid].float(),
                float(settings.nms_iou_threshold),
            )
            valid = valid[kept]

        top_k = max(0, int(settings.top_k))
        if top_k > 0:
            valid = valid[:top_k]
        return valid

    @staticmethod
    def _box_to_pixels(
        box: torch.Tensor,
        width: int,
        height: int,
    ) -> List[float]:
        values = box.detach().float().clamp(0.0, 1.0).tolist()
        return [
            float(values[0] * width),
            float(values[1] * height),
            float(values[2] * width),
            float(values[3] * height),
        ]

    @torch.inference_mode()
    def infer(
        self,
        image: ImageInput,
        caption: str,
        phrases: Optional[Sequence[str]] = None,
        *,
        score_threshold: float = 0.05,
        top_k: int = 20,
        use_nms: bool = False,
        nms_iou_threshold: float = 0.5,
        token_reduction: str = "mean",
        score_fusion: str = "geometric_mean",
    ) -> Dict[str, Any]:
        caption = str(caption).strip()
        if not caption:
            raise ValueError("caption must not be empty")

        phrase_requests = self.prepare_phrases(caption, phrases)
        pil_image, image_path = self._load_image(image)
        original_width, original_height = pil_image.size
        image_tensor = self._preprocess(pil_image)

        settings = InferenceSettings(
            image_size=self.image_size,
            score_threshold=float(score_threshold),
            top_k=int(top_k),
            use_nms=bool(use_nms),
            nms_iou_threshold=float(nms_iou_threshold),
            token_reduction=str(token_reduction),
            score_fusion=str(score_fusion),
        )

        autocast_enabled = bool(self.use_amp)
        with torch.autocast(
            device_type=self.device.type,
            enabled=autocast_enabled,
            dtype=self.amp_dtype if autocast_enabled else None,
        ):
            outputs = forward_model_batch(
                model=self.model,
                images=image_tensor,
                query_texts=[caption],
                image_indices=None,
                return_aux=False,
            )

        (
            boxes_batch,
            quality_batch,
            token_logits_batch,
            token_offsets_batch,
            text_mask_batch,
        ) = self._resolve_model_outputs(outputs)

        boxes = boxes_batch[0].detach().float().clamp(0.0, 1.0)
        quality = quality_batch[0].detach().float()
        token_logits = token_logits_batch[0].detach().float()
        token_offsets = token_offsets_batch[0].detach()
        text_mask = text_mask_batch[0].detach()

        phrase_results: List[Dict[str, Any]] = []
        for phrase_request in phrase_requests:
            score_components = score_queries_for_char_spans(
                quality_logit=quality,
                token_alignment_logits=token_logits,
                token_offsets=token_offsets,
                char_spans=phrase_request.char_spans,
                valid_token_mask=text_mask,
                token_reduction=settings.token_reduction,
                score_fusion=settings.score_fusion,
                strict=True,
            )

            final_scores = score_components["final_score"]
            selected = self._select_indices(
                final_scores,
                boxes,
                settings,
            )

            detections: List[Dict[str, Any]] = []
            for query_index in selected.tolist():
                box_norm = boxes[query_index]
                detections.append({
                    "query_index": int(query_index),
                    "bbox_xyxy_normalized": [
                        float(value)
                        for value in box_norm.tolist()
                    ],
                    "bbox_xyxy_pixel": self._box_to_pixels(
                        box_norm,
                        original_width,
                        original_height,
                    ),
                    "quality_score": float(
                        score_components["quality_score"][query_index].item()
                    ),
                    "phrase_alignment_score": float(
                        score_components[
                            "phrase_alignment_score"
                        ][query_index].item()
                    ),
                    "final_score": float(
                        final_scores[query_index].item()
                    ),
                })

            phrase_results.append({
                "phrase": phrase_request.phrase,
                "char_spans": [
                    [int(start), int(end)]
                    for start, end in phrase_request.char_spans
                ],
                "positive_token_indices": torch.nonzero(
                    score_components["token_mask"],
                    as_tuple=False,
                ).flatten().cpu().tolist(),
                "num_detections": len(detections),
                "detections": detections,
            })

        return {
            "image_path": image_path,
            "original_size": {
                "width": int(original_width),
                "height": int(original_height),
            },
            "model_input_size": {
                "height": int(self.image_size[0]),
                "width": int(self.image_size[1]),
            },
            "caption": caption,
            "score_threshold": float(settings.score_threshold),
            "top_k": int(settings.top_k),
            "use_nms": bool(settings.use_nms),
            "nms_iou_threshold": float(settings.nms_iou_threshold),
            "token_reduction": settings.token_reduction,
            "score_fusion": settings.score_fusion,
            "device": str(self.device),
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_info": self.checkpoint_info,
            "phrases": phrase_results,
        }

    @staticmethod
    def render(
        image: ImageInput,
        result: Mapping[str, Any],
        *,
        line_width: int = 3,
    ) -> Image.Image:
        pil_image, _ = LightDetODVGInferencer._load_image(image)
        draw = ImageDraw.Draw(pil_image)
        palette = [
            (255, 64, 64),
            (64, 255, 64),
            (64, 128, 255),
            (255, 192, 64),
            (192, 64, 255),
            (64, 255, 255),
        ]

        for phrase_index, phrase_result in enumerate(
            result.get("phrases", [])
        ):
            color = palette[phrase_index % len(palette)]
            for detection in phrase_result.get("detections", []):
                x1, y1, x2, y2 = detection["bbox_xyxy_pixel"]
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline=color,
                    width=max(1, int(line_width)),
                )
                label = (
                    f"P{phrase_index} Q{detection['query_index']} "
                    f"F={detection['final_score']:.3f} "
                    f"Q={detection['quality_score']:.3f} "
                    f"A={detection['phrase_alignment_score']:.3f}"
                )
                text_box = draw.textbbox((x1, y1), label)
                text_height = text_box[3] - text_box[1]
                label_y = max(0.0, y1 - text_height - 4)
                draw.rectangle(
                    [
                        x1,
                        label_y,
                        x1 + (text_box[2] - text_box[0]) + 4,
                        label_y + text_height + 4,
                    ],
                    fill=color,
                )
                draw.text(
                    (x1 + 2, label_y + 2),
                    label,
                    fill=(0, 0, 0),
                )
        return pil_image

# ---------------------------------------------------------------------------
# YOLO-style / function-call interface
# ---------------------------------------------------------------------------

class LightDet:
    """
    Function-call interface for LightDet inference.

    Example:
        model = LightDet(
            model="/path/to/model.yaml"
        )

        results = model.predict(
            weights="/path/to/best.pt",
            source="/path/to/image.jpg",
            caption="紅色的船",
            phrases=["紅色的船"],
            imgsz=1024,
            device=0,
            conf=0.30,
            quality_thr=0.0,
            alignment_thr=0.0,
            top_k=20,
            use_nms=False,
            nms_iou_threshold=0.5,
            token_reduction="mean",
            score_fusion="geometric_mean",
            project="runs/predict",
            name="lightdet_odvg",
            prefer_ema=False,
            save=True,
            save_json=True,
        )
    """

    def __init__(
        self,
        model: Union[str, Path],
        train_config: Optional[Union[str, Path]] = None,
    ) -> None:
        self.model_config_path = Path(model).expanduser().resolve()

        if train_config is None:
            default_train_config = (
                PROJECT_ROOT / "units/model/cards/config/train.yaml"
            )
            self.train_config_path = default_train_config.resolve()
        else:
            self.train_config_path = Path(
                train_config
            ).expanduser().resolve()

        if not self.model_config_path.is_file():
            raise FileNotFoundError(
                f"Model config not found: {self.model_config_path}"
            )

        if not self.train_config_path.is_file():
            raise FileNotFoundError(
                f"Train config not found: {self.train_config_path}"
            )

        self._inferencer: Optional[LightDetODVGInferencer] = None
        self._loaded_signature: Optional[Tuple[Any, ...]] = None

    @staticmethod
    def _normalize_device_arg(
        device: Optional[Any],
    ) -> Optional[str]:
        if device is None:
            return None

        if isinstance(device, torch.device):
            return str(device)

        if isinstance(device, int):
            if device < 0:
                return "cpu"
            return f"cuda:{device}"

        value = str(device).strip()
        if not value:
            return None

        if value.isdigit():
            return f"cuda:{value}"

        return value

    def clear_inference_cache(self) -> None:
        self._inferencer = None
        self._loaded_signature = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_inferencer(
        self,
        *,
        weights: Union[str, Path],
        device: Optional[Any],
        prefer_ema: bool,
        imgsz: int,
        use_amp: bool,
        strict_checkpoint: bool,
    ) -> Tuple[LightDetODVGInferencer, bool]:
        checkpoint_path = Path(weights).expanduser().resolve()

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Check the filename for accidental trailing characters "
                "(for example '.ptß' instead of '.pt')."
            )

        resolved_device = self._normalize_device_arg(device)

        signature = (
            str(checkpoint_path),
            str(self.model_config_path),
            str(self.train_config_path),
            resolved_device,
            bool(prefer_ema),
            int(imgsz),
            bool(use_amp),
            bool(strict_checkpoint),
        )

        if (
            self._inferencer is not None
            and self._loaded_signature == signature
        ):
            return self._inferencer, True

        print("\n[LightDet] Loading inference model")
        print(f"  weights : {checkpoint_path}")
        print(f"  device  : {resolved_device or 'auto'}")

        self._inferencer = LightDetODVGInferencer(
            checkpoint_path=checkpoint_path,
            model_config_path=self.model_config_path,
            train_config_path=self.train_config_path,
            device=resolved_device,
            prefer_ema=bool(prefer_ema),
            strict_checkpoint=bool(strict_checkpoint),
            image_size=int(imgsz),
            use_amp=bool(use_amp),
        )
        self._loaded_signature = signature

        return self._inferencer, False

    @staticmethod
    def _filter_thresholds(
        result: Dict[str, Any],
        *,
        quality_thr: float,
        alignment_thr: float,
    ) -> Dict[str, Any]:
        for phrase_result in result.get("phrases", []):
            detections = phrase_result.get("detections", [])

            filtered = [
                detection
                for detection in detections
                if float(
                    detection.get("quality_score", 0.0)
                ) >= float(quality_thr)
                and float(
                    detection.get("phrase_alignment_score", 0.0)
                ) >= float(alignment_thr)
            ]

            phrase_result["detections"] = filtered
            phrase_result["num_detections"] = len(filtered)

        result["quality_threshold"] = float(quality_thr)
        result["alignment_threshold"] = float(alignment_thr)
        return result

    def predict(
        self,
        weights: str,
        source: Union[str, Path, Image.Image],
        caption: str,
        phrases: Union[str, Sequence[str]],
        imgsz: int = 1024,
        device: Optional[Any] = None,
        conf: float = 0.05,
        quality_thr: float = 0.0,
        alignment_thr: float = 0.0,
        top_k: int = 20,
        use_nms: bool = False,
        nms_iou_threshold: float = 0.5,
        token_reduction: str = "mean",
        score_fusion: str = "geometric_mean",
        include_all_occurrences: bool = True,
        project: str = "runs/predict",
        name: str = "lightdet_odvg",
        prefer_ema: bool = True,
        save: bool = True,
        save_json: bool = True,
        use_amp: bool = True,
        strict_checkpoint: bool = True,
    ) -> Dict[str, Any]:
        """
        Run phrase-level LightDet inference.

        Model structure is read from model.yaml. The current FPN,
        image-projector and staged query-refinement configuration is passed
        into VisionTextModel by LightDetODVGInferencer._build_model().
        """
        if not 0.0 <= float(conf) <= 1.0:
            raise ValueError(
                f"conf must be within [0, 1], got {conf}"
            )

        if not 0.0 <= float(quality_thr) <= 1.0:
            raise ValueError(
                f"quality_thr must be within [0, 1], got {quality_thr}"
            )

        if not 0.0 <= float(alignment_thr) <= 1.0:
            raise ValueError(
                "alignment_thr must be within [0, 1], "
                f"got {alignment_thr}"
            )

        if int(top_k) <= 0:
            raise ValueError(
                f"top_k must be > 0, got {top_k}"
            )

        if int(imgsz) <= 0:
            raise ValueError(
                f"imgsz must be > 0, got {imgsz}"
            )

        if not 0.0 <= float(nms_iou_threshold) <= 1.0:
            raise ValueError(
                "nms_iou_threshold must be within [0, 1], "
                f"got {nms_iou_threshold}"
            )

        normalized_caption = str(caption).strip()
        if not normalized_caption:
            raise ValueError("caption must not be empty")

        if isinstance(phrases, str):
            normalized_phrases = [phrases]
        else:
            normalized_phrases = [
                str(phrase).strip()
                for phrase in phrases
                if str(phrase).strip()
            ]

        if not normalized_phrases:
            normalized_phrases = [normalized_caption]

        # Preserve the original interface option. The lower-level inferencer
        # already resolves phrase spans against the complete caption.
        _ = bool(include_all_occurrences)

        inferencer, cache_hit = self._get_inferencer(
            weights=weights,
            device=device,
            prefer_ema=prefer_ema,
            imgsz=imgsz,
            use_amp=use_amp,
            strict_checkpoint=strict_checkpoint,
        )

        result = inferencer.infer(
            image=source,
            caption=normalized_caption,
            phrases=normalized_phrases,
            score_threshold=float(conf),
            top_k=int(top_k),
            use_nms=bool(use_nms),
            nms_iou_threshold=float(nms_iou_threshold),
            token_reduction=str(token_reduction),
            score_fusion=str(score_fusion),
        )

        result = self._filter_thresholds(
            result,
            quality_thr=float(quality_thr),
            alignment_thr=float(alignment_thr),
        )
        result["cache_hit"] = bool(cache_hit)

        output_dir = (
            Path(project).expanduser() / str(name)
        ).resolve()

        rendered_path: Optional[Path] = None
        json_path: Optional[Path] = None

        source_path: Optional[Path]
        if isinstance(source, (str, Path)):
            source_path = Path(source).expanduser().resolve()
        else:
            source_path = None

        if save or save_json:
            output_dir.mkdir(parents=True, exist_ok=True)

        if save:
            rendered = inferencer.render(source, result)

            if source_path is not None:
                suffix = source_path.suffix or ".jpg"
                stem = source_path.stem
            else:
                suffix = ".jpg"
                stem = "prediction"

            rendered_path = output_dir / f"{stem}_pred{suffix}"
            rendered.save(rendered_path)

        if save_json:
            if source_path is not None:
                stem = source_path.stem
            else:
                stem = "prediction"

            json_path = output_dir / f"{stem}_pred.json"
            json_path.write_text(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )

        result["saved_image"] = (
            str(rendered_path)
            if rendered_path is not None
            else None
        )
        result["saved_json"] = (
            str(json_path)
            if json_path is not None
            else None
        )

        return result


def main() -> None:
    """
    Example function-call usage.

    Edit the paths and query below if you want to run this file directly.
    No argparse / CLI parameters are used.
    """
    model = LightDet(
        model=(
            "path/to/your/model_config.yaml"
        )
    )

    model.predict(
        weights=(
            "path/to/your/model_weights.pt"
        ),
        source=(
            "path/to/your/image.jpg"
        ),
        caption="紅色黑色的船",
        phrases=[
            "紅色黑色的船",
        ],
        imgsz=1024,
        device=1,
        conf=0.60,
        quality_thr=0.3,
        alignment_thr=0.3,
        top_k=20,
        use_nms=False,
        nms_iou_threshold=0.5,
        token_reduction="mean",
        score_fusion="geometric_mean",
        include_all_occurrences=True,
        project="runs/predict",
        name="lightdet_odvg2",
        prefer_ema=False,
        save=True,
        save_json=True,
    )


if __name__ == "__main__":
    main()
