from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence

import torch


VALID_COMPONENT_MODES = {
    "constant",
    "linear",
    "cosine",
    "freeze",
}


def _component_name(group_name: str) -> str:
    return str(group_name).split("_", 1)[0].strip().lower()


def _resolve_epoch_position(
    value: float,
    total_epochs: int,
) -> float:
    value = float(value)

    if value < 0.0:
        raise ValueError(
            f"Schedule epoch position must be >= 0, got {value}"
        )

    if value <= 1.0:
        return value * float(total_epochs)

    return value


def normalize_component_schedules(
    schedules: Mapping[str, Sequence[Any]],
    *,
    total_epochs: int,
) -> Dict[str, List[Any]]:
    normalized: Dict[str, List[Any]] = {}

    for component in ("vision", "text", "transformer", "head"):
        if component not in schedules:
            raise KeyError(
                f"Missing optimizer component schedule: {component}"
            )

        raw = schedules[component]

        if not isinstance(raw, (list, tuple)) or len(raw) != 5:
            raise ValueError(
                f"optim.components.{component} must be "
                "[mode, max_lr, min_lr, start_epoch, end_epoch]"
            )

        mode = str(raw[0]).strip().lower()

        if mode not in VALID_COMPONENT_MODES:
            raise ValueError(
                f"Unsupported schedule mode for {component}: {mode}. "
                f"Expected one of {sorted(VALID_COMPONENT_MODES)}"
            )

        max_lr = float(raw[1])
        min_lr = float(raw[2])
        start_epoch = _resolve_epoch_position(
            float(raw[3]),
            total_epochs,
        )
        end_epoch = _resolve_epoch_position(
            float(raw[4]),
            total_epochs,
        )

        if max_lr < 0.0 or min_lr < 0.0:
            raise ValueError(
                f"{component} learning rates must be >= 0"
            )

        if min_lr > max_lr:
            raise ValueError(
                f"{component} min_lr must be <= max_lr"
            )

        if end_epoch < start_epoch:
            raise ValueError(
                f"{component} end_epoch must be >= start_epoch"
            )

        normalized[component] = [
            mode,
            max_lr,
            min_lr,
            start_epoch,
            end_epoch,
        ]

    return normalized


class ComponentLRScheduler:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        component_schedules: Mapping[str, Sequence[Any]],
        warmup_steps: int,
        total_steps: int,
        steps_per_epoch: int,
        total_epochs: int,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.total_steps = int(total_steps)
        self.steps_per_epoch = int(steps_per_epoch)
        self.total_epochs = int(total_epochs)
        self.step_num = 0

        if self.total_steps <= 0:
            raise ValueError(
                f"total_steps must be > 0, got {self.total_steps}"
            )

        if self.steps_per_epoch <= 0:
            raise ValueError(
                f"steps_per_epoch must be > 0, got {self.steps_per_epoch}"
            )

        if self.total_epochs <= 0:
            raise ValueError(
                f"total_epochs must be > 0, got {self.total_epochs}"
            )

        if not 0 <= self.warmup_steps < self.total_steps:
            raise ValueError(
                "warmup_steps must satisfy "
                "0 <= warmup_steps < total_steps"
            )

        self.component_schedules = normalize_component_schedules(
            component_schedules,
            total_epochs=self.total_epochs,
        )

        self.base_lrs: List[float] = []

        for group in self.optimizer.param_groups:
            component = _component_name(group.get("name", ""))

            if component not in self.component_schedules:
                raise KeyError(
                    "Optimizer group name must start with one of "
                    "vision/text/transformer/head, got "
                    f"{group.get('name')!r}"
                )

            max_lr = float(
                self.component_schedules[component][1]
            )

            group["lr"] = max_lr
            group["initial_lr"] = max_lr
            group["component"] = component
            self.base_lrs.append(max_lr)

    def _warmup_factor(self, step_num: int) -> float:
        if self.warmup_steps <= 0:
            return 1.0

        if step_num >= self.warmup_steps:
            return 1.0

        return max(
            0.0,
            float(step_num) / float(self.warmup_steps),
        )

    def _scheduled_lr(
        self,
        component: str,
        step_num: int,
    ) -> float:
        (
            mode,
            max_lr,
            min_lr,
            start_epoch,
            end_epoch,
        ) = self.component_schedules[component]

        current_epoch = 1.0 + (
            float(max(step_num, 1) - 1)
            / float(self.steps_per_epoch)
        )

        if mode == "constant":
            return float(max_lr)

        if current_epoch < float(start_epoch):
            return float(max_lr)

        if current_epoch >= float(end_epoch):
            if mode == "freeze":
                return 0.0

            return float(min_lr)

        duration = max(
            float(end_epoch) - float(start_epoch),
            1e-12,
        )
        progress = (
            current_epoch - float(start_epoch)
        ) / duration
        progress = min(max(progress, 0.0), 1.0)

        if mode in {"cosine", "freeze"}:
            alpha = 0.5 - 0.5 * math.cos(
                math.pi * progress
            )
        elif mode == "linear":
            alpha = progress
        else:
            alpha = 0.0

        return (
            float(max_lr)
            + (float(min_lr) - float(max_lr)) * alpha
        )

    def _apply_step_lr(self, step_num: int) -> None:
        warmup_factor = self._warmup_factor(step_num)

        for group in self.optimizer.param_groups:
            component = str(group["component"])
            target_lr = self._scheduled_lr(
                component,
                step_num,
            )
            group["lr"] = float(target_lr) * warmup_factor

    def step(self) -> None:
        if self.step_num < self.total_steps:
            self.step_num += 1

        self._apply_step_lr(self.step_num)

    def get_lr(self) -> List[float]:
        return [
            float(group["lr"])
            for group in self.optimizer.param_groups
        ]

    def get_group_lrs(self) -> Dict[str, float]:
        result: Dict[str, float] = {}

        for group in self.optimizer.param_groups:
            component = str(group["component"])
            result[component] = max(
                result.get(component, 0.0),
                float(group["lr"]),
            )

        return result

    def state_dict(self) -> Dict[str, Any]:
        return {
            "name": self.__class__.__name__,
            "warmup_steps": int(self.warmup_steps),
            "total_steps": int(self.total_steps),
            "steps_per_epoch": int(self.steps_per_epoch),
            "total_epochs": int(self.total_epochs),
            "step_num": int(self.step_num),
            "base_lrs": list(self.base_lrs),
            "component_schedules": {
                key: list(value)
                for key, value in self.component_schedules.items()
            },
        }

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
    ) -> None:
        if not isinstance(state_dict, dict):
            raise TypeError(
                "scheduler state_dict must be a dict"
            )

        self.step_num = min(
            max(int(state_dict.get("step_num", 0)), 0),
            self.total_steps,
        )

        saved_schedules = state_dict.get(
            "component_schedules"
        )

        if isinstance(saved_schedules, dict):
            self.component_schedules = (
                normalize_component_schedules(
                    saved_schedules,
                    total_epochs=self.total_epochs,
                )
            )

        self._apply_step_lr(self.step_num)
