from __future__ import annotations

from typing import Any, Dict

from units.model.cards.loss import (
    GroundingLoss,
    RankingGroundingLoss,
)


def build_grounding_loss_from_config(
    config: Dict[str, Any],
) -> RankingGroundingLoss:
    return RankingGroundingLoss.from_config(config)


__all__ = [
    "GroundingLoss",
    "RankingGroundingLoss",
    "build_grounding_loss_from_config",
]
