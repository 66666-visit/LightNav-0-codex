"""Independent runtime settings for the evidence-driven RGM V3 backend."""

from __future__ import annotations

from dataclasses import dataclass


RGM_V3_PROTOCOL_VERSION = "rgm_v3_v1"


@dataclass(frozen=True)
class RGMV3Config:
    enabled: bool = False
    scope: str = "episode_local"
    prompt_max_records: int = 3
    prompt_max_chars: int = 400
    low_confidence_threshold: float = 0.5
    protocol_version: str = RGM_V3_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.scope not in {"episode_local", "scene_persistent"}:
            raise ValueError("rgm_v3.scope must be episode_local or scene_persistent")
        if self.prompt_max_records < 1 or self.prompt_max_chars < 1:
            raise ValueError("rgm_v3 prompt budgets must be positive")
        if not 0.0 <= self.low_confidence_threshold <= 1.0:
            raise ValueError("rgm_v3.low_confidence_threshold must be in [0, 1]")
        if self.protocol_version != RGM_V3_PROTOCOL_VERSION:
            raise ValueError("unsupported RGM V3 protocol_version")
