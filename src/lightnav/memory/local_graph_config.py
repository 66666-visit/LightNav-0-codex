"""Independent settings for frozen-policy local graph memory context."""

from __future__ import annotations

import math
from dataclasses import dataclass

LOCAL_GRAPH_PROTOCOL_VERSION = "local_graph_v1"
LOCAL_GRAPH_SPATIAL_PROTOCOL_VERSION = "local_graph_spatial_v1"


@dataclass(frozen=True)
class LocalGraphConfig:
    enabled: bool = False
    scope: str = "episode_local"
    prompt_max_chars: int = 320
    max_landmarks: int = 2
    reactivation_steps: int = 8
    neighbor_radius_m: float = 3.0
    fusion_mode: str = "legacy"
    protocol_version: str | None = None

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("local_graph.enabled must be boolean")
        if self.scope not in {"episode_local", "scene_persistent"}:
            raise ValueError("local_graph.scope must be episode_local or scene_persistent")
        for name in ("prompt_max_chars", "max_landmarks", "reactivation_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"local_graph.{name} must be a positive integer")
        if (isinstance(self.neighbor_radius_m, bool)
                or not isinstance(self.neighbor_radius_m, (int, float))
                or not math.isfinite(self.neighbor_radius_m)
                or self.neighbor_radius_m <= 0):
            raise ValueError("local_graph.neighbor_radius_m must be positive and finite")
        if self.fusion_mode not in {"legacy", "spatial"}:
            raise ValueError("local_graph.fusion_mode must be legacy or spatial")
        expected = (LOCAL_GRAPH_PROTOCOL_VERSION if self.fusion_mode == "legacy"
                    else LOCAL_GRAPH_SPATIAL_PROTOCOL_VERSION)
        if self.protocol_version is None:
            object.__setattr__(self, "protocol_version", expected)
        elif self.protocol_version != expected:
            raise ValueError("unsupported local graph protocol_version")
