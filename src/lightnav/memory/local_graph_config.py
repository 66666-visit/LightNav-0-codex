"""Independent settings for frozen-policy local graph memory context."""

from __future__ import annotations

from dataclasses import dataclass

LOCAL_GRAPH_PROTOCOL_VERSION = "local_graph_v1"


@dataclass(frozen=True)
class LocalGraphConfig:
    enabled: bool = False
    scope: str = "episode_local"
    prompt_max_chars: int = 320
    max_landmarks: int = 2
    reactivation_steps: int = 8
    protocol_version: str = LOCAL_GRAPH_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("local_graph.enabled must be boolean")
        if self.scope not in {"episode_local", "scene_persistent"}:
            raise ValueError("local_graph.scope must be episode_local or scene_persistent")
        for name in ("prompt_max_chars", "max_landmarks", "reactivation_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"local_graph.{name} must be a positive integer")
        if self.protocol_version != LOCAL_GRAPH_PROTOCOL_VERSION:
            raise ValueError("unsupported local graph protocol_version")
