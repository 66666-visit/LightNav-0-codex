"""Independent settings for frozen-policy local graph memory context."""

from __future__ import annotations

from dataclasses import dataclass

LOCAL_GRAPH_PROTOCOL_VERSION = "local_graph_v1"
LOCAL_GRAPH_NO_L2_PROTOCOL_VERSION = "local_graph_no_l2_v1"


@dataclass(frozen=True)
class LocalGraphConfig:
    enabled: bool = False
    scope: str = "episode_local"
    prompt_max_chars: int = 320
    max_landmarks: int = 2
    reactivation_steps: int = 8
    mid_enabled: bool = True
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
        if type(self.mid_enabled) is not bool:
            raise ValueError("local_graph.mid_enabled must be boolean")
        if not self.mid_enabled and not self.enabled:
            raise ValueError("local_graph.mid_enabled=false requires local_graph.enabled")
        expected = (LOCAL_GRAPH_PROTOCOL_VERSION if self.mid_enabled
                    else LOCAL_GRAPH_NO_L2_PROTOCOL_VERSION)
        if self.protocol_version is None:
            object.__setattr__(self, "protocol_version", expected)
        elif self.protocol_version != expected:
            raise ValueError("unsupported local graph protocol_version")
