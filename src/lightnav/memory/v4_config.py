"""Independent runtime settings for selective RGM V4 memory fusion."""

from __future__ import annotations

from dataclasses import dataclass

RGM_V4_PROTOCOL_VERSION = "rgm_v4_v1"


@dataclass(frozen=True)
class RGMV4Config:
    enabled: bool = False
    scope: str = "episode_local"
    prompt_max_records: int = 3
    prompt_max_chars: int = 400
    revisit_enabled: bool = True
    protocol_version: str = RGM_V4_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.revisit_enabled, bool):
            raise ValueError("rgm_v4 enabled switches must be boolean")
        if self.scope not in {"episode_local", "scene_persistent"}:
            raise ValueError("rgm_v4.scope must be episode_local or scene_persistent")
        if (type(self.prompt_max_records) is not int or self.prompt_max_records < 1
                or type(self.prompt_max_chars) is not int or self.prompt_max_chars < 1):
            raise ValueError("rgm_v4 prompt budgets must be positive integers")
        if self.protocol_version != RGM_V4_PROTOCOL_VERSION:
            raise ValueError("unsupported RGM V4 protocol_version")
