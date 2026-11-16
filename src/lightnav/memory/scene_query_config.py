"""Runtime settings for the optional, independent memory query API."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass


def _enabled(value: str | None) -> bool:
    if value is None:
        return False
    if value.casefold() in {"1", "true", "yes", "on"}:
        return True
    if value.casefold() in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError("MEMORY_QUERY_ENABLED must be a boolean")


@dataclass(frozen=True)
class SceneQueryConfig:
    enabled: bool = False
    base_url: str = ""
    model: str = ""
    timeout: float = 30.0
    max_rounds: int = 4

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("scene query enabled must be boolean")
        if not isinstance(self.base_url, str):
            raise ValueError("scene query endpoint must be a string")
        if not isinstance(self.model, str):
            raise ValueError("scene query model must be a string")
        if not isinstance(self.timeout, (int, float)) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("MEMORY_QUERY_TIMEOUT must be finite and positive")
        if type(self.max_rounds) is not int or self.max_rounds < 1:
            raise ValueError("MEMORY_QUERY_MAX_ROUNDS must be a positive integer")

    @classmethod
    def from_env(cls, *, enabled: bool | None = None) -> SceneQueryConfig:
        active = _enabled(os.getenv("MEMORY_QUERY_ENABLED")) if enabled is None else enabled
        if not active:
            # Query-only settings must not affect an unchanged LightNav run.
            return cls()
        return cls(
            enabled=True,
            base_url=os.getenv("MEMORY_QUERY_BASE_URL", ""),
            model=os.getenv("MEMORY_QUERY_MODEL", ""),
            timeout=float(os.getenv("MEMORY_QUERY_TIMEOUT", "30")),
            max_rounds=int(os.getenv("MEMORY_QUERY_MAX_ROUNDS", "4")),
        )
