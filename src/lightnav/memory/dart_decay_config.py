"""Configuration for the DART-VLN read-side memory decay adaptation.

``off`` leaves the original LightNav memory readout byte-for-byte unchanged.
``decay_only`` enables the paper's recency/repetition/novelty weight at read
time. Write-side DART modes (``update_only`` and ``full``) are deliberately not
implemented: they would rewrite stored memory and are outside this project.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DART_DECAY_PROTOCOL_VERSION = "dart_decay_v1"
_DART_MODES = frozenset({"off", "decay_only"})


@dataclass(frozen=True)
class DartDecayConfig:
    """Paper defaults for the decay-only controller.

    ``rho`` is the novelty EMA coefficient, ``decay_lambda`` the recency rate,
    ``alpha`` the repetition coefficient and ``w_min``/``w_max`` the final
    clamp. The values match the DART-VLN release and paper configuration.
    """

    mode: str = "off"
    rho: float = 0.5
    decay_lambda: float = 0.12
    alpha: float = 0.15
    w_min: float = 0.35
    w_max: float = 1.0
    protocol_version: str = DART_DECAY_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.mode not in _DART_MODES:
            raise ValueError(
                "dart_decay.mode must be 'off' or 'decay_only'; the DART write-side "
                "update_only/full ablations are intentionally unsupported"
            )
        if self.protocol_version != DART_DECAY_PROTOCOL_VERSION:
            raise ValueError("unsupported DART decay protocol_version")
        for name in ("rho", "decay_lambda", "alpha", "w_min", "w_max"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"dart_decay.{name} must be finite")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("dart_decay.rho must be in [0, 1]")
        if self.decay_lambda < 0.0:
            raise ValueError("dart_decay.decay_lambda must be nonnegative")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("dart_decay.alpha must be in [0, 1]")
        if not 0.0 < self.w_min <= self.w_max <= 1.0:
            raise ValueError("dart_decay requires 0 < w_min <= w_max <= 1")

    @property
    def enabled(self) -> bool:
        return self.mode == "decay_only"
