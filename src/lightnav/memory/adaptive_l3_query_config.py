"""Independent settings for the Adaptive L3 Query sidecar.

The sidecar is opt-in and defaults to off.  It does not own a model, a weight
file or an endpoint: the controller reuses the already-loaded local LightNav
inference engine.  Every field below is an engineering bound (prompt sizing,
output length, evidence filter), never a research threshold, similarity weight
or cooldown that decides *when* to query; the controller makes that decision.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

ADAPTIVE_L3_PROTOCOL_VERSION = "adaptive_l3_query_v1"


@dataclass(frozen=True)
class AdaptiveL3QueryConfig:
    """Settings for one Adaptive L3 Query arm (``enabled=False`` by default)."""

    enabled: bool = False
    controller_max_new_tokens: int = 24
    max_query_chars: int = 256
    # Reuses spatial_l3.min_injection_confidence semantics for an Evidence record;
    # stored here so a run records the exact value the validator used.
    min_evidence_confidence: float = 0.5
    protocol_version: str = ADAPTIVE_L3_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("adaptive_l3_query.enabled must be boolean")
        for name in ("controller_max_new_tokens", "max_query_chars"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"adaptive_l3_query.{name} must be a positive integer")
        if (
            not isinstance(self.min_evidence_confidence, (int, float))
            or not math.isfinite(float(self.min_evidence_confidence))
            or not 0.0 <= float(self.min_evidence_confidence) <= 1.0
        ):
            raise ValueError(
                "adaptive_l3_query.min_evidence_confidence must be finite and in [0, 1]"
            )
        if self.protocol_version != ADAPTIVE_L3_PROTOCOL_VERSION:
            raise ValueError("unsupported adaptive_l3_query.protocol_version")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
