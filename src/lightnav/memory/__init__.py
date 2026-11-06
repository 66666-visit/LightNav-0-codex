"""Episode-local explicit memory for Habitat navigation.

RGB is passed to an optional extractor in memory only; records never contain frames.
Imports are lazy so schema and retrieval tools remain usable without NumPy.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DetectedObject": "lightnav.memory.schemas",
    "LayerMemoryItem": "lightnav.memory.schemas",
    "MemoryFusion": "lightnav.memory.memory_fusion",
    "MemoryManager": "lightnav.memory.manager",
    "MemoryContext": "lightnav.memory.schemas",
    "MemoryEvent": "lightnav.memory.schemas",
    "MemoryQuery": "lightnav.memory.schemas",
    "ObservationPacket": "lightnav.memory.schemas",
    "ObservationEvidence": "lightnav.memory.schemas",
    "SceneEdge": "lightnav.memory.schemas",
    "SceneExtraction": "lightnav.memory.schemas",
    "SceneNode": "lightnav.memory.schemas",
    "SemanticIndex": "lightnav.memory.semantic_index",
    "WorkingMemory": "lightnav.memory.schemas",
}

__all__ = [
    "DetectedObject",
    "LayerMemoryItem",
    "MemoryContext",
    "MemoryEvent",
    "MemoryFusion",
    "MemoryManager",
    "MemoryQuery",
    "ObservationPacket",
    "ObservationEvidence",
    "SceneEdge",
    "SceneExtraction",
    "SceneNode",
    "SemanticIndex",
    "WorkingMemory",
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(module_name), name)
