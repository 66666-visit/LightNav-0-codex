"""CPU contracts for L1-failure-gated L3 CLIP query and Prompt formatting."""

from __future__ import annotations

from html import escape

import numpy as np
import pytest

from lightnav.memory.conceptgraphs_lite import (
    ObjectMap,
    ObjectNode,
    Pose3D,
    SpatialEvidence,
)
from lightnav.memory.l3_on_demand import (
    L1_EMPTY_STATUSES,
    L3QueryController,
    build_l3_query_context,
)


class _TextEncoder:
    def __init__(self, feature=(1.0, 0.0)) -> None:
        self.feature = np.asarray(feature, dtype=np.float32)
        self.text_calls = 0

    def encode_image(self, rgb, boxes):
        raise AssertionError("controller tests attach features directly")

    def encode_text(self, texts):
        self.text_calls += 1
        return np.asarray([self.feature], dtype=np.float32)


def _add_object(
    object_map: ObjectMap, object_id: str, label: str, feature,
    *, center=(2.0, 0.0, -2.0), step: int = 0,
) -> ObjectNode:
    evidence_id = f"ev-{object_id}"
    node = ObjectNode(
        object_id, label, center,
        ((center[0] - 0.5, -0.5, center[2] - 0.5),
         (center[0] + 0.5, 0.5, center[2] + 0.5)),
        3, [evidence_id], "observed", np.zeros((5, 3)),
    )
    object_map.objects[object_id] = node
    object_map.evidence[evidence_id] = SpatialEvidence(
        evidence_id, "scene", "episode", 0, step, label, 0.88,
        "yolo11n-seg", 20, 5, "observed",
    )
    object_map.attach_clip_features({evidence_id: feature})
    return node


def _map() -> ObjectMap:
    object_map = ObjectMap()
    _add_object(object_map, "sofa-1", "sofa", [1.0, 0.0], step=1)
    return object_map


def test_l1_empty_statuses_are_explicit_and_narrow() -> None:
    assert L1_EMPTY_STATUSES == frozenset({"no_candidates", "budget_excluded"})


def test_l1_satisfied_never_calls_the_text_encoder() -> None:
    encoder = _TextEncoder()
    controller = L3QueryController(encoder)
    result = controller.maybe_query(
        _map(), step=2, instruction="find the sofa", l1_status="built",
    )
    assert result is None
    assert encoder.text_calls == 0
    assert controller.last_diagnostic["skipped_reason"] == "l1_satisfied"


def test_l1_reactivation_suppression_does_not_trigger_l3() -> None:
    encoder = _TextEncoder()
    controller = L3QueryController(encoder)
    result = controller.maybe_query(
        _map(), step=2, instruction="find the sofa",
        l1_status="reactivation_suppressed",
    )
    assert result is None
    assert encoder.text_calls == 0
    assert controller.last_diagnostic["skipped_reason"] == "l1_reactivation_suppressed"


def test_l1_empty_triggers_one_query_and_reuses_it() -> None:
    encoder = _TextEncoder()
    controller = L3QueryController(encoder)
    object_map = _map()
    result = controller.maybe_query(
        object_map, step=2, instruction="find the sofa", l1_status="no_candidates",
    )
    assert result is not None
    assert result.matches[0].object_id == "sofa-1"
    assert encoder.text_calls == 1
    assert controller.last_diagnostic["trigger"] == "no_candidates"
    assert controller.last_diagnostic["candidate_count"] == 1

    again = controller.maybe_query(
        object_map, step=3, instruction="find the sofa", l1_status="no_candidates",
    )
    assert again is None
    assert encoder.text_calls == 1
    assert controller.last_diagnostic["skipped_reason"] == "map_unchanged_since_last_query"


def test_new_visual_evidence_reopens_the_query() -> None:
    encoder = _TextEncoder()
    controller = L3QueryController(encoder)
    object_map = _map()
    controller.maybe_query(
        object_map, step=2, instruction="find the sofa", l1_status="no_candidates",
    )
    node = object_map.objects["sofa-1"]
    node.evidence_ids.append("ev-second")
    object_map.evidence["ev-second"] = SpatialEvidence(
        "ev-second", "scene", "episode", 0, 5, "sofa", 0.9,
        "yolo11n-seg", 22, 6, "observed",
    )
    object_map.attach_clip_features({"ev-second": [0.0, 1.0]})
    result = controller.maybe_query(
        object_map, step=6, instruction="find the sofa", l1_status="no_candidates",
    )
    assert result is not None
    assert encoder.text_calls == 2


def test_new_instruction_reopens_the_query() -> None:
    encoder = _TextEncoder()
    controller = L3QueryController(encoder)
    object_map = _map()
    controller.maybe_query(
        object_map, step=2, instruction="find the sofa", l1_status="no_candidates",
    )
    controller.maybe_query(
        object_map, step=3, instruction="find the chair", l1_status="no_candidates",
    )
    assert encoder.text_calls == 2


def test_empty_map_skips_without_encoding() -> None:
    encoder = _TextEncoder()
    controller = L3QueryController(encoder)
    result = controller.maybe_query(
        ObjectMap(), step=0, instruction="find the sofa", l1_status="no_candidates",
    )
    assert result is None
    assert encoder.text_calls == 0
    assert controller.last_diagnostic["skipped_reason"] == "empty_map"


def test_no_valid_match_returns_none_and_caches_the_empty_result() -> None:
    encoder = _TextEncoder([-1.0, 0.0])
    controller = L3QueryController(encoder)
    object_map = _map()
    assert controller.maybe_query(
        object_map, step=2, instruction="find the sofa", l1_status="no_candidates",
    ) is None
    assert controller.last_diagnostic["skipped_reason"] == "no_similarity_above_floor"
    assert encoder.text_calls == 1
    assert controller.maybe_query(
        object_map, step=3, instruction="find the sofa", l1_status="no_candidates",
    ) is None
    assert encoder.text_calls == 1


def test_query_error_is_reported_without_escaping() -> None:
    class _Broken(_TextEncoder):
        def encode_text(self, texts):
            raise RuntimeError("local encoder failed")

    controller = L3QueryController(_Broken())
    assert controller.maybe_query(
        _map(), step=2, instruction="find the sofa", l1_status="no_candidates",
    ) is None
    assert controller.last_diagnostic["skipped_reason"] == "query_error"
    assert "local encoder failed" in controller.last_diagnostic["error"]


def test_context_contains_only_top_k_spatial_evidence() -> None:
    encoder = _TextEncoder([1.0, 0.0])
    object_map = _map()
    _add_object(
        object_map, "sofa-2", "sofa", [0.6, 0.8], center=(0.0, 0.0, -3.0), step=4,
    )
    controller = L3QueryController(encoder, top_k=1)
    result = controller.maybe_query(
        object_map, step=5, instruction="find the sofa", l1_status="no_candidates",
    )
    assert result is not None and len(result.matches) == 1
    context = build_l3_query_context(
        result, object_map, agent_pose=Pose3D((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        max_chars=320,
    )
    assert context is not None
    assert context.text.startswith("[Navigation Memory]\n")
    lines = context.text.splitlines()
    assert len(lines) == 2
    assert "L3 CLIP cos 1.000: sofa" in lines[1]
    assert "at world (2.0,0.0,-2.0)m" in lines[1]
    assert "2.8m, -45deg (left +)" in lines[1]
    assert "3obs" in lines[1]
    assert "passage unknown" in lines[1]
    assert len(context.layer_items) == 1
    item = context.layer_items[0]
    assert item.kind == "l3_clip_query_object"
    assert item.metadata["object_id"] == "sofa-1"
    assert item.metadata["similarity"] == pytest.approx(1.0)
    assert item.metadata["detector_confidence"] == pytest.approx(0.88)
    assert item.metadata["serialized_text"] == escape(lines[1], quote=False)
    assert item.evidence_refs[0].event_id == "ev-sofa-1"
    assert item.step_id == 1


def test_context_omits_matches_that_do_not_fit_the_existing_budget() -> None:
    object_map = _map()
    result = L3QueryController(_TextEncoder()).maybe_query(
        object_map, step=2, instruction="find the sofa", l1_status="no_candidates",
    )
    assert result is not None
    context = build_l3_query_context(
        result, object_map,
        agent_pose=Pose3D((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        max_chars=len("[Navigation Memory]") + 2,
    )
    assert context is None


def test_context_is_none_without_a_valid_match() -> None:
    assert build_l3_query_context(
        None, _map(), agent_pose=Pose3D((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        max_chars=320,
    ) is None


def test_spatial_l3_config_keeps_off_mode_backward_compatible() -> None:
    from lightnav.memory.spatial_l3_config import SpatialL3Config

    config = SpatialL3Config()
    assert config.mode == "off"
    assert config.clip_model_path == ""
    assert config.clip_top_k == 3
    assert config.clip_precision == "fp32"
    assert config.clip_batch_size == 8
    with pytest.raises(ValueError, match="clip_model_path"):
        SpatialL3Config(
            mode="on_demand", seg_model_path="/fake/seg.pt",
        )
    with pytest.raises(ValueError):
        SpatialL3Config(mode="on_demand", clip_model_path="/fake/clip.pt")
    with pytest.raises(ValueError, match="clip_precision"):
        SpatialL3Config(
            mode="on_demand", seg_model_path="/fake/seg.pt",
            clip_model_path="/fake/clip.pt", clip_precision="int8",
        )
    with pytest.raises(ValueError, match="CUDA"):
        SpatialL3Config(
            mode="on_demand", seg_model_path="/fake/seg.pt",
            clip_model_path="/fake/clip.pt", clip_precision="fp16",
        )
    with pytest.raises(ValueError, match="clip_batch_size"):
        SpatialL3Config(
            mode="on_demand", seg_model_path="/fake/seg.pt",
            clip_model_path="/fake/clip.pt", clip_batch_size=0,
        )


def test_manager_on_demand_requires_encoder_and_allows_l1_original_only() -> None:
    from lightnav.memory.local_graph_config import LocalGraphConfig
    from lightnav.memory.manager import MemoryManager
    from lightnav.memory.spatial_l3_config import SpatialL3Config

    class _Extractor:
        def reset_episode(self, episode_id, instruction):
            pass

        def close_episode(self):
            pass

        def extract(self, rgb, step_id):
            return []

    spatial = SpatialL3Config(
        mode="on_demand", seg_model_path="/fake/seg.pt",
        clip_model_path="/fake/clip.pt",
    )
    with pytest.raises(ValueError, match="CLIP encoder"):
        MemoryManager(
            strategy="hierarchical",
            local_graph_config=LocalGraphConfig(enabled=True),
            spatial_l3_config=spatial,
            spatial_extractor=_Extractor(),
        )
    with pytest.raises(ValueError, match="l1_original_only"):
        MemoryManager(
            strategy="hierarchical",
            local_graph_config=LocalGraphConfig(enabled=True),
            spatial_l3_config=spatial,
            spatial_extractor=_Extractor(),
            clip_encoder=_TextEncoder(),
            prompt_mode="l1_organized_only",
        )
    manager = MemoryManager(
        strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True),
        spatial_l3_config=spatial,
        spatial_extractor=_Extractor(),
        clip_encoder=_TextEncoder(),
        prompt_mode="l1_original_only",
    )
    assert manager.l3_query_controller is not None
    assert manager.plan.mode == "l1_original_only"


def test_budget_excluded_l1_status_also_triggers_l3() -> None:
    encoder = _TextEncoder()
    controller = L3QueryController(encoder)
    result = controller.maybe_query(
        _map(), step=2, instruction="find the sofa", l1_status="budget_excluded",
    )
    assert result is not None
    assert encoder.text_calls == 1
    assert controller.last_diagnostic["trigger"] == "budget_excluded"


def test_three_compact_matches_fit_the_original_320_char_budget() -> None:
    from lightnav.memory.memory_context_builder import (
        audit_prompt_injection,
        inject_memory_context,
    )

    object_map = ObjectMap()
    _add_object(object_map, "a-sofa", "sofa", [1.0, 0.0], center=(2.0, 0.0, -2.0))
    _add_object(object_map, "b-table", "table", [0.8, 0.6], center=(1.0, 0.0, -3.0))
    _add_object(object_map, "c-window", "window", [0.6, 0.8], center=(0.0, 0.0, -4.0))
    controller = L3QueryController(_TextEncoder([1.0, 0.0]), top_k=3)
    result = controller.maybe_query(
        object_map, step=5,
        instruction=(
            "Walk past the sofa and turn right at the wooden table, "
            "then stop near the window."
        ),
        l1_status="no_candidates",
    )
    assert result is not None
    context = build_l3_query_context(
        result, object_map,
        agent_pose=Pose3D((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
        max_chars=320,
    )
    assert context is not None
    assert len(context.layer_items) == 3
    assert len(context.text) <= 320

    base_prompt = (
        "Imagine you are a robot. <navigation_task>Walk past the sofa and turn "
        "right at the wooden table, then stop near the window.</navigation_task>. "
        "Predict the trajectory."
    )
    base = {"conversations": [
        {"from": "human", "value": base_prompt},
        {"from": "gpt", "value": ""},
    ]}
    augmented = inject_memory_context(base, context, max_chars=320)
    injected = augmented["conversations"][0]["value"]
    assert injected.count("</navigation_task>.") == 1
    assert "[Navigation Memory]" in injected
    assert injected.startswith(base_prompt.split("</navigation_task>.", 1)[0])
    assert injected.endswith(
        base_prompt.split("</navigation_task>.", 1)[1].lstrip(" ")
    )
    audit = audit_prompt_injection(base, augmented, context, max_chars=320)
    assert audit["roundtrip_ok"] is True


def test_manager_records_query_trigger_rate_and_skip_reasons() -> None:
    from lightnav.memory.local_graph_config import LocalGraphConfig
    from lightnav.memory.manager import MemoryManager
    from lightnav.memory.spatial_l3_config import SpatialL3Config

    class _Extractor:
        def reset_episode(self, episode_id, instruction):
            pass

        def close_episode(self):
            pass

        def extract(self, rgb, step_id):
            return []

    manager = MemoryManager(
        strategy="hierarchical",
        local_graph_config=LocalGraphConfig(enabled=True),
        spatial_l3_config=SpatialL3Config(
            mode="on_demand", seg_model_path="/fake/seg.pt",
            clip_model_path="/fake/clip.pt",
        ),
        spatial_extractor=_Extractor(),
        clip_encoder=_TextEncoder(),
    )
    manager.reset_episode("episode", "instruction", scene_id="scene")
    manager._record_l1_status("no_candidates", 0)
    manager._record_l1_status("budget_excluded", 1)
    manager._record_l3_query({"query_executed": True, "candidate_count": 1})
    manager._record_l3_query({
        "query_executed": False,
        "skipped_reason": "map_unchanged_since_last_query",
    })
    summary = manager.l3_query_stats_summary()
    assert summary["steps"] == 2
    assert summary["l1_empty_steps"] == 2
    assert summary["queries_attempted"] == 1
    assert summary["queries_with_match"] == 1
    assert summary["query_trigger_rate"] == pytest.approx(0.5)
    assert summary["match_rate_of_queries"] == pytest.approx(1.0)
    assert summary["injection_rate_of_l1_empty"] == pytest.approx(0.5)
    assert summary["skipped_reasons"] == {"map_unchanged_since_last_query": 1}
    manager.close_episode()
    assert manager.last_episode_l3_query_stats["queries_attempted"] == 1


def test_explicit_landmark_query_text_is_not_parsed_or_truncated() -> None:
    encoder = _TextEncoder([1.0, 0.0])
    controller = L3QueryController(encoder)
    phrase = "the window beside the wooden table"
    result = controller.maybe_query(
        _map(), step=2, instruction="Walk past the sofa and stop at the window.",
        l1_status="no_candidates", query_text=phrase,
    )
    assert result is not None
    assert result.query == phrase
    assert controller.last_diagnostic["query"] == phrase
    assert controller.last_diagnostic["query_source"] == "explicit"
    assert controller.last_diagnostic["instruction"].startswith("Walk past")
