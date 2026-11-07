"""CPU contract tests for local YOLO extraction; no weights or package download."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lightnav.memory.manager import MemoryManager
from lightnav.memory.schemas import DetectedObject, ObservationPacket, SceneExtraction
from lightnav.memory.yolo_scene_extractor import YOLOSceneExtractor


def _rgb() -> np.ndarray:
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[:, :, 0] = 220
    rgb[:, :, 2] = 30
    return rgb


class FakeYOLO:
    def __init__(self, output: list[tuple[int, float]] | None = None, *, fail: bool = False) -> None:
        self.output = output or []
        self.fail = fail
        self.calls: list[dict] = []

    def predict(self, *, source, conf, device, verbose, save):
        self.calls.append({
            "pixels": source.copy(), "conf": conf, "device": device,
            "verbose": verbose, "save": save,
        })
        if self.fail:
            raise RuntimeError("synthetic inference failure")
        boxes = SimpleNamespace(
            cls=np.asarray([item[0] for item in self.output], dtype=np.float32),
            conf=np.asarray([item[1] for item in self.output], dtype=np.float32),
            xyxy=np.asarray([[1, 1, 5, 6] for _ in self.output], dtype=np.float32).reshape(-1, 4),
        )
        return [SimpleNamespace(boxes=boxes, names={0: "chair", 1: "sofa"})]


def _extractor(tmp_path, model: FakeYOLO, *, every_n_steps: int = 5) -> YOLOSceneExtractor:
    weights = tmp_path / "yolo11n.pt"
    weights.write_bytes(b"fake local weights: model_factory prevents loading")
    return YOLOSceneExtractor(
        str(weights), device="cpu", confidence=0.25,
        every_n_steps=every_n_steps, model_factory=lambda path: model,
    )


def test_mock_detection_uses_rgb_and_preserves_per_object_confidence(tmp_path) -> None:
    model = FakeYOLO([(0, 0.91), (1, 0.72), (0, 0.10)])
    extractor = _extractor(tmp_path, model)
    extraction = extractor.extract(ObservationPacket(step_id=0, rgb=_rgb()))

    assert len(model.calls) == 1
    assert model.calls[0]["pixels"][0, 0].tolist() == [30, 0, 220]
    assert model.calls[0]["conf"] == 0.25
    assert model.calls[0]["device"] == "cpu"
    assert model.calls[0]["save"] is False
    assert extraction.place_label is None and extraction.landmarks == []
    assert extraction.objects == ["chair", "sofa"]
    assert [detection.confidence for detection in extraction.detections] == pytest.approx([0.91, 0.72])
    assert extraction.detections[0].bbox_xyxy == (1.0, 1.0, 5.0, 6.0)
    assert extraction.source == "ultralytics_yolo:yolo11n.pt"
    assert extractor.last_detection_ms >= 0.0


def test_detection_record_json_roundtrip() -> None:
    extraction = SceneExtraction(
        objects=["chair"], confidence=0.8, source="local_yolo",
        detections=[DetectedObject("chair", 0.8, (1.0, 2.0, 3.0, 4.0))],
    )
    assert SceneExtraction.from_dict(extraction.to_dict()) == extraction


def test_every_five_steps_and_episode_reset(tmp_path) -> None:
    model = FakeYOLO([(0, 0.9)])
    extractor = _extractor(tmp_path, model)
    extractor.reset_episode("first", "find chair")
    for step in range(11):
        result = extractor.extract(ObservationPacket(step_id=step, rgb=_rgb()))
        assert bool(result.detections) == (step in {0, 5, 10})
        if step in {0, 5, 10}:
            assert extractor.last_detection_ms >= 0.0
        else:
            assert extractor.last_detection_ms is None
    assert len(model.calls) == 3
    assert extractor.last_attempt_step == 10
    extractor.reset_episode("second", "find sofa")
    assert extractor.last_attempt_step is None and extractor.last_error is None
    assert extractor.last_detection_ms is None
    assert extractor.extract(ObservationPacket(step_id=0, rgb=_rgb())).objects == ["chair"]
    assert len(model.calls) == 4


def test_empty_detection_adds_no_semantic_memory(tmp_path) -> None:
    model = FakeYOLO([])
    manager = MemoryManager(extractor=_extractor(tmp_path, model))
    manager.reset_episode("empty", "find chair")
    assert manager.observe(ObservationPacket(step_id=0, rgb=_rgb(), instruction="find chair")) == []
    assert manager.episodic.events == []
    assert manager.scene_graph.nodes == [] and manager.scene_graph.edges == []
    assert manager.get_context().text == ""
    manager.close_episode()


def test_detected_objects_enter_events_graph_and_later_retrieval(tmp_path) -> None:
    model = FakeYOLO([(0, 0.91), (1, 0.72)])
    manager = MemoryManager(extractor=_extractor(tmp_path, model))
    manager.reset_episode("first", "find chair")
    first_events = manager.observe(ObservationPacket(step_id=0, rgb=_rgb(), instruction="find chair"))
    observations = [event for event in first_events if event.kind == "object_observation"]
    assert [event.kind for event in observations] == ["object_observation"] * 2
    assert [event.confidence for event in observations] == pytest.approx([0.91, 0.72])
    assert all(event.source == "ultralytics_yolo:yolo11n.pt" for event in first_events)
    assert len(manager.scene_graph.query_nodes(kind="object")) == 2
    assert manager.scene_graph.query_nodes(kind="place") == []
    assert manager.scene_graph.edges == []
    assert manager.observe(ObservationPacket(step_id=1, rgb=_rgb(), instruction="find chair")) == []
    context = manager.get_context()
    assert "chair" in context.text and "source=ultralytics_yolo:yolo11n.pt" in context.text
    assert any(event.step_id == 0 for event in context.events)
    manager.close_episode()
    manager.reset_episode("second", "find sofa")
    assert manager.episodic.events == [] and manager.scene_graph.nodes == []
    assert manager.get_context().text == ""
    manager.close_episode()


def test_inference_error_is_logged_and_does_not_invent_objects(tmp_path, caplog) -> None:
    model = FakeYOLO(fail=True)
    extractor = _extractor(tmp_path, model)
    manager = MemoryManager(extractor=extractor)
    manager.reset_episode("failed", "find chair")
    with caplog.at_level("WARNING"):
        assert manager.observe(ObservationPacket(step_id=0, rgb=_rgb(), instruction="find chair")) == []
    assert "synthetic inference failure" in extractor.last_error
    assert extractor.last_detection_ms >= 0.0
    assert "YOLO extraction failed" in caplog.text
    assert "step=0" in caplog.text and "yolo11n.pt" in caplog.text
    assert manager.scene_graph.nodes == [] and manager.episodic.events == []
    manager.close_episode()


def test_missing_or_unloadable_local_weights_fail_clearly(tmp_path) -> None:
    factory_calls = []

    def factory(path):
        factory_calls.append(path)
        return FakeYOLO()

    other_format = tmp_path / "model.onnx"
    other_format.write_bytes(b"fake")
    with pytest.raises(ValueError, match=r"local \.pt file"):
        YOLOSceneExtractor(str(other_format), model_factory=factory)
    with pytest.raises(FileNotFoundError, match="local YOLO model file not found"):
        YOLOSceneExtractor(str(tmp_path / "missing.pt"), model_factory=factory)
    assert factory_calls == []
    weights = tmp_path / "yolo11n.pt"
    weights.write_bytes(b"fake")

    def fail_load(path):
        raise RuntimeError("bad weights")

    with pytest.raises(RuntimeError, match="failed to load local YOLO model"):
        YOLOSceneExtractor(str(weights), model_factory=fail_load)
    wrong_task = FakeYOLO()
    wrong_task.task = "classify"
    with pytest.raises(RuntimeError, match="must have detection task"):
        YOLOSceneExtractor(str(weights), model_factory=lambda path: wrong_task)
