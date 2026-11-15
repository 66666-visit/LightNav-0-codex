"""Independent YOLO segmentation parsing with local fake weights and no GPU."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lightnav.memory.yolo_seg_extractor import YOLOSegExtractor


class FakeSegmentModel:
    task = "segment"

    def __init__(self, *, mask: np.ndarray | None = None, with_masks: bool = True) -> None:
        self.calls: list[dict] = []
        self.mask = mask
        self.with_masks = with_masks

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        boxes = SimpleNamespace(
            cls=np.array([0.0], dtype=np.float32),
            conf=np.array([0.9], dtype=np.float32),
            xyxy=np.array([[1, 1, 5, 5]], dtype=np.float32),
        )
        masks = None if not self.with_masks else SimpleNamespace(data=self.mask)
        return [SimpleNamespace(boxes=boxes, masks=masks, names={0: "chair"})]


def _extractor(tmp_path, model: FakeSegmentModel) -> YOLOSegExtractor:
    path = tmp_path / "yolo11n-seg.pt"
    path.write_bytes(b"fake local weights")
    return YOLOSegExtractor(str(path), every_n_steps=2, model_factory=lambda _: model)


def test_yolo_parses_native_instance_mask_without_using_box_as_mask(tmp_path) -> None:
    mask = np.zeros((6, 6), dtype=np.float32)
    mask[1:5, 2:4] = 1
    model = FakeSegmentModel(mask=mask[None, ...])
    extractor = _extractor(tmp_path, model)
    rgb = np.zeros((6, 6, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 2] = 220, 30

    instances = extractor.extract(rgb, 0)

    assert len(instances) == 1
    assert model.calls[0]["source"][0, 0].tolist() == [30, 0, 220]
    assert model.calls[0]["retina_masks"] is True
    assert instances[0].class_name == "chair"
    assert instances[0].confidence == pytest.approx(0.9)
    assert instances[0].mask.sum() == 8
    assert not instances[0].mask[1, 1]  # Inside bbox, outside the real mask.
    assert extractor.extract(rgb, 1) == []
    assert len(model.calls) == 1


@pytest.mark.parametrize("case", ["missing", "wrong_size"])
def test_yolo_rejects_box_only_or_unaligned_masks(tmp_path, case: str) -> None:
    mask = np.ones((1, 3, 3), dtype=np.float32)
    model = FakeSegmentModel(mask=mask, with_masks=(case != "missing"))
    extractor = _extractor(tmp_path, model)

    assert extractor.extract(np.zeros((6, 6, 3), dtype=np.uint8), 0) == []
    assert extractor.last_error is not None


def test_yolo_requires_existing_local_segmentation_weights(tmp_path) -> None:
    model = FakeSegmentModel()
    with pytest.raises(FileNotFoundError):
        YOLOSegExtractor(str(tmp_path / "missing.pt"), model_factory=lambda _: model)
    path = tmp_path / "segment.pt"
    path.write_bytes(b"fake local weights")
    model.task = "detect"
    with pytest.raises(RuntimeError, match="segment task"):
        YOLOSegExtractor(str(path), model_factory=lambda _: model)
