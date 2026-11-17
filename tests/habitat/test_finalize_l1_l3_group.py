"""Recover a completed shard set without rerunning inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lightnav.habitat.prompt_artifacts import build_prompt_protocol
from lightnav.habitat.runner import MemoryConfig
from lightnav.memory.local_graph_config import LocalGraphConfig
from lightnav.memory.prompt_modes import (
    PROMPT_PROTOCOL_VERSION,
    effective_layer_activation,
    mode_plan,
)
from lightnav.memory.spatial_l3_config import SpatialL3Config
from scripts.finalize_l1_l3_group import finalize_group
from scripts.verify_local_graph_shards import verify_local_graph_shards


def _memory(mode: str = "l1_organized_only") -> MemoryConfig:
    return MemoryConfig(
        enabled=True, extractor="yolo11", extractor_model_path="/data/yolo11n.pt",
        strategy="hierarchical", layers=("short", "mid", "long"),
        local_graph=LocalGraphConfig(enabled=True),
        spatial_l3=SpatialL3Config(mode="inject", seg_model_path="/data/seg.pt"),
        prompt_mode=mode,
    )


def _episode_rows(count: int = 4) -> list[dict[str, str]]:
    return [
        {"scene_id": f"/data/scene{index // 2:03d}", "episode_id": f"ep{index:02d}"}
        for index in range(count)
    ]


def _group(tmp_path: Path, *, old_conflict_shape: bool = False) -> Path:
    root = tmp_path / "l1_original_only"
    root.mkdir(parents=True)
    episodes = _episode_rows()
    manifest = {"split": "val_unseen", "expected_count": 4, "episodes": episodes}
    (root / "episode_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    canonical_digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    (root.parent / "provenance.json").write_text(json.dumps({
        "checkpoint_sha256": "e" * 64,
        "checkpoint_config_sha256": "c" * 64,
        "checkpoint_eval_config_sha256": "d" * 64,
        "habitat_config_sha256": "f" * 64,
        "episode_manifest_canonical_sha256": canonical_digest,
        "seed": 42,
        "gpu_ids": ["0", "1"],
    }), encoding="utf-8")
    effective = {
        "enabled": True, "scope": "episode_local", "prompt_max_chars": 320,
        "max_landmarks": 2, "reactivation_steps": 8,
        "protocol_version": "local_graph_v1",
    }
    prompt_effective = {
        "prompt_mode": "l1_organized_only",
        "protocol_version": PROMPT_PROTOCOL_VERSION,
        "budget_strategy": "token_aware_context",
        "original_char_budget_chars": 320,
        "layer_activation": effective_layer_activation(mode_plan("l1_organized_only")),
    }
    snapshot = {
        "protocol_version": (
            PROMPT_PROTOCOL_VERSION if old_conflict_shape else "local_graph_v1"
        ),
        "common": {
            "gpu_ids": ["0", "1"], "task": "vlnce", "split": "val_unseen",
            "expected_episodes": 4, "max_steps": 500,
            "force_stop_at_max_steps": False, "seed": 42,
            "model_weights_sha256": "e" * 64, "model_config_sha256": "c" * 64,
            "eval_config_sha256": "d" * 64, "habitat_config_sha256": "f" * 64,
        },
        "local_graph_effective": effective,
        "memory_effective": {
            "local_graph_effective": effective,
            "prompt_effective": prompt_effective,
        },
        "prompt_effective": prompt_effective,
    }
    if not old_conflict_shape:
        snapshot["prompt_protocol_version"] = PROMPT_PROTOCOL_VERSION
    (root / "config_snapshot.yaml").write_text(json.dumps(snapshot), encoding="utf-8")
    protocol = build_prompt_protocol(_memory())
    marker = {key: value for key, value in effective.items() if key != "enabled"}
    for shard_index in range(2):
        shard = root / f"shard_{shard_index}"
        shard.mkdir()
        shard_rows = episodes[shard_index * 2:(shard_index + 1) * 2]
        shard_manifest = {
            "split": "val_unseen", "expected_count": 2, "episodes": shard_rows,
        }
        (shard / "episode_manifest.json").write_text(
            json.dumps(shard_manifest), encoding="utf-8",
        )
        (shard / "local_graph_protocol.json").write_text(json.dumps(marker))
        (shard / "prompt_protocol.json").write_text(
            json.dumps(protocol), encoding="utf-8",
        )
        results = []
        memory_rows = []
        prompt_rows = []
        for index, row in enumerate(shard_rows):
            episode_id = f"episode_{shard_index * 2 + index:03d}"
            results.append({
                "episode_id": episode_id,
                "habitat_episode_id": row["episode_id"],
                "scene_id": row["scene_id"],
                "success": 1.0 if index % 2 == 0 else 0.0,
                "oracle_success": 1.0,
                "spl": 1.0 if index % 2 == 0 else 0.0,
                "ndtw": 0.8, "final_distance": 0.5, "min_distance": 0.4,
                "steps": 1, "path_length": 3.0, "forced_stop": False,
                "terminated": True, "truncated": False,
                "termination_reason": "agent_stop", "episode_time_sec": 1.0,
                "inference_latency_ms_mean": 10.0,
                "instruction": "find the chair",
            })
            memory_rows.append({
                "episode_key": {"scene_id": row["scene_id"],
                                "episode_id": row["episode_id"]},
                "scene_id": row["scene_id"],
                "habitat_episode_id": row["episode_id"],
                "step": 0, "prompt_chars": 500,
                "inference_latency_ms": 10.0, "memory_enabled": True,
            })
            prompt_rows.append({
                "episode_id": episode_id,
                "habitat_episode_id": row["episode_id"],
                "scene_id": row["scene_id"],
                "step": 0, "mode": "l1_organized_only",
                "prompt_chars": 500, "injected_memory_chars": 200,
                "prompt_tokens": 700, "budget_chars": None,
                "budget_source": "token_aware_context",
                "actual_injection": True,
                "l1_item_count": 1, "l1_fact_count": 1,
                "l3_candidate_count": 0, "l3_item_count": 0,
                "duplicates_removed": 0, "conflict_count": 0,
                "alignment_attempted": 0, "alignment_confirmed": 0,
                "omitted_items": 0, "processor_length_rejections": 0,
                "token_overflow_drops": 0,
            })
        (shard / "results.jsonl").write_text(
            "\n".join(json.dumps(item) for item in results) + "\n", encoding="utf-8",
        )
        (shard / "memory_diagnostics.jsonl").write_text(
            "\n".join(json.dumps(item) for item in memory_rows) + "\n",
            encoding="utf-8",
        )
        (shard / "prompt_diagnostics.jsonl").write_text(
            "\n".join(json.dumps(item) for item in prompt_rows) + "\n",
            encoding="utf-8",
        )
    return root


def test_finalize_complete_shards_without_inference(tmp_path: Path) -> None:
    group = _group(tmp_path)
    result = finalize_group(
        group, gpu_ids=["0", "1"], episode_manifest=group / "episode_manifest.json",
        expected_episodes=4,
    )
    assert result["status"] == "complete"
    assert (group / "formal_complete.json").is_file()
    assert (group / "postprocess_provenance.json").is_file()
    assert len((group / "results.jsonl").read_text().splitlines()) == 4
    assert (group / "summary.json").is_file()
    assert len((group / "memory_diagnostics.jsonl").read_text().splitlines()) == 4
    assert len((group / "prompt_diagnostics.jsonl").read_text().splitlines()) == 4
    assert (group / "per_episode.csv").is_file()
    assert not (group / "shard_0" / "results.jsonl").read_text() == ""


def test_finalize_accepts_the_old_protocol_conflict_shape(tmp_path: Path) -> None:
    group = _group(tmp_path, old_conflict_shape=True)
    result = finalize_group(
        group, gpu_ids=["0", "1"], episode_manifest=group / "episode_manifest.json",
        expected_episodes=4,
    )
    assert result["status"] == "complete"
    assert (group / "formal_complete.json").is_file()


def test_finalize_refuses_incomplete_shards_without_writing_marker(
    tmp_path: Path,
) -> None:
    group = _group(tmp_path)
    (group / "shard_1" / "results.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="shard results are missing"):
        finalize_group(
            group, gpu_ids=["0", "1"],
            episode_manifest=group / "episode_manifest.json", expected_episodes=4,
        )
    assert not (group / "formal_complete.json").exists()
    assert not (group / "results.jsonl").exists()


def test_finalize_rejects_duplicate_episode_across_shards(tmp_path: Path) -> None:
    group = _group(tmp_path)
    rows = (group / "shard_1" / "results.jsonl").read_text().splitlines()
    duplicated = json.loads(rows[0])
    duplicated["habitat_episode_id"] = "ep00"
    duplicated["scene_id"] = "/data/scene000"
    (group / "shard_1" / "results.jsonl").write_text(
        "\n".join([json.dumps(duplicated), *rows]) + "\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate Episode across shards"):
        finalize_group(
            group, gpu_ids=["0", "1"],
            episode_manifest=group / "episode_manifest.json", expected_episodes=4,
        )
    assert not (group / "formal_complete.json").exists()


def test_verifier_passes_new_and_old_prompt_snapshots(tmp_path: Path) -> None:
    new_group = _group(tmp_path / "new")
    verify_local_graph_shards(new_group, ["0", "1"])
    old_group = _group(tmp_path / "old", old_conflict_shape=True)
    verify_local_graph_shards(old_group, ["0", "1"])


def test_finalize_rejects_incomplete_prompt_diagnostics(tmp_path: Path) -> None:
    group = _group(tmp_path)
    (group / "shard_1" / "prompt_diagnostics.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="prompt diagnostics incomplete"):
        finalize_group(
            group, gpu_ids=["0", "1"],
            episode_manifest=group / "episode_manifest.json", expected_episodes=4,
        )
    assert not (group / "formal_complete.json").exists()


def test_finalize_rejects_provenance_mismatch(tmp_path: Path) -> None:
    group = _group(tmp_path)
    provenance = json.loads((group.parent / "provenance.json").read_text())
    provenance["checkpoint_sha256"] = "0" * 64
    (group.parent / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="differs from run provenance"):
        finalize_group(
            group, gpu_ids=["0", "1"],
            episode_manifest=group / "episode_manifest.json", expected_episodes=4,
        )
    assert not (group / "formal_complete.json").exists()
