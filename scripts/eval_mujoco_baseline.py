#!/usr/bin/env python3
"""Run repeatable MuJoCo VLN baseline trials without changing inference code.

The evaluator owns one demo WebSocket per trial. A local WebSocket relay observes
the existing LightNav protocol and forwards messages unchanged, allowing exact
request counts and server-reported history lengths to be recorded.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import socket
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit


SCHEMA_VERSION = 1
BUDGET_FIELDS = (
    "selected_frame_count",
    "vit_patch_rows",
    "vit_output_visual_tokens_before_pool",
    "llm_visual_tokens_after_pool",
)
UNSCORABLE_REASONS = {
    "history_discontinuity",
    "inference_error",
    "lost_control",
    "model_stop_unconfirmed",
    "mpc_error",
    "proxy_error",
    "reset_error",
    "runtime_error",
}


def finite_number(value: Any, name: str) -> float:
    """Reject booleans, infinities, and malformed numeric task fields."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def xy_pair(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be [x, y]")
    return finite_number(value[0], f"{name}[0]"), finite_number(value[1], f"{name}[1]")


def validate_manifest(value: Any) -> dict[str, Any]:
    """Validate a fixed, explicitly labelled set of navigation tasks."""
    if not isinstance(value, dict):
        raise ValueError("task manifest must be a JSON object")
    scene_id = value.get("scene_id")
    robot = value.get("robot")
    if not isinstance(scene_id, str) or not scene_id.strip():
        raise ValueError("scene_id must be a nonempty string")
    if not isinstance(robot, str) or not robot.strip():
        raise ValueError("robot must be a nonempty string")
    spawn_xy = xy_pair(value.get("expected_spawn_xy"), "expected_spawn_xy")
    spawn_yaw = finite_number(value.get("expected_spawn_yaw"), "expected_spawn_yaw")
    raw_tasks = value.get("tasks")
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise ValueError("tasks must be a nonempty list")
    tasks = []
    task_ids = set()
    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, dict):
            raise ValueError(f"tasks[{index}] must be an object")
        task_id = raw.get("id")
        instruction = raw.get("instruction")
        verified = raw.get("goal_verified")
        if not isinstance(task_id, str) or not task_id.strip() or task_id in task_ids:
            raise ValueError(f"tasks[{index}].id must be unique and nonempty")
        if not isinstance(instruction, str) or not instruction.strip() or len(instruction) > 500:
            raise ValueError(f"tasks[{index}].instruction must have 1 to 500 characters")
        if not isinstance(verified, bool):
            raise ValueError(f"tasks[{index}].goal_verified must be boolean")
        goal_xy = xy_pair(raw.get("goal_xy"), f"tasks[{index}].goal_xy")
        radius = finite_number(raw.get("radius_m"), f"tasks[{index}].radius_m")
        if radius <= 0:
            raise ValueError(f"tasks[{index}].radius_m must be positive")
        task_ids.add(task_id)
        tasks.append({
            "id": task_id,
            "instruction": instruction.strip(),
            "goal_xy": list(goal_xy),
            "radius_m": radius,
            "goal_verified": verified,
        })
    return {
        "scene_id": scene_id,
        "robot": robot,
        "expected_spawn_xy": list(spawn_xy),
        "expected_spawn_yaw": spawn_yaw,
        "tasks": tasks,
    }


def pose_xy(snapshot: dict[str, Any]) -> tuple[float, float]:
    pose = snapshot["simulation"]["pose"]
    return finite_number(pose["x"], "pose.x"), finite_number(pose["y"], "pose.y")


def goal_distance(snapshot: dict[str, Any], goal_xy: list[float]) -> float:
    x, y = pose_xy(snapshot)
    return math.hypot(x - goal_xy[0], y - goal_xy[1])


def yaw_error(actual: float, expected: float) -> float:
    """Shortest absolute angle between two yaw values, in radians."""
    delta = actual - expected
    return abs(math.atan2(math.sin(delta), math.cos(delta)))


def score_trial(
    goal_verified: bool,
    stop_reason: str,
    distance_m: float | None,
    radius_m: float,
    trajectory_valid: bool | None,
    trajectory_reason: str | None = None,
) -> tuple[bool | None, str | None]:
    """Return strict success and an explicit reason when scoring is unavailable."""
    if not goal_verified:
        return None, "goal_unverified"
    if stop_reason in UNSCORABLE_REASONS:
        return None, stop_reason
    if distance_m is None:
        return None, "final_pose_unavailable"
    if stop_reason in {"timeout", "max_steps"}:
        return False, None
    if stop_reason != "model_stop":
        return None, "unknown_stop_reason"
    if distance_m > radius_m:
        return False, None
    if trajectory_valid is None:
        return None, trajectory_reason or "trajectory_unavailable"
    return trajectory_valid, None


def history_is_continuous(steps: list[dict[str, Any]]) -> bool:
    """The service's actions.step must count 1, 2, ... within one trial."""
    return all(step.get("history_frames") == index for index, step in enumerate(steps, start=1))


def _mean(values: list[float | int]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    verified = [record for record in records if record.get("goal_verified") is True]
    scored = [record for record in verified if record.get("success") is not None]
    completed = [record for record in records if record.get("stop_reason") in {"model_stop", "timeout", "max_steps"}]
    steps = [step for record in completed for step in record.get("steps", []) if step.get("rc") == 0]
    totals = [record.get("visual_token_totals") for record in completed]
    totals = [total for total in totals if isinstance(total, dict)]
    stop_reasons = Counter(record.get("stop_reason") or "unknown" for record in records)
    unscorable_reasons = Counter(
        record["unscorable_reason"] for record in records if record.get("unscorable_reason")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "total_trials": len(records),
        "verified_trials": len(verified),
        "scored_trials": len(scored),
        "unscorable_verified_trials": len(verified) - len(scored),
        "scoring_coverage": len(scored) / len(verified) if verified else None,
        "abnormal_trials": sum(record.get("stop_reason") in UNSCORABLE_REASONS for record in records),
        "invalid_trajectory_trials": sum(record.get("trajectory_valid") is False for record in records),
        "unknown_trajectory_trials": sum(record.get("trajectory_valid") is None for record in records),
        "stop_reason_counts": dict(stop_reasons),
        "unscorable_reason_counts": dict(unscorable_reasons),
        "successful_trials": sum(record["success"] is True for record in scored),
        "success_rate": (
            sum(record["success"] is True for record in verified) / len(verified)
            if verified else None
        ),
        "valid_scored_success_rate": _mean([int(record["success"]) for record in scored]),
        "mean_navigation_steps": _mean([record["navigation_steps"] for record in completed]),
        "mean_duration_s": _mean([record["duration_s"] for record in completed]),
        "mean_server_latency_ms_per_step": _mean([
            step["server_latency_ms"] for step in steps if step.get("server_latency_ms") is not None
        ]),
        "mean_proxy_roundtrip_ms_per_step": _mean([
            step["proxy_roundtrip_ms"] for step in steps if step.get("proxy_roundtrip_ms") is not None
        ]),
        "mean_selected_frames_per_step": _mean([
            step["budget"]["selected_frame_count"] for step in steps if isinstance(step.get("budget"), dict)
        ]),
        "mean_llm_visual_tokens_per_step": _mean([
            step["budget"]["llm_visual_tokens_after_pool"] for step in steps
            if isinstance(step.get("budget"), dict)
        ]),
        "mean_vit_patch_rows_per_step": _mean([
            step["budget"]["vit_patch_rows"] for step in steps if isinstance(step.get("budget"), dict)
        ]),
        "mean_vit_output_visual_tokens_per_step": _mean([
            step["budget"]["vit_output_visual_tokens_before_pool"] for step in steps
            if isinstance(step.get("budget"), dict)
        ]),
        "mean_total_llm_visual_tokens_per_trial": _mean([
            total["llm_visual_tokens_after_pool"] for total in totals
        ]),
        "token_count_source": "derived_from_checkpoint_diagnostic; not live processor telemetry",
    }


def normalize_demo_url(value: str) -> str:
    parsed = urlsplit(value)
    scheme = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}.get(parsed.scheme)
    if scheme is None or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("--demo-url must be an HTTP or WebSocket URL")
    path = "/ws" if parsed.path in {"", "/"} else parsed.path
    return urlunsplit((scheme, parsed.netloc, path, parsed.query, ""))


def validate_model_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"ws", "wss"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("--vln-server must be a ws:// or wss:// URL")
    return value


def _json_object(raw: str | bytes) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("WebSocket payload must be a JSON object")
    return value


def _positive_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


@dataclass
class ModelProxy:
    """Per-trial 1:1 relay. Its step log never crosses trial connections."""

    target_url: str

    def __post_init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.error: str | None = None
        self.url: str | None = None
        self._runner = None
        self._downstreams: set[Any] = set()
        self._handlers: set[asyncio.Task[Any]] = set()

    async def start(self) -> str:
        from aiohttp import web

        app = web.Application()
        app.router.add_get("/ws", self._handle)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            sock.listen(16)
            sock.setblocking(False)
            port = sock.getsockname()[1]
            site = web.SockSite(runner, sock)
            await site.start()
        except Exception:
            sock.close()
            await runner.cleanup()
            raise
        self._runner = runner
        self.url = f"ws://127.0.0.1:{port}/ws"
        return self.url

    async def close(self) -> None:
        for downstream in tuple(self._downstreams):
            await downstream.close()
        for handler in tuple(self._handlers):
            handler.cancel()
        if self._handlers:
            await asyncio.gather(*self._handlers, return_exceptions=True)
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _handle(self, request: Any) -> Any:
        from aiohttp import ClientSession, ClientTimeout, WSMsgType, web

        downstream = web.WebSocketResponse(heartbeat=10.0, max_msg_size=0)
        await downstream.prepare(request)
        self._downstreams.add(downstream)
        handler = asyncio.current_task()
        if handler is not None:
            self._handlers.add(handler)
        try:
            timeout = ClientTimeout(total=None, sock_connect=5.0)
            async with ClientSession(timeout=timeout) as client:
                async with client.ws_connect(self.target_url, heartbeat=10.0, max_msg_size=0) as upstream:
                    async for message in downstream:
                        if message.type not in {WSMsgType.TEXT, WSMsgType.BINARY}:
                            continue
                        raw = message.data
                        request_payload = _json_object(raw)
                        sent_at = time.monotonic()
                        if message.type == WSMsgType.TEXT:
                            await upstream.send_str(raw)
                        else:
                            await upstream.send_bytes(raw)
                        response = await upstream.receive()
                        received_at = time.monotonic()
                        if request_payload.get("action") == "next" and response.type in {
                            WSMsgType.TEXT, WSMsgType.BINARY,
                        }:
                            self._record_next(
                                request_payload,
                                _json_object(response.data),
                                (received_at - sent_at) * 1000.0,
                            )
                        if response.type == WSMsgType.TEXT:
                            await downstream.send_str(response.data)
                        elif response.type == WSMsgType.BINARY:
                            await downstream.send_bytes(response.data)
                        else:
                            raise RuntimeError("model WebSocket closed before its response")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._downstreams.discard(downstream)
            if handler is not None:
                self._handlers.discard(handler)
            await downstream.close()
        return downstream

    def _record_next(self, request: dict[str, Any], response: dict[str, Any], roundtrip_ms: float) -> None:
        request_data = request.get("data")
        response_data = response.get("data")
        if not isinstance(request_data, dict) or not isinstance(response_data, dict):
            self.error = "malformed next request or response"
            return
        if (
            response.get("action") != "next"
            or type(response_data.get("rc")) is not int
            or response_data.get("seq") != request_data.get("seq")
        ):
            self.error = "next response action, rc, or sequence mismatch"
        actions = response_data.get("actions")
        history_frames = _positive_int(actions.get("step")) if isinstance(actions, dict) else None
        latency = response_data.get("latency_ms")
        try:
            server_latency = finite_number(latency, "latency_ms") if latency is not None else None
        except ValueError:
            server_latency = None
        self.steps.append({
            "seq": request_data.get("seq"),
            "history_frames": history_frames,
            "rc": response_data.get("rc"),
            "stop": response_data.get("stop") if isinstance(response_data.get("stop"), bool) else None,
            "server_latency_ms": server_latency,
            "proxy_roundtrip_ms": roundtrip_ms,
        })


class DemoClient:
    """Serial command client; demo command results have no request IDs."""

    def __init__(self, websocket: Any) -> None:
        self.websocket = websocket
        self.snapshot: dict[str, Any] | None = None
        self.path_length_m = 0.0
        self._last_path_pose: tuple[float, float] | None = None
        self._goal_xy: list[float] | None = None
        self.min_goal_distance_m_sampled: float | None = None
        self.trajectory_samples: list[dict[str, float]] = []

    async def read(self, timeout_s: float) -> dict[str, Any]:
        message = await asyncio.wait_for(self.websocket.receive(), timeout=timeout_s)
        if message.type.name != "TEXT":
            raise RuntimeError("demo WebSocket closed or returned a non-JSON message")
        payload = _json_object(message.data)
        if payload.get("type") in {"snapshot", "runtime"}:
            data = payload.get("data")
            if not isinstance(data, dict):
                raise RuntimeError("demo runtime snapshot is malformed")
            self.snapshot = data
            if self._last_path_pose is not None:
                next_pose = pose_xy(data)
                self.path_length_m += math.dist(self._last_path_pose, next_pose)
                self._last_path_pose = next_pose
                sample = _trajectory_sample(data)
                if not self.trajectory_samples or sample["sim_time"] != self.trajectory_samples[-1]["sim_time"]:
                    self.trajectory_samples.append(sample)
                if self._goal_xy is not None:
                    distance = goal_distance(data, self._goal_xy)
                    previous = self.min_goal_distance_m_sampled
                    self.min_goal_distance_m_sampled = min(previous, distance) if previous is not None else distance
        return payload

    async def command(self, payload: dict[str, Any], timeout_s: float = 5.0) -> None:
        await self.websocket.send_json(payload)
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"demo command {payload['type']} timed out")
            reply = await self.read(remaining)
            if reply.get("type") == "command_result":
                if reply.get("ok") is not True:
                    raise RuntimeError(f"demo command {payload['type']} failed: {reply.get('message')}")
                return

    async def wait_snapshot(self, predicate: Any, timeout_s: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while True:
            if self.snapshot is not None and predicate(self.snapshot):
                return self.snapshot
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for a matching demo snapshot")
            await self.read(remaining)

    def start_path(self, goal_xy: list[float]) -> None:
        if self.snapshot is None:
            raise RuntimeError("cannot track a path without a snapshot")
        self._last_path_pose = pose_xy(self.snapshot)
        self._goal_xy = goal_xy
        self.min_goal_distance_m_sampled = goal_distance(self.snapshot, goal_xy)
        self.trajectory_samples = [_trajectory_sample(self.snapshot)]

    def stop_path(self) -> None:
        self._last_path_pose = None
        self._goal_xy = None


def _camera_stamp(snapshot: dict[str, Any]) -> int:
    stamp = snapshot["simulation"]["camera"].get("stamp_ns")
    return stamp if isinstance(stamp, int) and not isinstance(stamp, bool) else 0


def _snapshot_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    simulation = snapshot["simulation"]
    return {
        "pose": simulation["pose"],
        "sim_time": simulation["sim_time"],
        "camera_stamp_ns": _camera_stamp(snapshot),
    }


def _trajectory_sample(snapshot: dict[str, Any]) -> dict[str, float]:
    simulation = snapshot["simulation"]
    pose = simulation["pose"]
    return {
        "sim_time": finite_number(simulation["sim_time"], "sim_time"),
        "x": finite_number(pose["x"], "pose.x"),
        "y": finite_number(pose["y"], "pose.y"),
        "z": finite_number(pose["z"], "pose.z"),
        "yaw": finite_number(pose["yaw"], "pose.yaw"),
    }


async def await_post_reset_frame(
    demo: DemoClient,
    ack_ns: int,
    spawn_xy: tuple[float, float],
    spawn_yaw: float,
    timeout_s: float = 10.0,
) -> tuple[dict[str, Any], list[int]]:
    """Require two post-ack frames because one pre-reset render can finish late.

    Simulation renders under a lock but stamps the encoded frame after releasing
    it. A pre-reset render can therefore receive a timestamp after the reset
    command was acknowledged. The next distinct post-ack frame is captured after
    that in-flight render and cannot precede the reset.
    """
    seen: list[int] = []

    def ready(snapshot: dict[str, Any]) -> bool:
        stamp = _camera_stamp(snapshot)
        if stamp <= ack_ns or (seen and stamp <= seen[-1]):
            return False
        seen.append(stamp)
        if len(seen) < 2:
            return False
        if math.dist(pose_xy(snapshot), spawn_xy) > 0.05:
            raise RuntimeError("reset pose differs from expected spawn position")
        yaw = finite_number(snapshot["simulation"]["pose"]["yaw"], "pose.yaw")
        if yaw_error(yaw, spawn_yaw) > 0.05:
            raise RuntimeError("reset pose differs from expected spawn yaw")
        return True

    snapshot = await demo.wait_snapshot(ready, timeout_s)
    return snapshot, seen


async def _run_navigation(
    demo: DemoClient,
    proxy: ModelProxy,
    timeout_s: float,
    max_steps: int,
    *,
    started_at: float | None = None,
) -> str:
    deadline = (time.monotonic() if started_at is None else started_at) + timeout_s
    stop_seen_at: float | None = None
    while True:
        snapshot = demo.snapshot
        if snapshot is None:
            return "runtime_error"
        if proxy.error:
            return "proxy_error"
        if any(step.get("rc") != 0 for step in proxy.steps):
            return "inference_error"
        if not history_is_continuous(proxy.steps):
            return "history_discontinuity"
        if snapshot["mpc"].get("error"):
            return "mpc_error"
        if snapshot["vln"].get("state") == "ERROR":
            return "inference_error"

        model_stop = bool(proxy.steps and proxy.steps[-1]["stop"] is True)
        if model_stop:
            if snapshot["vln"].get("stop") is True and snapshot["control"].get("auto") is False:
                return "model_stop"
            if stop_seen_at is None:
                stop_seen_at = time.monotonic()
            if time.monotonic() - stop_seen_at > 5.0 or time.monotonic() >= deadline:
                return "model_stop_unconfirmed"
        elif snapshot["control"].get("auto") is False:
            return "lost_control"
        elif len(proxy.steps) >= max_steps:
            return "max_steps"
        elif time.monotonic() >= deadline:
            return "inference_error" if not proxy.steps and not snapshot["vln"].get("connected") else "timeout"

        try:
            await demo.read(0.15)
        except asyncio.TimeoutError:
            pass


async def run_trial(
    *,
    demo_url: str,
    model_url: str,
    manifest: dict[str, Any],
    task: dict[str, Any],
    repeat_index: int,
    model_id: str,
    budget: Any,
    budget_sha256: str,
    manifest_sha256: str,
    trajectory_probe: Any,
    geometry_error: str | None,
    timeout_s: float,
    max_steps: int,
) -> dict[str, Any]:
    from aiohttp import ClientSession, ClientTimeout

    proxy = ModelProxy(model_url)
    started_at = None
    ended_at = None
    demo: DemoClient | None = None
    stop_reason = "runtime_error"
    error: str | None = None
    initial_state = None
    final_state = None
    final_distance = None
    cleanup_error = None
    try:
        await proxy.start()
        client_timeout = ClientTimeout(total=None, sock_connect=5.0)
        async with ClientSession(timeout=client_timeout) as session:
            async with session.ws_connect(demo_url, heartbeat=20.0, max_msg_size=0) as websocket:
                demo = DemoClient(websocket)
                initial_snapshot = await demo.wait_snapshot(lambda _: True, 5.0)
                original_server_url = initial_snapshot["vln"].get("server_url", "")
                url_changed = False
                try:
                    if initial_snapshot["scene"].get("id") != manifest["scene_id"]:
                        raise RuntimeError("demo scene does not match task manifest")
                    if initial_snapshot["robot"].get("name") != manifest["robot"]:
                        raise RuntimeError("demo robot does not match task manifest")
                    if initial_snapshot["control"].get("locked"):
                        raise RuntimeError("another client owns demo control")
                    await demo.command({"type": "reset"})
                    reset_ack_time_ns = time.time_ns()
                    spawn = tuple(manifest["expected_spawn_xy"])
                    reset_snapshot, post_ack_stamps = await await_post_reset_frame(
                        demo, reset_ack_time_ns, spawn, manifest["expected_spawn_yaw"]
                    )
                    initial_state = _snapshot_state(reset_snapshot)
                    initial_state["reset_ack_time_ns"] = reset_ack_time_ns
                    initial_state["post_ack_frame_stamps_ns"] = post_ack_stamps
                    await demo.command({"type": "set_server_url", "server_url": proxy.url})
                    url_changed = True
                    await demo.wait_snapshot(lambda data: data["vln"].get("server_url") == proxy.url, 5.0)
                    demo.start_path(task["goal_xy"])
                    started_at = time.monotonic()
                    await demo.command({"type": "set_vln", "enabled": True, "instruction": task["instruction"]})
                    await demo.wait_snapshot(
                        lambda data: data["control"].get("auto") is True
                        or (data["vln"].get("stop") is True and data["control"].get("auto") is False),
                        5.0,
                    )
                    stop_reason = await _run_navigation(
                        demo, proxy, timeout_s, max_steps, started_at=started_at
                    )
                    if stop_reason != "model_stop":
                        await demo.command({"type": "set_vln", "enabled": False})
                        await demo.wait_snapshot(lambda data: data["control"].get("auto") is False, 5.0)
                    if demo.snapshot is not None:
                        final_state = _snapshot_state(demo.snapshot)
                        final_distance = goal_distance(demo.snapshot, task["goal_xy"])
                    demo.stop_path()
                    ended_at = time.monotonic()
                finally:
                    if url_changed:
                        try:
                            if demo.snapshot is not None and demo.snapshot["control"].get("auto") is True:
                                await demo.command({"type": "set_vln", "enabled": False})
                            await demo.command({"type": "set_server_url", "server_url": original_server_url})
                            await demo.wait_snapshot(
                                lambda data: data["vln"].get("server_url") == original_server_url,
                                5.0,
                            )
                        except Exception as exc:
                            cleanup_error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        if started_at is not None and ended_at is None:
            ended_at = time.monotonic()
        error = f"{type(exc).__name__}: {exc}"
        if initial_state is None:
            stop_reason = "reset_error"
        elif stop_reason not in UNSCORABLE_REASONS:
            stop_reason = "runtime_error"
        if demo is not None and demo.snapshot is not None:
            final_state = _snapshot_state(demo.snapshot)
            final_distance = goal_distance(demo.snapshot, task["goal_xy"])
    finally:
        await proxy.close()

    steps = proxy.steps
    if stop_reason not in UNSCORABLE_REASONS and not history_is_continuous(steps):
        stop_reason = "history_discontinuity"
    budget_error = None
    for step in steps:
        history_frames = step.get("history_frames")
        if step.get("rc") == 0 and history_frames is not None:
            try:
                estimate = budget.estimate(history_frames)
                if not isinstance(estimate, dict) or any(
                    not isinstance(estimate.get(field), int) or isinstance(estimate.get(field), bool)
                    for field in BUDGET_FIELDS
                ):
                    raise ValueError("budget estimate is missing an integer token field")
                step["budget"] = {field: estimate[field] for field in BUDGET_FIELDS}
            except Exception as exc:
                budget_error = f"{type(exc).__name__}: {exc}"
                step["budget"] = None
        else:
            step["budget"] = None

    valid_budgets = [step["budget"] for step in steps if isinstance(step["budget"], dict)]
    visual_totals = None
    if (
        stop_reason in {"model_stop", "timeout", "max_steps"}
        and history_is_continuous(steps)
        and budget_error is None
        and len(valid_budgets) == len(steps)
    ):
        visual_totals = {field: sum(item[field] for item in valid_budgets) for field in BUDGET_FIELDS}
    duration = ended_at - started_at if started_at is not None and ended_at is not None else None
    from mujoco_baseline_trajectory import check_trajectory

    trajectory_samples = demo.trajectory_samples if demo is not None else []
    trajectory_check = check_trajectory(trajectory_samples, trajectory_probe)
    success, unscorable_reason = score_trial(
        task["goal_verified"], stop_reason, final_distance, task["radius_m"],
        trajectory_check.valid, trajectory_check.reason,
    )
    completed_steps = [step for step in steps if step.get("rc") == 0]
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": task["id"],
        "repeat_index": repeat_index,
        "instruction": task["instruction"],
        "scene_id": manifest["scene_id"],
        "robot": manifest["robot"],
        "model_id": model_id,
        "model_server": model_url,
        "budget_diagnostic_sha256": budget_sha256,
        "task_manifest_sha256": manifest_sha256,
        "budget_source": "derived_from_checkpoint_diagnostic; not live processor telemetry",
        "goal_xy": task["goal_xy"],
        "radius_m": task["radius_m"],
        "goal_verified": task["goal_verified"],
        "initial_state": initial_state,
        "final_state": final_state,
        "final_goal_distance_m": final_distance,
        "min_goal_distance_m_sampled": demo.min_goal_distance_m_sampled if demo is not None else None,
        "goal_reached_in_sampled_snapshots": (
            demo.min_goal_distance_m_sampled <= task["radius_m"]
            if demo is not None and demo.min_goal_distance_m_sampled is not None else None
        ),
        "duration_s": duration,
        "success": success,
        "unscorable_reason": unscorable_reason,
        "stop_reason": stop_reason,
        "model_stop_received": any(step.get("stop") is True for step in steps),
        "error": error or proxy.error or budget_error,
        "budget_error": budget_error,
        "cleanup_error": cleanup_error,
        "navigation_steps": len(completed_steps),
        "next_requests": len(steps),
        "history_frame_counts": [step["history_frames"] for step in steps],
        "final_history_frames": steps[-1]["history_frames"] if steps else None,
        "mean_server_latency_ms": _mean([
            step["server_latency_ms"] for step in completed_steps
            if step.get("server_latency_ms") is not None
        ]),
        "mean_proxy_roundtrip_ms": _mean([
            step["proxy_roundtrip_ms"] for step in completed_steps
            if step.get("proxy_roundtrip_ms") is not None
        ]),
        "mean_llm_visual_tokens_per_step": _mean([
            step["budget"]["llm_visual_tokens_after_pool"] for step in completed_steps
            if isinstance(step.get("budget"), dict)
        ]),
        "trajectory_length_m_sampled": demo.path_length_m if demo is not None and started_at is not None else None,
        "trajectory_valid": trajectory_check.valid,
        "trajectory_check": trajectory_check.as_dict(),
        "trajectory_check_method": "sampled_mujoco_robot_collision_contacts_v1",
        "trajectory_geometry_error": geometry_error,
        "trajectory_samples": trajectory_samples,
        "visual_token_totals": visual_totals,
        "steps": steps,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-url", default="ws://127.0.0.1:8088/ws")
    parser.add_argument("--vln-server", required=True, help="Actual LightNav model WebSocket URL")
    parser.add_argument("--tasks", type=Path, required=True, help="Fixed task manifest JSON")
    parser.add_argument("--budget-diagnostic", type=Path, required=True)
    parser.add_argument("--model-id", required=True, help="Checkpoint/version label recorded with every trial")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True, help="New directory for results.jsonl and summary.json; never overwritten")
    return parser.parse_args(argv)


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from mujoco_baseline_budget import load_budget
    from mujoco_baseline_trajectory import MujocoFootprintProbe

    manifest_bytes = args.tasks.read_bytes()
    manifest = validate_manifest(json.loads(manifest_bytes))
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    budget = load_budget(args.budget_diagnostic)
    budget_sha256 = budget.diagnostic_sha256
    demo_url = normalize_demo_url(args.demo_url)
    model_url = validate_model_url(args.vln_server)
    if not args.model_id.strip():
        raise ValueError("--model-id must be nonempty")
    if args.repeat < 1 or args.max_steps < 1 or not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        raise ValueError("--repeat, --max-steps, and --timeout-s must be positive")

    trajectory_probe = None
    geometry_error = None
    if manifest["robot"] == "TurtleBot":
        try:
            trajectory_probe = MujocoFootprintProbe()
        except Exception as exc:
            geometry_error = f"{type(exc).__name__}: {exc}"
    else:
        geometry_error = f"no footprint collision checker for robot {manifest['robot']}"

    output_dir = args.output
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    output = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    records = []
    # Reserve both paths before any simulation command. Existing runs are never overwritten.
    with output.open("x", encoding="utf-8") as jsonl, summary_path.open("x", encoding="utf-8") as summary_file:
        for task in manifest["tasks"]:
            for repeat_index in range(1, args.repeat + 1):
                record = await run_trial(
                    demo_url=demo_url,
                    model_url=model_url,
                    manifest=manifest,
                    task=task,
                    repeat_index=repeat_index,
                    model_id=args.model_id,
                    budget=budget,
                    budget_sha256=budget_sha256,
                    manifest_sha256=manifest_sha256,
                    trajectory_probe=trajectory_probe,
                    geometry_error=geometry_error,
                    timeout_s=args.timeout_s,
                    max_steps=args.max_steps,
                )
                jsonl.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                jsonl.flush()
                records.append(record)
                print(
                    f"{task['id']} repeat={repeat_index} stop={record['stop_reason']} "
                    f"success={record['success']} steps={record['navigation_steps']}",
                    flush=True,
                )
        result = summarize(records)
        result.update({
            "model_id": args.model_id,
            "demo_url": demo_url,
            "model_server": model_url,
            "task_manifest": str(args.tasks),
            "task_manifest_sha256": manifest_sha256,
            "task_manifest_content": manifest,
            "budget_diagnostic": str(args.budget_diagnostic),
            "budget_diagnostic_sha256": budget_sha256,
            "trajectory_geometry_available": trajectory_probe is not None,
            "trajectory_geometry_error": geometry_error,
            "repeat": args.repeat,
            "timeout_s": args.timeout_s,
            "max_steps": args.max_steps,
            "jsonl": str(output),
        })
        summary_file.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        summary_file.flush()
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = asyncio.run(evaluate(args))
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"MuJoCo baseline evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
