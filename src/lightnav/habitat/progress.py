"""Human-readable, evaluator-only Episode progress and approximate ETA.

The reporter consumes loaded Episode counts and already measured result rows.
It never enters a policy or changes navigation, termination, or metric values.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from collections import deque
from statistics import median
from typing import Any, Callable, Iterable, TextIO


def _duration(seconds: float) -> str:
    whole = max(0, math.ceil(seconds))
    hours, remainder = divmod(whole, 3600)
    minutes, seconds_left = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_left:02d}"


def _metric(value: Any, *, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not math.isfinite(number):
        return "N/A"
    return f"{number:.2f}{suffix}" if suffix else f"{number:.3f}"


class ProgressReporter:
    """Report per-shard progress without estimating a route from its step budget."""

    def __init__(
        self,
        total: int,
        *,
        shard_index: int = 0,
        shard_count: int = 1,
        completed_results: Iterable[dict[str, Any]] = (),
        group_label: str = "LightNav",
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
        heartbeat_interval: float = 12.0,
        tty_override: bool | None = None,
        auto_heartbeat: bool = False,
    ) -> None:
        if type(total) is not int or total < 1:
            raise ValueError("progress total must be a positive loaded Episode count")
        if (
            type(shard_index) is not int or type(shard_count) is not int
            or shard_count < 1 or not 0 <= shard_index < shard_count
        ):
            raise ValueError("invalid progress shard index/count")
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        self.total = total
        self.shard_index = shard_index
        self.shard_count = shard_count
        self.group_label = group_label
        self.stream = stream if stream is not None else sys.stderr
        self.clock = clock
        self.heartbeat_interval = heartbeat_interval
        self.auto_heartbeat = auto_heartbeat
        self._lock = threading.RLock()
        self._ticker_stop: threading.Event | None = None
        self._ticker_thread: threading.Thread | None = None
        self._tty = (
            bool(getattr(self.stream, "isatty", lambda: False)())
            if tty_override is None else tty_override
        )
        self._run_start = clock()
        self._completed_keys: set[tuple[str, str]] = set()
        self.completed_count = 0
        self._measured_total = 0.0
        self._measured_count = 0
        self._durations: deque[float] = deque(maxlen=20)
        self.typical_duration_sec: float | None = None
        self._active_key: tuple[str, str] | None = None
        self._active_ordinal: int | None = None
        self._active_started: float | None = None
        self._active_step = 0
        self._active_max_steps = 500
        self._last_heartbeat = self._run_start
        self._tty_line_visible = False
        for result in completed_results:
            self._remember(result)
        if self.completed_count > self.total:
            raise ValueError("completed Episode count exceeds loaded shard manifest")
        self._emit(
            "=" * 60,
            f"LightNav Habitat Evaluation | {self.group_label} | {self.shard_label}",
            "=" * 60,
            self._overall_line(self._run_start),
            f"Resumed: {self.completed_count}/{self.total} completed Episodes",
        )

    @property
    def shard_label(self) -> str:
        return f"shard {self.shard_index}/{self.shard_count}"

    def _key(self, result: dict[str, Any]) -> tuple[str, str]:
        scene = result.get("scene_id")
        episode = result.get("habitat_episode_id")
        if scene is None or episode is None or not str(scene) or not str(episode):
            raise ValueError("progress result requires scene_id and habitat_episode_id")
        return str(scene), str(episode)

    def _record_duration(self, value: Any) -> None:
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(seconds) or seconds < 0:
            return
        self._durations.append(seconds)
        self._measured_total += seconds
        self._measured_count += 1
        if len(self._durations) >= 3:
            current_median = median(self._durations)
            self.typical_duration_sec = (
                current_median if self.typical_duration_sec is None
                else 0.75 * self.typical_duration_sec + 0.25 * current_median
            )

    def _remember(self, result: dict[str, Any]) -> None:
        key = self._key(result)
        if key in self._completed_keys:
            raise ValueError(f"duplicate completed Episode in progress: {key}")
        self._completed_keys.add(key)
        self.completed_count += 1
        self._record_duration(result.get("episode_time_sec"))

    def _elapsed(self, now: float) -> float:
        # Earlier attempts' wall time is unknown. Show this attempt explicitly
        # while counting saved Episode results in the progress numerator.
        return max(0.0, now - self._run_start)

    def _episode_elapsed(self, now: float) -> float:
        if self._active_started is None:
            return 0.0
        return max(0.0, now - self._active_started)

    def episode_eta(self, now: float | None = None) -> str:
        """Current Episode ETA, or an explicit calibration/uncertainty state."""
        with self._lock:
            if self.typical_duration_sec is None:
                return f"CALIBRATING ({len(self._durations)}/3 measured Episodes)"
            if self._active_key is None:
                return "N/A"
            elapsed = self._episode_elapsed(self.clock() if now is None else now)
            if elapsed >= self.typical_duration_sec:
                return "uncertain (past typical Episode duration)"
            return f"~{_duration(self.typical_duration_sec - elapsed)} (approximate)"

    def total_eta(self, now: float | None = None) -> str:
        """Smoothed ETA for this shard; never labelled as cross-shard global ETA."""
        with self._lock:
            if self.typical_duration_sec is None:
                return f"CALIBRATING ({len(self._durations)}/3 measured Episodes)"
            remaining = self.total - self.completed_count
            if remaining <= 0:
                return "00:00:00"
            now = self.clock() if now is None else now
            if self._active_key is not None:
                elapsed = self._episode_elapsed(now)
                if elapsed >= self.typical_duration_sec:
                    return "uncertain (current Episode past typical duration)"
                seconds = (remaining - 1) * self.typical_duration_sec
                seconds += self.typical_duration_sec - elapsed
            else:
                seconds = remaining * self.typical_duration_sec
            return f"~{_duration(seconds)} (approximate)"

    def _overall_line(self, now: float) -> str:
        width = 20
        filled = int(width * self.completed_count / self.total)
        bar = "█" * filled + "░" * (width - filled)
        pct = 100.0 * self.completed_count / self.total
        average = (
            _duration(self._measured_total / self._measured_count)
            if self._measured_count else "CALIBRATING"
        )
        return (
            f"Overall {self.shard_label}: [{bar}] {self.completed_count}/{self.total} "
            f"({pct:.1f}%) | Elapsed (this attempt): {_duration(self._elapsed(now))} | "
            f"Average: {average}/Episode | Remaining: {self.total - self.completed_count} | "
            f"ETA: {self.total_eta(now)}"
        )

    def _clear_tty(self) -> None:
        if self._tty_line_visible:
            self.stream.write("\r\x1b[2K")
            self._tty_line_visible = False

    def _emit(self, *lines: str) -> None:
        self._clear_tty()
        self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()

    def _ticker_loop(self, stop: threading.Event) -> None:
        """Keep heartbeats alive during a long policy.act or env.step call."""
        interval = 0.5 if self._tty else self.heartbeat_interval
        while not stop.wait(interval):
            with self._lock:
                if self._ticker_stop is not stop or self._active_key is None:
                    return
                self._tick_locked()

    def _stop_ticker_locked(self) -> threading.Thread | None:
        thread = self._ticker_thread
        if self._ticker_stop is not None:
            self._ticker_stop.set()
        self._ticker_stop = None
        self._ticker_thread = None
        return thread

    @staticmethod
    def _join_ticker(thread: threading.Thread | None) -> None:
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)

    def start(self, scene_id: str, episode_id: str, *, max_steps: int = 500) -> int:
        """Start one not-yet-completed Episode and return its shard-local ordinal."""
        with self._lock:
            if self._active_key is not None:
                raise RuntimeError("an Episode is already active in progress")
            key = (str(scene_id), str(episode_id))
            if not all(key) or key in self._completed_keys:
                raise ValueError(f"Episode is invalid or already completed in progress: {key}")
            if self.completed_count >= self.total:
                raise ValueError("no remaining Episodes in loaded shard manifest")
            if type(max_steps) is not int or max_steps < 1:
                raise ValueError("progress max_steps must be a positive integer")
            self._active_key = key
            self._active_ordinal = self.completed_count + 1
            self._active_started = self.clock()
            self._last_heartbeat = self._active_started
            self._active_step = 0
            self._active_max_steps = max_steps
            self._emit(
                f"[{self._active_ordinal}/{self.total}] START Episode ID: {key[1]} "
                f"| scene: {key[0]} | {self.shard_label}",
                self._overall_line(self._active_started),
            )
            if self.auto_heartbeat:
                stop = threading.Event()
                thread = threading.Thread(
                    target=self._ticker_loop, args=(stop,), daemon=True,
                    name=f"lightnav-progress-shard-{self.shard_index}",
                )
                self._ticker_stop = stop
                self._ticker_thread = thread
                thread.start()
            return self._active_ordinal

    def update_step(self, step: int, *, max_steps: int = 500) -> None:
        """Update the latest completed Step without performing progress I/O."""
        with self._lock:
            if self._active_key is None:
                raise RuntimeError("no active Episode for progress Step update")
            if (
                type(step) is not int or step < 0 or type(max_steps) is not int
                or max_steps < 1
            ):
                raise ValueError("progress Step and max_steps must be nonnegative integers")
            self._active_step = step
            self._active_max_steps = max_steps

    def _tick_locked(self) -> bool:
        if self._active_key is None or self._active_ordinal is None:
            return False
        now = self.clock()
        interval = 0.5 if self._tty else self.heartbeat_interval
        if now - self._last_heartbeat < interval:
            return False
        self._last_heartbeat = now
        eta_label = "Shard ETA" if self.shard_count > 1 else "Total ETA"
        line = (
            f"[{self._active_ordinal}/{self.total}] {self.shard_label} RUNNING | "
            f"Episode ID: {self._active_key[1]} | "
            f"Step: {self._active_step}/{self._active_max_steps} | "
            f"Episode elapsed: {_duration(self._episode_elapsed(now))} | "
            f"Episode ETA: {self.episode_eta(now)} | {eta_label}: {self.total_eta(now)}"
        )
        if self._tty:
            self.stream.write("\r\x1b[2K" + line)
            self.stream.flush()
            self._tty_line_visible = True
        else:
            self._emit(line, self._overall_line(now))
        return True

    def tick(self) -> bool:
        """Render the current Step if due; deterministic with an injected clock."""
        with self._lock:
            return self._tick_locked()

    def heartbeat(self, step: int, *, max_steps: int = 500) -> bool:
        """Update the Step and render a rate-limited heartbeat when due."""
        with self._lock:
            self.update_step(step, max_steps=max_steps)
            return self._tick_locked()

    def complete(self, result: dict[str, Any]) -> int:
        """Record one result exactly once and display its already measured metrics."""
        with self._lock:
            if self._active_key is None or self._active_ordinal is None:
                raise RuntimeError("no active Episode to complete in progress")
            key = self._key(result)
            if key != self._active_key:
                raise ValueError(f"progress result {key} differs from active {self._active_key}")
            ordinal = self._active_ordinal
            thread = self._stop_ticker_locked()
            self._remember(result)
            self._active_key = None
            self._active_started = None
            self._active_ordinal = None
            now = self.clock()
            duration = result.get("episode_time_sec")
            duration_text = (
                _duration(float(duration)) if isinstance(duration, (int, float))
                and math.isfinite(duration) and duration >= 0 else "N/A"
            )
            self._emit(
                f"[{ordinal}/{self.total}] COMPLETED | {self.shard_label} | "
                f"Episode ID: {key[1]} | Success: {result.get('success', 'N/A')} | "
                f"Steps: {result.get('steps', 'N/A')} | Duration: {duration_text}",
                "SR: " + _metric(result.get("success"))
                + " | OS: " + _metric(result.get("oracle_success"))
                + " | SPL: " + _metric(result.get("spl"))
                + " | nDTW: " + _metric(result.get("ndtw"))
                + " | NE: " + _metric(result.get("final_distance"), suffix="m"),
                self._overall_line(now),
            )
        self._join_ticker(thread)
        return ordinal

    def abort(self, reason: str = "INTERRUPTED") -> None:
        """Leave an unfinished Episode uncounted and clear any TTY heartbeat."""
        with self._lock:
            if self._active_key is None or self._active_ordinal is None:
                return
            thread = self._stop_ticker_locked()
            now = self.clock()
            self._emit(
                f"[{self._active_ordinal}/{self.total}] {reason} | {self.shard_label} | "
                f"Episode ID: {self._active_key[1]} | Step: {self._active_step} | "
                f"Episode elapsed: {_duration(self._episode_elapsed(now))}",
                self._overall_line(now),
            )
            self._active_key = None
            self._active_started = None
            self._active_ordinal = None
        self._join_ticker(thread)

    def finish(self) -> None:
        """Summarize this shard without claiming incomplete results completed."""
        self.abort()
        with self._lock:
            state = "COMPLETE" if self.completed_count == self.total else "INCOMPLETE"
            self._emit(f"{self.shard_label} {state}: {self.completed_count}/{self.total} Episodes")
