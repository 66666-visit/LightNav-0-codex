"""Progress and ETA checks with a fake monotonic clock (no sleeps or model)."""

from __future__ import annotations

import io
import unittest

from lightnav.habitat.progress import ProgressReporter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeStream(io.StringIO):
    def __init__(self, *, tty: bool = False) -> None:
        super().__init__()
        self.tty = tty
        self.flushes = 0

    def isatty(self) -> bool:
        return self.tty

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


def result(scene: str, episode: str, duration: float, **overrides):
    row = {
        "scene_id": scene, "habitat_episode_id": episode,
        "episode_time_sec": duration, "success": True,
        "oracle_success": True, "spl": 0.832, "ndtw": 0.923,
        "final_distance": 1.24, "steps": 53,
    }
    row.update(overrides)
    return row


class ProgressReporterTest(unittest.TestCase):
    def make(self, total=220, *, completed=(), shard_index=0, shard_count=1,
             tty=False, tty_override=None, auto_heartbeat=False,
             heartbeat_interval=12.0):
        clock = FakeClock()
        stream = FakeStream(tty=tty)
        reporter = ProgressReporter(
            total, shard_index=shard_index, shard_count=shard_count,
            completed_results=completed, group_label="baseline_off",
            stream=stream, clock=clock, tty_override=tty_override,
            auto_heartbeat=auto_heartbeat, heartbeat_interval=heartbeat_interval,
        )
        return reporter, clock, stream

    def test_pilot_220_uses_loaded_total(self) -> None:
        reporter, _, stream = self.make(220)
        self.assertEqual(reporter.start("scene", "159"), 1)
        self.assertIn("[1/220] START Episode ID: 159", stream.getvalue())
        self.assertIn("0/220", stream.getvalue())
        self.assertIn("LightNav Habitat Evaluation", stream.getvalue())

    def test_full_1839_uses_loaded_total(self) -> None:
        reporter, _, stream = self.make(1839)
        reporter.start("scene", "0")
        self.assertIn("[1/1839] START", stream.getvalue())

    def test_start_complete_ordinals_advance_once(self) -> None:
        reporter, _, stream = self.make(2)
        reporter.start("a", "1")
        self.assertEqual(reporter.complete(result("a", "1", 7)), 1)
        self.assertEqual(reporter.completed_count, 1)
        reporter.start("b", "2")
        self.assertEqual(reporter.complete(result("b", "2", 9)), 2)
        self.assertIn("[2/2] COMPLETED", stream.getvalue())
        reporter.finish()
        self.assertIn("shard 0/1 COMPLETE: 2/2", stream.getvalue())

    def test_current_episode_elapsed_is_from_clock(self) -> None:
        reporter, clock, stream = self.make()
        reporter.start("a", "1")
        clock.advance(135)
        self.assertTrue(reporter.heartbeat(60, max_steps=500))
        self.assertIn("Step: 60/500", stream.getvalue())
        self.assertIn("Episode elapsed: 00:02:15", stream.getvalue())
        self.assertIn("Elapsed (this attempt): 00:02:15", stream.getvalue())

    def test_eta_calibrates_until_three_measured_episodes(self) -> None:
        reporter, _, _ = self.make(completed=(
            result("a", "1", 10), result("a", "2", 12),
        ))
        reporter.start("a", "3")
        self.assertIn("CALIBRATING", reporter.episode_eta())
        self.assertIn("CALIBRATING", reporter.total_eta())
        reporter.complete(result("a", "3", 11))
        self.assertEqual(reporter.typical_duration_sec, 11)
        reporter.start("a", "4")
        self.assertIn("approximate", reporter.episode_eta())
        self.assertIn("approximate", reporter.total_eta())

    def test_rolling_median_then_ema_smooths_outlier(self) -> None:
        reporter, _, _ = self.make(completed=(
            result("a", str(index), 10) for index in range(3)
        ))
        self.assertEqual(reporter.typical_duration_sec, 10)
        for index in range(3, 7):
            reporter.start("a", str(index))
            reporter.complete(result("a", str(index), 100))
        self.assertGreater(reporter.typical_duration_sec, 10)
        self.assertLess(reporter.typical_duration_sec, 100)

    def test_past_typical_duration_is_uncertain_not_zero(self) -> None:
        reporter, clock, _ = self.make(completed=(
            result("a", "1", 10), result("a", "2", 10), result("a", "3", 10),
        ))
        reporter.start("a", "4")
        clock.advance(11)
        self.assertIn("uncertain", reporter.episode_eta())
        self.assertIn("uncertain", reporter.total_eta())
        self.assertNotIn("~00:00:00", reporter.episode_eta())

    def test_resume_counts_saved_results_before_next_start(self) -> None:
        saved = [result("a", str(index), 10) for index in range(3)]
        reporter, clock, stream = self.make(completed=saved)
        self.assertEqual(reporter.completed_count, 3)
        self.assertEqual(reporter.start("a", "3"), 4)
        self.assertIn("[4/220] START", stream.getvalue())
        self.assertIn("Resumed: 3/220", stream.getvalue())
        clock.advance(2)
        self.assertIn("Elapsed (this attempt): 00:00:02", reporter._overall_line(clock()))

    def test_duplicate_completed_episode_is_rejected(self) -> None:
        saved = result("a", "1", 10)
        with self.assertRaisesRegex(ValueError, "duplicate completed Episode"):
            self.make(completed=(saved, saved))
        reporter, _, _ = self.make(completed=(saved,))
        with self.assertRaisesRegex(ValueError, "already completed"):
            reporter.start("a", "1")
        self.assertEqual(reporter.completed_count, 1)

    def test_multi_shard_is_explicitly_local(self) -> None:
        reporter, clock, stream = self.make(
            110, shard_index=1, shard_count=2, completed=(result("a", "1", 10),),
        )
        reporter.start("a", "2")
        clock.advance(12)
        reporter.heartbeat(5)
        self.assertIn("shard 1/2", stream.getvalue())
        self.assertIn("[2/110] START", stream.getvalue())
        self.assertIn("Shard ETA:", stream.getvalue())
        self.assertNotIn("/220] START", stream.getvalue())

    def test_non_tty_heartbeat_is_rate_limited_and_flushed(self) -> None:
        reporter, clock, stream = self.make()
        reporter.start("a", "1")
        baseline_flushes = stream.flushes
        clock.advance(11)
        self.assertFalse(reporter.heartbeat(2))
        clock.advance(1)
        self.assertTrue(reporter.heartbeat(3))
        self.assertEqual(stream.getvalue().count("RUNNING"), 1)
        self.assertGreater(stream.flushes, baseline_flushes)

    def test_update_step_has_no_io_and_manual_tick_uses_latest_step(self) -> None:
        reporter, clock, stream = self.make()
        reporter.start("a", "1", max_steps=500)
        prior = stream.getvalue()
        reporter.update_step(60, max_steps=500)
        self.assertEqual(stream.getvalue(), prior)
        clock.advance(12)
        self.assertTrue(reporter.tick())
        self.assertIn("Step: 60/500", stream.getvalue())

    def test_auto_ticker_starts_and_stops_without_waiting_in_test(self) -> None:
        reporter, clock, stream = self.make(auto_heartbeat=True, heartbeat_interval=1000.0)
        reporter.start("a", "1")
        ticker = reporter._ticker_thread
        self.assertIsNotNone(ticker)
        self.assertTrue(ticker.is_alive())
        reporter.update_step(4)
        clock.advance(1000)
        # The same tick path is called by the daemon after its Event.wait.
        self.assertTrue(reporter.tick())
        self.assertIn("Step: 4/500", stream.getvalue())
        reporter.complete(result("a", "1", 10))
        self.assertFalse(ticker.is_alive())
        self.assertIsNone(reporter._ticker_thread)

    def test_auto_ticker_stops_on_abort(self) -> None:
        reporter, _, _ = self.make(auto_heartbeat=True, heartbeat_interval=1000.0)
        reporter.start("a", "1")
        ticker = reporter._ticker_thread
        reporter.abort()
        self.assertFalse(ticker.is_alive())
        self.assertFalse(reporter.tick())

    def test_tty_uses_one_dynamic_line(self) -> None:
        reporter, clock, stream = self.make(tty=True)
        reporter.start("a", "1")
        clock.advance(1)
        self.assertTrue(reporter.heartbeat(2))
        self.assertIn("\r\x1b[2K", stream.getvalue())
        self.assertNotIn("\nOverall", stream.getvalue().split("RUNNING")[-1])
        reporter.abort()
        self.assertIn("INTERRUPTED", stream.getvalue())

    def test_tty_override_can_force_line_mode(self) -> None:
        reporter, clock, stream = self.make(tty=True, tty_override=False)
        reporter.start("a", "1")
        clock.advance(12)
        self.assertTrue(reporter.heartbeat(1))
        self.assertNotIn("\r\x1b[2K", stream.getvalue())

    def test_metrics_and_measured_duration_are_printed_unchanged(self) -> None:
        reporter, clock, stream = self.make()
        reporter.start("a", "159")
        clock.advance(100)
        reporter.complete(result("a", "159", 135, success=True, oracle_success=False))
        output = stream.getvalue()
        self.assertIn("Episode ID: 159", output)
        self.assertIn("Duration: 00:02:15", output)
        self.assertIn("Success: True", output)
        self.assertIn("SR: 1.000 | OS: 0.000 | SPL: 0.832", output)
        self.assertIn("nDTW: 0.923 | NE: 1.24m", output)

    def test_abort_does_not_count_episode_and_finish_says_incomplete(self) -> None:
        reporter, clock, stream = self.make(2)
        reporter.start("a", "1")
        clock.advance(30)
        reporter.abort("FAILED")
        self.assertEqual(reporter.completed_count, 0)
        reporter.finish()
        self.assertIn("[1/2] FAILED", stream.getvalue())
        self.assertIn("INCOMPLETE: 0/2", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
