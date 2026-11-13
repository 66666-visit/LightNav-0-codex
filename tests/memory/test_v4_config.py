"""V4 settings have a separate protocol and do not mutate V3 or Legacy."""

from __future__ import annotations

import unittest
from dataclasses import asdict

from lightnav.memory.rgm_config import RGMConfig
from lightnav.memory.v3_config import RGMV3Config
from lightnav.memory.v4_config import RGM_V4_PROTOCOL_VERSION, RGMV4Config


class V4ConfigTests(unittest.TestCase):
    def test_defaults_and_independent_protocol(self) -> None:
        config = RGMV4Config()
        self.assertFalse(config.enabled)
        self.assertEqual(config.scope, "episode_local")
        self.assertEqual(config.prompt_max_records, 3)
        self.assertEqual(config.prompt_max_chars, 400)
        self.assertTrue(config.revisit_enabled)
        self.assertEqual(config.protocol_version, RGM_V4_PROTOCOL_VERSION)
        self.assertNotEqual(config.protocol_version, RGMV3Config().protocol_version)

    def test_explicit_scene_scope_and_revisit_switch(self) -> None:
        config = RGMV4Config(enabled=True, scope="scene_persistent", revisit_enabled=False)
        self.assertTrue(config.enabled)
        self.assertFalse(config.revisit_enabled)
        self.assertEqual(config.scope, "scene_persistent")

    def test_invalid_scope_budget_or_version_is_rejected(self) -> None:
        for kwargs in (
            {"scope": "all_scenes"},
            {"prompt_max_records": 0},
            {"prompt_max_records": True},
            {"prompt_max_chars": 0},
            {"prompt_max_chars": 1.5},
            {"protocol_version": "rgm_v3_v1"},
            {"revisit_enabled": "yes"},
            {"enabled": 1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RGMV4Config(**kwargs)

    def test_legacy_and_v3_defaults_have_no_v4_fields(self) -> None:
        legacy = asdict(RGMConfig())
        v3 = asdict(RGMV3Config())
        for values in (legacy, v3):
            self.assertNotIn("revisit_enabled", values)
            self.assertNotIn("rgm_v4_enabled", values)
        self.assertFalse(legacy["enabled"])
        self.assertFalse(v3["enabled"])


if __name__ == "__main__":
    unittest.main()
