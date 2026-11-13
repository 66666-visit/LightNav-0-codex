"""RGM V3 settings remain independent of immutable legacy RGM configuration."""

from __future__ import annotations

import unittest
from dataclasses import asdict

from lightnav.memory.rgm_config import RGMConfig
from lightnav.memory.v3_config import RGMV3Config, RGM_V3_PROTOCOL_VERSION


class V3ConfigTests(unittest.TestCase):
    def test_default_scope_and_protocol(self) -> None:
        config = RGMV3Config(enabled=True)
        self.assertEqual(config.scope, "episode_local")
        self.assertEqual(config.protocol_version, RGM_V3_PROTOCOL_VERSION)

    def test_invalid_scope_or_version_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RGMV3Config(scope="all_scenes")
        with self.assertRaises(ValueError):
            RGMV3Config(protocol_version="rgm_pilot_88_v1")

    def test_legacy_config_has_no_v3_fields(self) -> None:
        legacy = asdict(RGMConfig())
        self.assertNotIn("scope", legacy)
        self.assertNotIn("protocol_version", legacy)


if __name__ == "__main__":
    unittest.main()
