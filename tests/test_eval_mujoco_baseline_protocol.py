"""Exercise the evaluator's transparent LightNav WebSocket relay."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_mujoco_baseline import ModelProxy, _run_navigation  # noqa: E402


class _RunningDemo:
    snapshot = {
        "mpc": {"error": ""},
        "vln": {"state": "RUNNING", "stop": False, "connected": True},
        "control": {"auto": True},
    }


def _request(action: str, data: dict) -> str:
    return json.dumps({"action": action, "data": data}, separators=(",", ":"))


def _response(action: str, data: dict) -> str:
    return json.dumps({"action": action, "data": data}, separators=(",", ":"))


def _fake_aiohttp(respond):
    """A tiny in-memory transport that runs ModelProxy._handle end to end."""
    text_type, binary_type, closed_type = 1, 2, 3
    received: list[str] = []
    downstreams = []

    class Downstream:
        def __init__(self, **_kwargs):
            self.sent: list[str] = []
            self.request = None
            downstreams.append(self)

        async def prepare(self, request):
            self.request = request

        def __aiter__(self):
            async def messages():
                for raw in self.request.messages:
                    yield SimpleNamespace(type=text_type, data=raw)

            return messages()

        async def send_str(self, raw):
            self.sent.append(raw)

        async def send_bytes(self, raw):
            self.sent.append(raw)

        async def close(self):
            pass

    class Upstream:
        def __init__(self):
            self.pending = None

        async def send_str(self, raw):
            received.append(raw)
            self.pending = respond(raw)

        async def send_bytes(self, raw):
            received.append(raw)
            self.pending = respond(raw)

        async def receive(self):
            if self.pending is None:
                return SimpleNamespace(type=closed_type, data=None)
            return SimpleNamespace(type=text_type, data=self.pending)

    class UpstreamContext:
        async def __aenter__(self):
            return Upstream()

        async def __aexit__(self, *_args):
            pass

    class ClientSession:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        def ws_connect(self, _url, **_kwargs):
            return UpstreamContext()

    fake = types.ModuleType("aiohttp")
    fake.ClientSession = ClientSession
    fake.ClientTimeout = lambda **_kwargs: None
    fake.WSMsgType = SimpleNamespace(TEXT=text_type, BINARY=binary_type)
    fake.web = SimpleNamespace(WebSocketResponse=Downstream)
    return fake, received, downstreams


class ProxyProtocolTest(unittest.TestCase):
    def test_in_memory_full_exchange_preserves_login_next_reset(self) -> None:
        requests = [
            _request("login", {"clientId": "baseline-test"}),
            _request("next", {"seq": 11, "image": "abc", "instruction": "go"}),
            _request("next", {"seq": 12, "image": "def", "instruction": "go"}),
            _request("reset", {}),
        ]
        replies = [
            _response("login", {"rc": 0, "msg": "ok"}),
            _response("next", {"rc": 0, "seq": 11, "actions": {"step": 1, "actions": []},
                               "stop": False, "latency_ms": 7.5}),
            _response("next", {"rc": 0, "seq": 12, "actions": {"step": 2, "actions": []},
                               "stop": True, "latency_ms": 8.5}),
            _response("reset", {"rc": 0, "msg": "ok"}),
        ]
        pending = iter(replies)
        fake, upstream_requests, downstreams = _fake_aiohttp(lambda _raw: next(pending))
        proxy = ModelProxy("ws://fake-model/ws")
        with patch.dict(sys.modules, {"aiohttp": fake}):
            asyncio.run(proxy._handle(SimpleNamespace(messages=requests)))

        self.assertEqual(upstream_requests, requests)
        self.assertEqual(downstreams[0].sent, replies)
        self.assertIsNone(proxy.error)
        self.assertEqual([step["seq"] for step in proxy.steps], [11, 12])
        self.assertEqual([step["history_frames"] for step in proxy.steps], [1, 2])
        self.assertEqual([step["stop"] for step in proxy.steps], [False, True])
        self.assertEqual([step["server_latency_ms"] for step in proxy.steps], [7.5, 8.5])
        self.assertTrue(all(step["proxy_roundtrip_ms"] >= 0 for step in proxy.steps))

    def test_in_memory_disconnect_is_error_without_stop(self) -> None:
        login = _request("login", {"clientId": "baseline-test"})
        next_request = _request("next", {"seq": 1, "image": "abc", "instruction": "go"})

        def respond(raw):
            return _response("login", {"rc": 0, "msg": "ok"}) if raw == login else None

        fake, upstream_requests, downstreams = _fake_aiohttp(respond)
        proxy = ModelProxy("ws://fake-model/ws")
        with patch.dict(sys.modules, {"aiohttp": fake}):
            asyncio.run(proxy._handle(SimpleNamespace(messages=[login, next_request])))

        self.assertEqual(upstream_requests, [login, next_request])
        self.assertEqual(len(downstreams[0].sent), 1)
        self.assertIn("closed before its response", proxy.error)
        self.assertFalse(any(step.get("stop") is True for step in proxy.steps))
        self.assertEqual(asyncio.run(_run_navigation(_RunningDemo(), proxy, 1.0, 5)), "proxy_error")

    def test_in_memory_bad_sequence_is_not_a_valid_stop(self) -> None:
        request = _request("next", {"seq": 7, "image": "abc", "instruction": "go"})
        wrong = _response("next", {
            "rc": 0, "seq": 8, "actions": {"step": 1, "actions": []},
            "stop": False, "latency_ms": 3.0,
        })
        fake, _, downstreams = _fake_aiohttp(lambda _raw: wrong)
        proxy = ModelProxy("ws://fake-model/ws")
        with patch.dict(sys.modules, {"aiohttp": fake}):
            asyncio.run(proxy._handle(SimpleNamespace(messages=[request])))

        self.assertEqual(downstreams[0].sent, [wrong])  # relay never rewrites it
        self.assertIn("sequence mismatch", proxy.error)
        self.assertFalse(any(step.get("stop") is True for step in proxy.steps))
        self.assertEqual(asyncio.run(_run_navigation(_RunningDemo(), proxy, 1.0, 5)), "proxy_error")

    def test_in_memory_model_error_response_is_inference_failure(self) -> None:
        request = _request("next", {"seq": 1, "image": "abc", "instruction": "go"})
        failure = _response("next", {"rc": 500, "seq": 1, "msg": "synthetic model failure"})
        fake, _, downstreams = _fake_aiohttp(lambda _raw: failure)
        proxy = ModelProxy("ws://fake-model/ws")
        with patch.dict(sys.modules, {"aiohttp": fake}):
            asyncio.run(proxy._handle(SimpleNamespace(messages=[request])))

        self.assertEqual(downstreams[0].sent, [failure])
        self.assertEqual(proxy.steps[0]["rc"], 500)
        self.assertIsNone(proxy.steps[0]["stop"])
        self.assertEqual(asyncio.run(_run_navigation(_RunningDemo(), proxy, 1.0, 5)), "inference_error")

    @unittest.skipUnless(importlib.util.find_spec("aiohttp"), "aiohttp is not installed locally")
    def test_loopback_fake_model_server_and_real_proxy(self) -> None:
        async def check() -> None:
            from aiohttp import ClientSession, web

            received = []
            replies = []

            async def model_handler(request):
                websocket = web.WebSocketResponse()
                await websocket.prepare(request)
                async for message in websocket:
                    received.append(message.data)
                    payload = json.loads(message.data)
                    action = payload["action"]
                    if action == "next":
                        seq = payload["data"]["seq"]
                        response = _response("next", {
                            "rc": 0, "seq": seq,
                            "actions": {"step": seq, "actions": [[0.0, 0.0, 0.0]]},
                            "stop": seq == 2, "latency_ms": 4.0 + seq,
                        })
                    else:
                        response = _response(action, {"rc": 0, "msg": "ok"})
                    replies.append(response)
                    await websocket.send_str(response)
                return websocket

            app = web.Application()
            app.router.add_get("/ws", model_handler)
            runner = web.AppRunner(app, access_log=None)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = site._server.sockets[0].getsockname()[1]
            proxy = ModelProxy(f"ws://127.0.0.1:{port}/ws")
            try:
                await proxy.start()
                requests = [
                    _request("login", {"clientId": "loopback-test"}),
                    _request("next", {"seq": 1, "image": "abc", "instruction": "go"}),
                    _request("next", {"seq": 2, "image": "def", "instruction": "go"}),
                    _request("reset", {}),
                ]
                async with ClientSession() as client:
                    async with client.ws_connect(proxy.url) as websocket:
                        observed = []
                        for raw in requests:
                            await websocket.send_str(raw)
                            observed.append((await websocket.receive()).data)
                self.assertEqual(received, requests)
                self.assertEqual(observed, replies)
                self.assertEqual([step["history_frames"] for step in proxy.steps], [1, 2])
                self.assertEqual([step["stop"] for step in proxy.steps], [False, True])
                self.assertEqual([step["server_latency_ms"] for step in proxy.steps], [5.0, 6.0])
                self.assertIsNone(proxy.error)
            finally:
                await proxy.close()
                await runner.cleanup()

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
