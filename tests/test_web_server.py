import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from flytype.config import Settings
from flytype.decoder import FixtureDecoder
from flytype.play_session import ContinuousPlaySession
from flytype.web_server import GameRunner, create_server


def fixture_session(out):
    settings = Settings(seed=3, fixture=True, fast=True, egocentric=True)
    return ContinuousPlaySession(
        settings=settings,
        out=out,
        dataset_info={"mode": "fixture"},
        fixture_decoder=FixtureDecoder(settings.seed ^ 0x9E3779B9),
    )


@pytest.fixture
def live_server(tmp_path):
    runner = GameRunner(fixture_session(tmp_path), tick_seconds=0.01)
    server = create_server(runner, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    runner.start()
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    yield runner, server, url
    runner.stop()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def get_json(url):
    with urllib.request.urlopen(url, timeout=3) as response:
        return json.load(response)


def test_state_endpoint_streams_committed_fixture_updates(live_server):
    runner, _, url = live_server
    deadline = time.monotonic() + 3
    state = get_json(url + "/api/state")
    while state.get("observation", 0) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
        state = get_json(url + "/api/state")
    assert state["source"] == "fixture"
    assert state["observation"] >= 2
    assert state["runner_status"] in ("running", "computing")
    assert "token" not in state
    assert runner.revision >= 2


def test_controls_require_token_and_pause_after_current_step(live_server):
    runner, _, url = live_server
    request = urllib.request.Request(
        url + "/api/control",
        data=b'{"action":"pause"}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as denied:
        urllib.request.urlopen(request, timeout=3)
    assert denied.value.code == 403

    request.add_header("X-FlyType-Token", runner.control_token)
    with urllib.request.urlopen(request, timeout=3) as response:
        assert json.load(response)["ok"] is True
    deadline = time.monotonic() + 3
    while runner.status not in ("paused", "stopped") and time.monotonic() < deadline:
        time.sleep(0.01)
    assert runner.status == "paused"


def test_static_paths_cannot_escape_packaged_root(live_server):
    _, _, url = live_server
    with urllib.request.urlopen(url + "/", timeout=3) as response:
        html = response.read().decode()
    assert "FLYBREAK" in html
    assert "__CONTROL_TOKEN__" not in html
    with pytest.raises(urllib.error.HTTPError) as missing:
        urllib.request.urlopen(url + "/../pyproject.toml", timeout=3)
    assert missing.value.code == 404

