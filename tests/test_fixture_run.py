import json
import sys

import pytest

from flytype import cli
from flytype.config import Settings
from flytype.state import acquire_lock, release_lock


def run_cli(args):
    import subprocess

    return subprocess.run(
        [sys.executable, "-m", "flytype", *args], capture_output=True, text=True
    )


def test_fixture_run_produces_exact_final_text(tmp_path):
    out = tmp_path / "run"
    result = run_cli(
        ["run", "--fixture", "--seed", "42", "--steps", "2000", "--out", str(out)]
    )
    assert result.returncode == 0, result.stderr
    assert (out / "final.txt").read_text() == Settings.target
    assert not list(out.glob("brain-*.npz"))  # fixture mode never touches a brain
    events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
    assert all(e["source"] == "fixture" for e in events)
    assert all(e["mode"] == "fixture" for e in events)


def test_resume_continues_after_a_partial_run(tmp_path):
    out = tmp_path / "run"
    r1 = run_cli(["run", "--fixture", "--seed", "3", "--steps", "3", "--out", str(out)])
    assert r1.returncode == 0, r1.stderr
    state1 = json.loads((out / "state.json").read_text())
    assert state1["observation_count"] == 3

    r2 = run_cli(["run", "--fixture", "--seed", "3", "--steps", "2000", "--out", str(out)])
    assert r2.returncode == 0, r2.stderr
    assert (out / "final.txt").read_text() == Settings.target
    state2 = json.loads((out / "state.json").read_text())
    assert state2["observation_count"] > state1["observation_count"]


def test_stop_file_halts_cleanly_after_current_commit(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "STOP").write_text("")
    result = run_cli(
        ["run", "--fixture", "--seed", "1", "--steps", "100", "--out", str(out)]
    )
    assert result.returncode == 0, result.stderr
    assert not (out / "final.txt").exists()
    assert not (out / "state.json").exists()  # never even took one observation


def test_incompatible_resume_configuration_is_rejected(tmp_path):
    out = tmp_path / "run"
    r1 = run_cli(["run", "--fixture", "--seed", "1", "--steps", "1", "--out", str(out)])
    assert r1.returncode == 0, r1.stderr

    r2 = run_cli(["run", "--fixture", "--seed", "2", "--steps", "1", "--out", str(out)])
    assert r2.returncode == 1
    assert (out / "error.json").exists()


def test_max_attempts_per_character_halts_without_inserting_character(
    tmp_path, monkeypatch
):
    class AlwaysWrong:
        def __init__(self, seed):
            pass

        def decode(self, correct_side):
            wrong = "RIGHT" if correct_side == "LEFT" else "LEFT"
            return {
                "action": wrong,
                "left_hz": None,
                "right_hz": None,
                "difference_hz": None,
                "gate_spikes": None,
                "cell_ids": None,
                "source": "fixture",
            }

        def state(self):
            return {}

        def restore(self, state):
            pass

    monkeypatch.setattr(cli, "FixtureDecoder", AlwaysWrong)
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    settings = Settings(target="a", fixture=True, max_attempts_per_character=3, steps=0)
    lock = acquire_lock(out)
    try:
        with pytest.raises(SystemExit):
            cli._run_loop(settings, out)
    finally:
        release_lock(lock)
    assert not (out / "final.txt").exists()
    state = json.loads((out / "state.json").read_text())
    assert state["task"]["index"] == 0  # never inserted the character
    assert state["halted"] is not None
