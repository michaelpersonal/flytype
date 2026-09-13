import json

from flytype.breakout import FIELD_BOTTOM
from flytype.config import Settings
from flytype.decoder import FixtureDecoder
from flytype.play_session import ContinuousPlaySession


def open_fixture(out, seed=7, **overrides):
    settings = Settings(
        seed=seed,
        fixture=True,
        fast=True,
        egocentric=True,
        decoder_baseline_obs=0,
        **overrides,
    )
    decoder = FixtureDecoder(seed ^ 0x9E3779B9)
    return ContinuousPlaySession(
        settings=settings,
        out=out,
        dataset_info={"mode": "fixture"},
        fixture_decoder=decoder,
    )


def test_fixture_session_commits_public_snapshot(tmp_path):
    session = open_fixture(tmp_path)
    snapshot = session.advance()
    saved = json.loads((tmp_path / "state.json").read_text())
    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert snapshot["observation"] == saved["observation_count"] == 1
    assert snapshot["source"] == event["source"] == "fixture"
    assert -1 <= snapshot["motor"]["control"] <= 1
    assert snapshot["retinal_png"].startswith("data:image/png;base64,")
    assert saved["task"]["episode"] == 0


def test_fixture_session_rolls_over_continuously(tmp_path):
    session = open_fixture(
        tmp_path,
        lives=1,
        brick_rows=1,
        brick_columns=1,
        paddle_width=320,
    )
    for expected_episode in (1, 2):
        session.game.bricks = [False] * len(session.game.bricks)
        session.advance()
        assert session.episode == expected_episode
    episodes = sorted((tmp_path / "episodes").glob("episode-*.json"))
    assert len(episodes) >= 2
    assert [json.loads(path.read_text())["episode"] for path in episodes[:2]] == [0, 1]


def test_event_separates_the_paddle_action_from_the_dnp20_readout(tmp_path):
    session = open_fixture(tmp_path)
    session.advance()
    event = json.loads((tmp_path / "events.jsonl").read_text())
    assert event["action"] in ("LEFT", "RIGHT", "HOLD")
    assert event["dnp20_action"] == event["neural"]["action"]


def test_motion_term_is_dropped_across_serves_and_rollovers(tmp_path):
    """The motor decoder must not read a re-served ball as a huge movement."""
    session = open_fixture(tmp_path, lives=3)

    class Spy:
        cell_ids = ()
        artifact_sha256 = "spy"
        forgotten = 0

        def forget_motion(self):
            self.forgotten += 1

        def state(self):
            return {"previous_offset": None}

    session.motor = Spy()

    # Drop a ball past the paddle; the game re-serves it elsewhere.
    game = session.game
    game.paddle_x, game.ball_x = 260.0, 0.0
    game.ball_y = float(FIELD_BOTTOM)
    game.ball_vy = abs(game.ball_vy_magnitude)
    session.advance()
    assert game.last_event == "lost" and game.lives == 2
    assert session.motor.forgotten == 1

    # And again when the episode ends and a fresh game replaces it.
    session.game.bricks = [False] * len(session.game.bricks)
    assert session.game.done
    session.advance()
    assert session.episode == 1 and not session.game.done
    assert session.motor.forgotten == 2


def test_recorded_events_reproduce_the_exact_frame_the_network_saw(tmp_path):
    """A replay must rebuild the observation bytes, not something like them."""
    import numpy as np
    from PIL import Image

    from flytype.arena import render_arena

    session = open_fixture(tmp_path)
    for _ in range(3):
        session.advance()
    last = json.loads((tmp_path / "events.jsonl").read_text().splitlines()[-1])
    assert last["frame_annotated"] is False

    replayed = render_arena(
        last["before"],
        action="NONE" if not last["frame_annotated"] else last["action"],
        stimulus="NONE",
        egocentric=True,
    )
    saved = np.asarray(Image.open(tmp_path / "latest-input.png").convert("RGB"))
    assert np.array_equal(replayed, saved)


def test_fixture_session_resume_matches_uninterrupted_run(tmp_path):
    left = open_fixture(tmp_path / "left", seed=11)
    right = open_fixture(tmp_path / "right", seed=11)
    for _ in range(12):
        left.advance()
        right.advance()

    resumed = open_fixture(tmp_path / "right", seed=11)
    expected = left.advance()
    actual = resumed.advance()
    assert actual["observation"] == expected["observation"]
    assert actual["world"] == expected["world"]
    assert actual["motor"] == expected["motor"]
