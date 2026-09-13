import json
import random
import subprocess
import sys

import numpy as np
import pytest

from flytype.arena import render_arena
from flytype.breakout import (
    BALL_SIZE,
    FIELD_BOTTOM,
    FIELD_TOP,
    WIDTH,
    BreakoutGame,
    new_game_rng,
)
from flytype.config import Settings


def make_game(seed=0, **kwargs):
    return BreakoutGame(Settings(**kwargs), new_game_rng(seed))


def test_paddle_moves_only_where_the_action_says():
    game = make_game()
    start = game.paddle_x
    game.step("RIGHT")
    assert game.paddle_x == start + game.paddle_speed
    game.step("LEFT")
    assert game.paddle_x == start
    held = game.paddle_x
    game.step("HOLD")
    assert game.paddle_x == held


def test_paddle_is_never_nudged_toward_the_ball():
    """Across a whole episode the paddle only ever moves by exactly one step."""
    game = make_game(seed=3)
    rng = random.Random(11)
    while not game.done and game.tick < 400:
        before = game.paddle_x
        action = rng.choice(["LEFT", "RIGHT", "HOLD"])
        game.step(action)
        delta = game.paddle_x - before
        expected = {"LEFT": -game.paddle_speed, "RIGHT": game.paddle_speed, "HOLD": 0}
        # Clamping at a wall is the only permitted deviation.
        assert delta == expected[action] or game.paddle_x in (
            0.0,
            WIDTH - game.paddle_width,
        )


def test_random_play_tracks_at_chance():
    """The scoring definition must put an agent that cannot see the ball at 0.5."""
    toward = away = 0
    for seed in range(60):
        game = make_game(seed=seed)
        rng = random.Random(seed * 977 + 5)
        while not game.done and game.tick < 600:
            game.step(rng.choice(["LEFT", "RIGHT", "HOLD"]))
        toward += game.toward
        away += game.away
    rate = toward / (toward + away)
    assert 0.47 < rate < 0.53, rate


def test_a_tracking_agent_beats_chance_decisively():
    perfect = make_game(seed=1)
    while not perfect.done and perfect.tick < 3000:
        perfect.step(perfect.tracking_direction() or "HOLD")
    blind = make_game(seed=1)
    rng = random.Random(1)
    while not blind.done and blind.tick < 3000:
        blind.step(rng.choice(["LEFT", "RIGHT", "HOLD"]))
    assert perfect.paddle_hits > blind.paddle_hits * 3
    assert perfect.summary()["tracking_rate"] == 1.0


def test_ball_stays_inside_the_mapped_photoreceptor_band():
    game = make_game(seed=5)
    rng = random.Random(7)
    while not game.done and game.tick < 800:
        assert -1 <= game.ball_x <= WIDTH - BALL_SIZE + 1
        assert FIELD_TOP - 1 <= game.ball_y <= FIELD_BOTTOM + 1
        game.step(rng.choice(["LEFT", "RIGHT", "HOLD"]))


def test_losing_every_life_ends_the_episode():
    game = make_game(seed=2, lives=1)
    while not game.done and game.tick < 2000:
        game.step("HOLD")
    assert game.done and game.lives == 0 and game.misses == 1
    with pytest.raises(RuntimeError):
        game.step("HOLD")


def test_rejects_unknown_action():
    with pytest.raises(ValueError):
        make_game().step("UP")


def test_state_round_trips_exactly():
    game = make_game(seed=4)
    rng = random.Random(2)
    for _ in range(40):
        game.step(rng.choice(["LEFT", "RIGHT", "HOLD"]))
    saved = json.loads(json.dumps(game.state()))
    clone = make_game(seed=4)
    clone.restore(saved)
    for action in ["LEFT", "RIGHT", "HOLD", "RIGHT", "RIGHT"]:
        assert game.step(action) == clone.step(action)
        assert game.view() == clone.view()


def test_restore_refuses_mismatched_geometry():
    saved = make_game().state()
    with pytest.raises(ValueError):
        make_game(paddle_width=40).restore(saved)


def test_arena_frame_is_a_valid_observation():
    frame = render_arena(make_game().view())
    assert frame.shape == (180, WIDTH, 3) and frame.dtype == np.uint8


def test_arena_renders_both_paddle_positions_identically_apart_from_position():
    """Nothing in the drawing marks a preferred side."""
    left = make_game()
    left.paddle_x = 20.0
    right = make_game()
    right.paddle_x = WIDTH - right.paddle_width - 20.0
    a = render_arena(left.view())[:, :, :]
    b = render_arena(right.view())[:, :, :]
    # The same pixels, mirrored about the field's vertical axis, in the paddle band.
    band = slice(55, 64)
    assert np.array_equal(
        np.sort(a[band].reshape(-1, 3), axis=0),
        np.sort(b[band].reshape(-1, 3), axis=0),
    )


def test_fixture_play_run_is_reproducible(tmp_path):
    def run(out):
        subprocess.run(
            [sys.executable, "-m", "flytype", "play", "--out", str(out),
             "--fixture", "--fast", "--steps", "80", "--seed", "9"],
            check=True, capture_output=True,
        )
        return (out / "events.jsonl").read_text()

    assert run(tmp_path / "a") == run(tmp_path / "b")


def _paddle_columns(frame):
    """Columns where the paddle's own colour appears in the paddle band."""
    band = frame[55:64]
    mask = (band[:, :, 1] > 200) & (band[:, :, 0] < 120)
    return np.flatnonzero(mask.any(axis=0))


def test_egocentric_view_puts_the_paddle_at_frame_centre():
    for paddle_x in (0.0, 60.0, 160.0, WIDTH - 56.0):
        game = make_game()
        game.paddle_x = paddle_x
        cols = _paddle_columns(render_arena(game.view(), egocentric=True))
        centre = (cols[0] + cols[-1]) / 2
        assert abs(centre - WIDTH / 2) <= 1.5, (paddle_x, centre)


def test_egocentric_view_encodes_the_ball_offset_not_its_world_position():
    """The same ball-relative-to-paddle geometry renders the same way."""
    def ball_centre(paddle_x, offset):
        game = make_game()
        game.paddle_x = paddle_x
        game.ball_x = paddle_x + game.paddle_width / 2 - BALL_SIZE / 2 + offset
        game.ball_y = 34.0
        frame = render_arena(game.view(), egocentric=True)
        band = frame[34:44]
        cols = np.flatnonzero((band > 250).all(axis=2).any(axis=0))
        return (cols[0] + cols[-1]) / 2

    for offset in (-80.0, -30.0, 30.0, 80.0):
        seen = {ball_centre(px, offset) for px in (60.0, 120.0, 200.0)}
        assert len(seen) == 1, (offset, seen)
        assert (seen.pop() - WIDTH / 2 > 0) == (offset > 0)


def test_fixed_view_does_not_move_the_paddle_to_centre():
    game = make_game()
    game.paddle_x = 20.0
    cols = _paddle_columns(render_arena(game.view(), egocentric=False))
    assert abs((cols[0] + cols[-1]) / 2 - WIDTH / 2) > 50
