import json
import random
import subprocess
import sys

import numpy as np
import pytest

from flytype.arena import render_arena
from flytype.breakout import (
    BALL_D,
    PADDLE_HEIGHT,
    PADDLE_TOP,
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


def test_fractional_control_sets_direction_and_distance():
    game = make_game()
    start = game.paddle_x
    game.step_control(0.5)
    assert game.paddle_x == start + game.paddle_speed * 0.5
    assert game.view()["paddle_control"] == 0.5
    game.step_control(-0.25)
    assert game.paddle_x == start + game.paddle_speed * 0.25
    assert game.view()["paddle_control"] == -0.25


def test_fractional_control_rejects_nonfinite_or_out_of_range_values():
    game = make_game()
    for control in (-1.01, 1.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            game.step_control(control)


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
    """Compared on survival and clearing, not raw hit count: a tracker ends the
    episode early by clearing the wall, which caps the hits it can accumulate."""
    cleared = blind_cleared = 0
    for seed in range(8):
        perfect = make_game(seed=seed)
        while not perfect.done and perfect.tick < 3000:
            perfect.step(perfect.tracking_direction() or "HOLD")
        assert perfect.summary()["tracking_rate"] == 1.0
        assert perfect.misses == 0, "a perfect tracker should never drop a ball"
        cleared += perfect.cleared

        blind = make_game(seed=seed)
        rng = random.Random(seed)
        while not blind.done and blind.tick < 3000:
            blind.step(rng.choice(["LEFT", "RIGHT", "HOLD"]))
        blind_cleared += blind.cleared
        assert blind.misses == 3, "random play should lose every ball"
    assert cleared == 8 and blind_cleared == 0


def test_ball_stays_inside_the_mapped_photoreceptor_band():
    game = make_game(seed=5)
    rng = random.Random(7)
    while not game.done and game.tick < 800:
        assert -1 <= game.ball_x <= WIDTH - BALL_D + 1
        assert FIELD_TOP - 1 <= game.ball_y <= FIELD_BOTTOM + 1
        game.step(rng.choice(["LEFT", "RIGHT", "HOLD"]))


def test_losing_every_life_ends_the_episode():
    game = make_game(seed=2, lives=1)
    while not game.done and game.tick < 2000:
        game.step("HOLD")
    assert game.done and game.lives == 0 and game.misses == 1
    with pytest.raises(RuntimeError):
        game.step("HOLD")


def test_ball_corner_must_visibly_touch_paddle_before_bouncing():
    game = make_game()
    game.ball_vy = abs(game.ball_vy_magnitude)
    # The square bounds overlap by one pixel in both axes, but the circle is
    # still diagonally clear of the paddle's top-left corner.
    game.ball_x = game.paddle_x - BALL_D + 1
    game.ball_y = PADDLE_TOP - BALL_D + 1
    before_vy = game.ball_vy
    assert not game._hit_paddle()
    assert game.ball_vy == before_vy

    # Direct tangency with the top face is visible contact and must bounce.
    game.ball_x = game.paddle_x
    game.ball_y = PADDLE_TOP - BALL_D
    assert game._hit_paddle()
    assert game.ball_vy < 0


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
    band = slice(PADDLE_TOP, PADDLE_TOP + PADDLE_HEIGHT)
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
    band = frame[PADDLE_TOP:PADDLE_TOP + PADDLE_HEIGHT]
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
        game.ball_x = paddle_x + game.paddle_width / 2 - BALL_D / 2 + offset
        game.ball_y = float(game.wall_bottom + 4)
        frame = render_arena(game.view(), egocentric=True)
        band = frame[int(game.ball_y):int(game.ball_y) + BALL_D]
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


def test_the_website_pace_stays_winnable_by_tracking_alone():
    """The site runs at twice the legacy pace so it is watchable at ~2 s/tick.

    Speeding the ball up is only legitimate while the paddle can still reach it:
    the descent is halved and the paddle step doubled, so a tracker covers the
    same field per descent. A decoder that tracks must still clear; one that
    cannot must still lose.
    """
    fast = dict(ball_descent_ticks=6, paddle_speed=52, ball_speed=9.0)
    cleared = blind_cleared = dropped = 0
    for seed in range(8):
        perfect = make_game(seed=seed, **fast)
        while not perfect.done and perfect.tick < 3000:
            perfect.step(perfect.tracking_direction() or "HOLD")
        cleared += perfect.cleared
        dropped += perfect.misses

        blind = make_game(seed=seed, **fast)
        rng = random.Random(seed)
        while not blind.done and blind.tick < 3000:
            blind.step(rng.choice(["LEFT", "RIGHT", "HOLD"]))
        blind_cleared += blind.cleared
    assert cleared == 8 and dropped == 0, "a tracker must still win at this pace"
    assert blind_cleared == 0, "a blind agent must still lose at this pace"
