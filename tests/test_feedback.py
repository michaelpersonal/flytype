import random

import pytest

from flytype.feedback import FeedbackSchedule


def test_reinforcement_delivered_one_observation_after_the_outcome():
    fb = FeedbackSchedule(shuffle=False, rng=None)
    deliver, scheduled = fb.advance("correct")
    assert deliver == "none"  # nothing was pending before the first outcome
    assert scheduled == "reward"
    deliver, scheduled = fb.advance("hold")
    assert deliver == "reward"  # last observation's outcome, delivered now
    assert scheduled == "none"
    deliver, scheduled = fb.advance("incorrect")
    assert deliver == "none"
    assert scheduled == "aversive"
    deliver, scheduled = fb.advance("hold")
    assert deliver == "aversive"
    assert scheduled == "none"


def test_hold_never_schedules_a_stimulus():
    fb = FeedbackSchedule(shuffle=False, rng=None)
    _, scheduled = fb.advance("hold")
    assert scheduled == "none"


def test_shuffle_feedback_keeps_timing_but_can_change_identity():
    rng = random.Random(0)
    fb = FeedbackSchedule(shuffle=True, rng=rng)
    outcomes = ["correct"] * 40
    scheduled = [fb.advance(o)[1] for o in outcomes]
    assert set(scheduled) <= {"reward", "aversive"}
    assert len(set(scheduled)) == 2  # both labels appear under shuffling


def test_shuffle_feedback_still_delivers_none_after_hold():
    rng = random.Random(0)
    fb = FeedbackSchedule(shuffle=True, rng=rng)
    fb.advance("correct")
    _, scheduled = fb.advance("hold")
    assert scheduled == "none"


def test_shuffle_requires_rng():
    with pytest.raises(ValueError):
        FeedbackSchedule(shuffle=True, rng=None)


def test_state_round_trip():
    rng = random.Random(1)
    fb = FeedbackSchedule(shuffle=True, rng=rng)
    fb.advance("correct")
    fb.advance("incorrect")
    saved = fb.state()

    restored = FeedbackSchedule(shuffle=True, rng=random.Random(0))
    restored.restore(saved)
    assert restored.pending == fb.pending
    assert [restored.advance("correct")[1] for _ in range(5)] == [
        fb.advance("correct")[1] for _ in range(5)
    ]


def test_restore_rejects_mismatched_shuffle_flag():
    fb = FeedbackSchedule(shuffle=False, rng=None)
    saved = fb.state()
    other = FeedbackSchedule(shuffle=True, rng=random.Random(0))
    with pytest.raises(ValueError):
        other.restore(saved)
