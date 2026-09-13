import numpy as np
import pandas as pd
import pytest

from flytype.decoder import FixtureDecoder
from flytype.neural.controller import Decoder


def make_decoder(deadband=2.0):
    a = pd.DataFrame(
        {"type": ["DNp20", "DNp20", "DNpe017"], "somaSide": ["L", "R", "L"]}
    )
    return Decoder(np.array([1, 2, 3]), a, deadband)


def test_neural_decoder_depends_only_on_counts_and_deadband():
    d = make_decoder(deadband=2.0)
    assert d.decode(np.array([0, 10, 0]), 0.5)["action"] == "RIGHT"
    assert d.decode(np.array([10, 0, 0]), 0.5)["action"] == "LEFT"
    assert d.decode(np.array([4, 4, 0]), 0.5)["action"] == "HOLD"
    # Gate spikes are logged, never required for a decision.
    assert d.decode(np.array([0, 10, 0]), 0.5)["gate_spikes"] == 0
    assert d.decode(np.array([0, 10, 5]), 0.5)["gate_spikes"] == 5
    assert d.decode(np.array([0, 10, 5]), 0.5)["action"] == "RIGHT"


def test_neural_decoder_deadband_is_configurable():
    counts = np.array([0, 1, 0])  # 2 Hz difference over 0.5 s
    assert make_decoder(deadband=2.5).decode(counts, 0.5)["action"] == "HOLD"
    assert make_decoder(deadband=1.0).decode(counts, 0.5)["action"] == "RIGHT"


def test_neural_decoder_every_event_tagged_malecns():
    d = make_decoder()
    assert d.decode(np.array([0, 0, 0]), 0.5)["source"] == "malecns"


def test_fixture_decoder_only_available_in_fixture_mode_is_tagged():
    d = FixtureDecoder(seed=1, hold_probability=0, error_probability=0)
    result = d.decode("LEFT")
    assert result["action"] == "LEFT"
    assert result["source"] == "fixture"


def test_fixture_decoder_deterministic_for_a_seed():
    a = FixtureDecoder(seed=7)
    b = FixtureDecoder(seed=7)
    actions_a = [a.decode("LEFT")["action"] for _ in range(50)]
    actions_b = [b.decode("LEFT")["action"] for _ in range(50)]
    assert actions_a == actions_b


def test_fixture_decoder_state_round_trip():
    a = FixtureDecoder(seed=3)
    for _ in range(10):
        a.decode("RIGHT")
    saved = a.state()
    b = FixtureDecoder(seed=0)
    b.restore(saved)
    assert [a.decode("LEFT")["action"] for _ in range(20)] == [
        b.decode("LEFT")["action"] for _ in range(20)
    ]


def test_fixture_decoder_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        FixtureDecoder(seed=0, hold_probability=0.6, error_probability=0.6)
    with pytest.raises(ValueError):
        FixtureDecoder(seed=0, hold_probability=-0.1)


def test_median_baseline_splits_a_biased_readout_evenly():
    """Centering removes a standing offset without knowing the right answer.

    Uses the DNp20 differences recorded in a real typing run: one cell leads on
    39 of 52 observations, and an uncentered comparison spends nearly every
    decision on that offset.
    """
    import numpy as np

    biased = [12.0, 8.0, 10.0, 12.0, 14.0, 10.0, 12.0, 6.0, 12.0, 16.0] * 6
    raw = ["HOLD" if abs(x) < 0.1 else "RIGHT" if x > 0 else "LEFT" for x in biased]
    assert set(raw) == {"RIGHT"}, "raw readout is entirely one-sided"

    history, centered = [], []
    for x in biased:
        base = float(np.median(history)) if history else 0.0
        c = x - base
        centered.append("HOLD" if abs(c) < 0.1 else "RIGHT" if c > 0 else "LEFT")
        history.append(x)
        del history[:-16]
    left = centered.count("LEFT")
    right = centered.count("RIGHT")
    assert abs(left - right) <= max(3, 0.25 * (left + right)), (left, right)
