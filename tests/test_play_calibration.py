"""The calibration harness, exercised without loading the connectome.

A stand-in controller reads the ball out of the rendered frame itself, so
these tests also check that the probe frames really do encode the offset the
fit is told they encode.
"""

import numpy as np
import pytest

from flytype.config import Settings
from flytype.play_calibration import (
    calibrate_play_decoder,
    probe_conditions,
    reachable_heights,
    reachable_offsets,
)


class FrameReadingController:
    """Rates derived from the frame, with configurable signal and noise."""

    def __init__(self, cells=40, signal=True, noise=1.0, seed=0):
        self.cells = [f"cell-{i}" for i in range(cells)]
        self.signal = signal
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.observations = 0

    def _ball_offset(self, frame):
        """Recover the ball's offset from the paddle out of the pixels."""
        band = frame[:92]                       # above the paddle
        white = (band > 250).all(axis=2)
        columns = np.flatnonzero(white.any(axis=0))
        if not len(columns):
            return 0.0
        centre = (columns[0] + columns[-1]) / 2
        return (centre - frame.shape[1] / 2) * 2   # undo EGO_SCALE

    def observe(self, frame, reinforcement, population_ids=None):
        assert reinforcement == "none", "calibration must never reinforce"
        self.observations += 1
        result = {"compute_seconds": 0.001}
        if population_ids is None:
            return result
        offset = self._ball_offset(frame)
        rates = self.rng.normal(10.0, self.noise, size=len(self.cells))
        if self.signal:
            # Only the first three cells carry position.
            rates[:3] += offset / 20.0
        result["population_rates"] = dict(zip(self.cells, rates))
        return result


@pytest.fixture
def settings():
    return Settings(egocentric=True, frozen=True)


def run(tmp_path, controller, settings, **kwargs):
    import flytype.play_calibration as module

    original = module.descending_population
    module.descending_population = lambda _: list(controller.cells)
    try:
        return calibrate_play_decoder(
            settings,
            tmp_path / "decoder.json",
            dataset_info={"mode": "test"},
            controller=controller,
            offset_count=kwargs.pop("offset_count", 9),
            height_count=kwargs.pop("height_count", 3),
            repeats=kwargs.pop("repeats", 3),
            warmup=kwargs.pop("warmup", 4),
            feature_count=kwargs.pop("feature_count", 8),
            **kwargs,
        )
    finally:
        module.descending_population = original


def test_probe_grid_spans_the_reachable_arena(settings):
    low, high = reachable_offsets(settings)
    top, bottom = reachable_heights()
    conditions = probe_conditions(settings, 17, 5)
    assert len(conditions) == 85
    assert min(c["offset"] for c in conditions) == pytest.approx(low)
    assert max(c["offset"] for c in conditions) == pytest.approx(high)
    assert min(c["height"] for c in conditions) == pytest.approx(top)
    assert max(c["height"] for c in conditions) == pytest.approx(bottom)


def test_warmup_observations_are_discarded_not_recorded(tmp_path, settings):
    controller = FrameReadingController()
    decoder = run(tmp_path, controller, settings, warmup=6)
    meta = decoder.metadata
    assert meta["warmup_discarded"] == 6
    # 9 offsets x 3 heights x 3 repeats, plus the discarded warm-up.
    assert meta["conditions"] == 27
    assert meta["observations"] == 6 + 27 * 3 == controller.observations


def test_a_real_position_signal_beats_its_own_shuffled_control(tmp_path, settings):
    decoder = run(tmp_path, FrameReadingController(signal=True, noise=1.0), settings)
    meta = decoder.metadata
    assert meta["validation_correlation"] > 0.9
    assert meta["validation_mae_px"] < 60
    assert abs(meta["shuffled_control_correlation"] or 0.0) < 0.5
    assert meta["shuffled_control_mae_px"] > meta["validation_mae_px"]


def test_pure_noise_does_not_look_like_a_signal(tmp_path, settings):
    """The control exists to catch exactly this case."""
    decoder = run(tmp_path, FrameReadingController(signal=False, noise=1.0), settings)
    meta = decoder.metadata
    assert abs(meta["validation_correlation"] or 0.0) < 0.6
    assert meta["validation_mae_px"] > 80


def test_held_out_error_is_reported_per_ball_height(tmp_path, settings):
    decoder = run(tmp_path, FrameReadingController(), settings, height_count=3)
    bands = decoder.metadata["validation_by_ball_height"]
    assert bands and all("mae_px" in v for v in bands.values())


def test_calibration_records_the_conditions_it_was_fitted_under(tmp_path, settings):
    decoder = run(tmp_path, FrameReadingController(), settings)
    decoder.check_compatible(settings)
    with pytest.raises(ValueError, match="different viewing conditions"):
        decoder.check_compatible(Settings(egocentric=False, frozen=True))


def test_calibration_refuses_fixture_mode(tmp_path, settings):
    import dataclasses

    with pytest.raises(ValueError, match="requires MaleCNS"):
        calibrate_play_decoder(
            dataclasses.replace(settings, fixture=True),
            tmp_path / "d.json",
            dataset_info={},
        )
