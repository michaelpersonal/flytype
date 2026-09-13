import json

import numpy as np
import pytest

from flytype.config import Settings
from flytype.motor import (
    PopulationMotorDecoder,
    fit_population_decoder,
    frame_conditions,
)


def trained_decoder(settings=None):
    positions = np.asarray([-80, -40, 0, 40, 80], dtype=float)
    samples = np.column_stack((10 + positions / 10, np.ones(5) * 3))
    metadata = {"source": "test probes"}
    if settings is not None:
        metadata["frame_conditions"] = frame_conditions(settings)
    return fit_population_decoder(
        cell_ids=["directional", "flat"],
        samples=samples,
        offsets=positions,
        ridge=0.01,
        position_scale=40,
        motion_scale=40,
        metadata=metadata,
    )


def test_population_decoder_estimates_position_and_graded_control():
    decoder = trained_decoder()
    centre = decoder.decode({"directional": 10, "flat": 3})
    right = decoder.decode({"directional": 14, "flat": 3})
    farther = decoder.decode({"directional": 18, "flat": 3})
    assert abs(centre["offset_px"]) < 1
    assert right["control"] > 0
    assert farther["control"] > right["control"]
    assert farther["relative_motion_px"] > 0
    assert 0 < farther["control"] <= 1


def test_population_decoder_state_round_trips(tmp_path):
    decoder = trained_decoder()
    decoder.decode({"directional": 14, "flat": 3})
    artifact = tmp_path / "decoder.json"
    decoder.save(artifact)
    restored = PopulationMotorDecoder.load(artifact)
    restored.restore(decoder.state())
    assert restored.state() == decoder.state()
    assert json.loads(artifact.read_text())["version"] == PopulationMotorDecoder.VERSION
    assert restored.artifact_sha256 == decoder.artifact_sha256


def test_population_decoder_refuses_missing_or_invalid_rates():
    decoder = trained_decoder()
    with pytest.raises(ValueError):
        decoder.decode({"directional": 10})
    with pytest.raises(ValueError):
        decoder.decode({"directional": float("nan"), "flat": 3})


def test_forgetting_motion_drops_only_the_temporal_term():
    """A serve must not be read as the ball having moved across the field."""
    decoder = trained_decoder()
    decoder.decode({"directional": 2, "flat": 3})
    decoder.forget_motion()
    served = decoder.decode({"directional": 18, "flat": 3})
    assert served["relative_motion_px"] == 0.0
    assert served["offset_px"] > 0
    # Without the reset the same pair reports a large spurious motion.
    carried = trained_decoder()
    carried.decode({"directional": 2, "flat": 3})
    assert carried.decode({"directional": 18, "flat": 3})["relative_motion_px"] > 50


def test_decoder_refuses_frames_unlike_the_ones_it_was_fitted_on():
    calibrated = Settings(egocentric=True, paddle_width=56)
    decoder = trained_decoder(calibrated)
    decoder.check_compatible(calibrated)

    for mismatch in (
        Settings(egocentric=False, paddle_width=56),
        Settings(egocentric=True, paddle_width=80),
        Settings(egocentric=True, paddle_width=56, neural_ms=250),
    ):
        with pytest.raises(ValueError, match="different viewing conditions"):
            decoder.check_compatible(mismatch)


def test_decoder_without_recorded_conditions_is_refused():
    with pytest.raises(ValueError, match="predates frame-condition"):
        trained_decoder().check_compatible(Settings(egocentric=True))

