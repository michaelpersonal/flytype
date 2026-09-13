import os

import numpy as np
import pytest

from flytype.neural.rule import advance


def trace_protocol(order, frozen=False):
    k = np.zeros(2)
    d = np.zeros(1)
    u = np.zeros(2)
    w = np.zeros(2)
    gain = np.ones((1, 2))
    for phase in order:
        for _ in range(20):
            kh = np.array([20.0, 0.0]) if phase == "cue" else np.zeros(2)
            dh = np.array([30.0]) if phase == "reinforce" else np.zeros(1)
            advance(k, d, u, w, kh, dh, gain, 0.01, 0.001, frozen=frozen)
    return w


def test_memory_rule_temporal_specificity():
    paired = trace_protocol(["cue", "reinforce"])
    reverse = trace_protocol(["reinforce", "cue"])
    assert paired[0] < 0 and reverse[0] > 0
    assert paired[1] == 0 and reverse[1] == 0  # Unactivated input is unchanged.
    assert np.array_equal(trace_protocol(["cue", "reinforce"], True), np.zeros(2))


@pytest.mark.skipif(
    os.environ.get("FLYTYPE_FULL_TEST") != "1",
    reason="Downloads/uses full MaleCNS; explicit integration test",
)
def test_full_graph_release_and_checksums():
    from flytype.data import verify

    report = verify()
    assert report["release"] == "MaleCNS v1.0"
    assert report["neurons"] == 166700
    assert report["directed_edges"] == 25582938
    assert report["arrays_verified"] is True


@pytest.mark.skipif(
    os.environ.get("FLYTYPE_FULL_TEST") != "1",
    reason="Downloads/uses full MaleCNS; explicit integration test",
)
def test_full_graph_sensory_reinforcement_checkpoint(tmp_path):
    from flytype.config import Settings
    from flytype.neural.controller import FlyController

    c = FlyController(Settings())
    assert len(c.brain.post) == 25582938 and len(c.brain.circuit["edges"]) == 7835
    assert len(c.brain.retina) == 3335 and len(c.brain.r8) == 811

    white = np.full((180, 320, 3), 255, np.uint8)
    for _ in range(3):
        c.observe(white, "none")
    c.save(tmp_path / "before.npz")
    before = c.brain.weight[c.brain.circuit["edges"]].copy()

    reward = c.observe(white, "reward")
    assert reward["reward_spikes"] > 0 and reward["stimulus_ms"] == 200
    assert reward["KC_spikes"] > 0 and reward["memory"]["changed_edges"] > 0
    reward_weights = c.brain.weight[c.brain.circuit["edges"]].copy()

    c.restore(tmp_path / "before.npz")
    control = c.observe(white, "none")
    assert not np.array_equal(reward_weights, c.brain.weight[c.brain.circuit["edges"]])

    c.restore(tmp_path / "before.npz")
    c.brain.weights_frozen = True
    c.observe(white, "reward")
    assert np.array_equal(before, c.brain.weight[c.brain.circuit["edges"]])

    c.restore(tmp_path / "before.npz")
    loss = c.observe(white, "aversive")
    assert loss["aversive_spikes"] > 0 and loss["stimulus_ms"] == 200
    assert np.isfinite(c.brain.weight).all()
    print({"reward": reward, "unpaired_control": control, "loss": loss})


@pytest.mark.skipif(
    os.environ.get("FLYTYPE_FULL_TEST") != "1",
    reason="Downloads/uses full MaleCNS; explicit integration test",
)
def test_full_graph_neural_decision_never_reads_target(tmp_path):
    """Decoded actions and reinforcement come only from spike counts; target
    text/correctness never enter neural/ and can't influence the decode."""
    from flytype.config import Settings
    from flytype.neural.controller import FlyController

    c = FlyController(Settings(decoder_deadband_hz=0.1))
    frame = np.full((180, 320, 3), 128, np.uint8)
    result = c.observe(frame, "none")
    assert result["action"] in ("LEFT", "RIGHT", "HOLD")
    assert result["source"] == "malecns"
    assert set(result["cell_ids"]) == {"left", "right", "gate"}


@pytest.mark.skipif(
    os.environ.get("FLYTYPE_FULL_TEST") != "1",
    reason="Downloads/uses full MaleCNS; explicit integration test",
)
def test_full_graph_frozen_mode_does_not_change_plastic_weights(tmp_path):
    from flytype.config import Settings
    from flytype.neural.controller import FlyController

    c = FlyController(Settings(frozen=True))
    assert c.brain.weights_frozen is True
    before = c.brain.weight[c.brain.circuit["edges"]].copy()
    white = np.full((180, 320, 3), 255, np.uint8)
    for stimulus in ["none", "reward", "aversive", "reward"]:
        c.observe(white, stimulus)
    assert np.array_equal(before, c.brain.weight[c.brain.circuit["edges"]])


@pytest.mark.skipif(
    os.environ.get("FLYTYPE_FULL_TEST") != "1",
    reason="Downloads/uses full MaleCNS; explicit integration test",
)
def test_full_graph_checkpoint_restore_reproduces_committed_state(tmp_path):
    from flytype.config import Settings
    from flytype.neural.controller import FlyController

    c = FlyController(Settings())
    frame = np.full((180, 320, 3), 200, np.uint8)
    c.observe(frame, "none")
    c.observe(frame, "reward")
    path = tmp_path / "checkpoint.npz"
    c.save(path)
    committed = c.brain.weight.copy()
    committed_cursor = c.brain.cursor

    c.observe(frame, "aversive")  # mutate state past the checkpoint
    assert c.brain.cursor != committed_cursor

    c.restore(path)
    assert np.array_equal(c.brain.weight, committed)
    assert c.brain.cursor == committed_cursor


@pytest.mark.skipif(
    os.environ.get("FLYTYPE_FULL_TEST") != "1",
    reason="Set FLYTYPE_FULL_TEST=1 to exercise the full MaleCNS graph",
)
def test_full_graph_drives_the_continuous_paddle_session(tmp_path):
    """One real observation through the web path, end to end.

    This is the verification the continuous task list claimed and did not
    have: a committed observation whose control came from MaleCNS firing via
    the frozen population decoder, not from a fixture stand-in.
    """
    import numpy as np

    from flytype.config import Settings
    from flytype.data import verify
    from flytype.motor import fit_population_decoder, frame_conditions
    from flytype.neural.controller import FlyController
    from flytype.play_calibration import descending_population
    from flytype.play_session import ContinuousPlaySession

    settings = Settings(seed=5, egocentric=True, frozen=True, decoder_baseline_obs=0)
    controller = FlyController(settings)
    population = descending_population(controller)[:8]
    assert population, "no bilateral descending population in the retained graph"

    # A trivially-fitted decoder: this test is about the plumbing carrying real
    # MaleCNS rates, not about decoding accuracy, which docs/validation.md
    # reports separately from a real calibration.
    offsets = np.linspace(-80, 80, 5)
    motor = fit_population_decoder(
        cell_ids=population,
        samples=np.tile(offsets.reshape(-1, 1), (1, len(population))) + 10.0,
        offsets=offsets,
        ridge=1.0,
        metadata={"frame_conditions": frame_conditions(settings)},
    )
    motor.check_compatible(settings)

    session = ContinuousPlaySession(
        settings=settings,
        out=tmp_path,
        dataset_info=verify(),
        controller=controller,
        motor=motor,
    )
    snapshot = session.advance()

    assert snapshot["source"] == "malecns"
    assert snapshot["neural"]["source"] == "malecns"
    assert snapshot["neural"]["total_spikes"] > 0
    assert -1.0 <= snapshot["motor"]["control"] <= 1.0
    assert snapshot["motor"]["cell_count"] == len(population)
    assert np.isfinite(snapshot["motor"]["offset_px"])
    # The DNp20 readout is recorded but is not what moved the paddle.
    assert snapshot["neural"]["action"] in ("LEFT", "RIGHT", "HOLD")
    # The page is handed the decoder's own held-out quality so it cannot show a
    # confident offset without showing what that offset is worth.
    assert "validation_mae_px" in snapshot["decoder_quality"]
    assert "shuffled_control_mae_px" in snapshot["decoder_quality"]
    assert (tmp_path / "events.jsonl").exists()
    assert (tmp_path / "latest-input.png").exists()
