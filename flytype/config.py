"""Run configuration. No account, credential or market fields exist here."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

DEFAULT_TARGET = "follow @michaelzsguo on x"
DEFAULT_ALPHABET = "abcdefghijklmnopqrstuvwxyz @"
# Fixture mode has no neural compute to pace a human-watchable demo; this fixed
# pause (skipped by --fast) is cosmetic wall-clock pacing, never neural time.
FIXTURE_WALL_DELAY_SECONDS = 0.2


@dataclass(frozen=True)
class Settings:
    target: str = DEFAULT_TARGET
    alphabet: str = DEFAULT_ALPHABET
    seed: int = 0
    steps: int = 0
    max_attempts_per_character: int = 50
    keyboard: str = "pair"
    neural_ms: float = 500
    neural_bin_ms: float = 10
    pulse_ms: float = 200
    pulse_current: float = 20
    decoder_deadband_hz: float = 0.1
    # Running-median window for the decoder's decision threshold. 0 disables it
    # and compares the raw right-minus-left rate against the deadband, which is
    # what every typing run used. See neural/controller.py for what the
    # threshold may see (this decoder's own past output, and nothing else) and
    # why the window length decides whether the paddle can cross the field.
    decoder_baseline_obs: int = 0
    paddle_width: int = 56
    paddle_speed: int = 26
    ball_speed: float = 5.0
    ball_descent_ticks: int = 12
    brick_rows: int = 2
    brick_columns: int = 7
    lives: int = 3
    # Scroll the playfield so the paddle sits at the centre of the frame.
    # See arena.py for why a fixed view puts zero side information in a
    # mean-rate readout.
    egocentric: bool = False
    # Deliver the scheduled PAM11/PPL101 pulse at all. Reinforcement is the
    # point of the typing task, but it is inert for a task driven by a frozen
    # decoder fitted offline: no reward can change a control the network does
    # not learn. Turning it off makes a run match the regime its decoder was
    # calibrated in, which is the reason to do it. It does not make the run
    # faster: measured back to back after settling, frozen/plastic and
    # reinforced/unreinforced all cost 2.00-2.03 s per observation at ~431,000
    # spikes.
    # Compress the per-observation brain checkpoint. Default on, which is what
    # every experiment so far used. Off costs ~126 MB per checkpoint instead of
    # ~5 MB and saves ~700 ms per observation, which matters only for a live
    # session where the checkpoint would otherwise dominate the loop.
    compress_checkpoints: bool = True
    reinforce: bool = True
    fixture: bool = False
    frozen: bool = False
    shuffle_feedback: bool = False
    fast: bool = False

    @property
    def learning(self):
        return not self.frozen

    def __post_init__(self):
        if not isinstance(self.target, str) or not self.target:
            raise ValueError("Target text must be a nonempty string")
        alphabet = set(self.alphabet)
        if len(alphabet) < 2:
            raise ValueError("Alphabet needs at least two distinct characters")
        missing = sorted(set(self.target) - alphabet)
        if missing:
            raise ValueError(f"Target characters not in alphabet: {missing!r}")
        if type(self.seed) is not int:
            raise ValueError("Seed must be an integer")
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if (
            type(self.max_attempts_per_character) is not int
            or self.max_attempts_per_character < 1
        ):
            raise ValueError("max_attempts_per_character must be a positive integer")
        for x in [
            self.neural_ms,
            self.neural_bin_ms,
            self.pulse_ms,
            self.pulse_current,
            self.decoder_deadband_hz,
        ]:
            if not math.isfinite(x) or x <= 0:
                raise ValueError("Positive finite parameter required")
        if self.neural_bin_ms > 10 or self.pulse_ms > self.neural_ms:
            raise ValueError(
                "Use <=10 ms neural bins; pulse must fit a decision window"
            )
        if any(
            abs(x * 10 - round(x * 10)) > 1e-7
            for x in [self.neural_ms, self.neural_bin_ms, self.pulse_ms]
        ):
            raise ValueError("Neural intervals must be multiples of 0.1 ms")
        if type(self.decoder_baseline_obs) is not int or self.decoder_baseline_obs < 0:
            raise ValueError("decoder_baseline_obs must be a nonnegative integer")
        for name in ["paddle_width", "paddle_speed", "ball_descent_ticks",
                     "brick_rows", "brick_columns", "lives"]:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.ball_speed) or self.ball_speed <= 0:
            raise ValueError("ball_speed must be positive and finite")
        if self.paddle_width > 320 or self.brick_columns > 40:
            raise ValueError("Arena geometry does not fit the 320x180 field")
        if self.keyboard not in ("pair", "tree"):
            raise ValueError("keyboard must be 'pair' (one distractor) or 'tree' (full alphabet)")
        if self.fast and not self.fixture:
            raise ValueError("--fast is permitted only in --fixture mode")

    def signature(self):
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()
