"""Fixture-mode decoding and shared action constants.

The MaleCNS decoder (fixed DNp20 mean-rate mapping) lives in
neural/controller.py, next to the compiled circuit whose spike counts it
reads directly. This module holds the deterministic stand-in used only when
--fixture is set, so tests and short demos can exercise the full task/CLI
loop without the full connectome. Every action this module returns is
tagged "fixture"; the neural path always tags "malecns". Nothing here reads
target text or correctness beyond the one bit (which side is currently
correct) it needs to script a believable, reproducible neural substitute.
"""

import random

ACTIONS = ("LEFT", "RIGHT", "HOLD")


class FixtureDecoder:
    """Deterministic scripted decoder used only in --fixture mode.

    Mostly returns the correct side so a fixture run can complete a target
    sentence in a bounded number of observations, but seeded HOLD/incorrect
    draws exercise those code paths reproducibly too.
    """

    def __init__(self, seed, hold_probability=0.05, error_probability=0.10):
        if not 0 <= hold_probability <= 1 or not 0 <= error_probability <= 1:
            raise ValueError("Probabilities must be within [0, 1]")
        if hold_probability + error_probability > 1:
            raise ValueError("Combined hold/error probability exceeds 1")
        self.rng = random.Random(seed)
        self.hold_probability = hold_probability
        self.error_probability = error_probability

    def decode(self, correct_side):
        if correct_side not in ("LEFT", "RIGHT"):
            raise ValueError("correct_side must be LEFT or RIGHT")
        draw = self.rng.random()
        if draw < self.hold_probability:
            action = "HOLD"
        elif draw < self.hold_probability + self.error_probability:
            action = "RIGHT" if correct_side == "LEFT" else "LEFT"
        else:
            action = correct_side
        return {
            "action": action,
            "left_hz": None,
            "right_hz": None,
            "difference_hz": None,
            "gate_spikes": None,
            "cell_ids": None,
            "source": "fixture",
        }

    def state(self):
        return list(self.rng.getstate())

    def restore(self, state):
        from .task import restore_random_state

        self.rng.setstate(restore_random_state(state))
