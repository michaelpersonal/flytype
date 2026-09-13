"""Delays reinforcement by exactly one observation; never edits weights directly.

Correctness of observation N may choose only the stimulus scheduled for
observation N+1 (reward/aversive/none). This module never touches synaptic
weights: it hands a stimulus label to neural/controller.py, which stimulates
the identified PAM11/PPL101 cells during the next observation.
"""

from .task import restore_random_state

OUTCOME_TO_STIMULUS = {"correct": "reward", "incorrect": "aversive", "hold": "none"}


class FeedbackSchedule:
    def __init__(self, shuffle, rng):
        if shuffle and rng is None:
            raise ValueError("shuffle-feedback requires an RNG")
        self.shuffle = shuffle
        self.rng = rng
        self.pending = "none"

    def advance(self, outcome):
        """Return (deliver_now, scheduled_for_next) for one observation."""
        if outcome not in OUTCOME_TO_STIMULUS:
            raise ValueError("Unknown outcome")
        deliver = self.pending
        natural = OUTCOME_TO_STIMULUS[outcome]
        if self.shuffle and natural != "none":
            scheduled = self.rng.choice(["reward", "aversive"])
        else:
            scheduled = natural
        self.pending = scheduled
        return deliver, scheduled

    def state(self):
        return {
            "pending": self.pending,
            "shuffle": self.shuffle,
            "rng_state": list(self.rng.getstate()) if self.rng is not None else None,
        }

    def restore(self, state):
        if state["shuffle"] != self.shuffle:
            raise ValueError("Saved shuffle-feedback flag does not match this run")
        self.pending = state["pending"]
        if self.rng is not None and state["rng_state"] is not None:
            self.rng.setstate(restore_random_state(state["rng_state"]))
