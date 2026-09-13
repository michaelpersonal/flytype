"""Target text progress, candidate generation and correctness scoring.

Nothing here reads neural state, and neural code never reads this module's
target text or correctness: the two only meet in cli.py, which passes a
decoded action in and a plain correct/incorrect/hold outcome out.
"""

import random


def restore_random_state(serialized):
    """Undo JSON's tuple->list coercion so random.Random.setstate() accepts it."""

    def convert(value):
        if isinstance(value, list):
            return tuple(convert(item) for item in value)
        return value

    return convert(serialized)


class Task:
    def __init__(self, target, alphabet, rng):
        if not isinstance(target, str) or not target:
            raise ValueError("Target text must be a nonempty string")
        missing = sorted(set(target) - set(alphabet))
        if missing:
            raise ValueError(f"Target characters not in alphabet: {missing!r}")
        self.target = target
        self.alphabet = alphabet
        self.rng = rng
        self.index = 0
        self.attempts = 0
        self.pending = None

    @property
    def typed(self):
        return self.target[: self.index]

    @property
    def done(self):
        return self.index >= len(self.target)

    def candidates(self):
        """Return the current left/right candidates, drawing a fresh pair only
        when none is pending (a HOLD leaves the same pair in place)."""
        if self.done:
            raise RuntimeError("Target already complete")
        if self.pending is None:
            self.pending = self._draw()
        return self.pending

    def _draw(self):
        correct = self.target[self.index]
        choices = [c for c in self.alphabet if c != correct]
        distractor = self.rng.choice(choices)
        if self.rng.random() < 0.5:
            left, right = correct, distractor
        else:
            left, right = distractor, correct
        correct_side = "LEFT" if left == correct else "RIGHT"
        return {
            "left": left,
            "right": right,
            "correct_side": correct_side,
            "correct_char": correct,
        }

    def score(self, action):
        """Advance state from a decoded LEFT/RIGHT/HOLD action.

        Returns one of "hold", "correct", "incorrect". A HOLD never counts as
        an attempt and never changes the candidate pair.
        """
        if action not in ("LEFT", "RIGHT", "HOLD"):
            raise ValueError("Unknown action")
        candidates = self.candidates()
        if action == "HOLD":
            return "hold"
        correct = action == candidates["correct_side"]
        if correct:
            self.index += 1
            self.attempts = 0
        else:
            self.attempts += 1
        self.pending = None
        return "correct" if correct else "incorrect"

    def state(self):
        return {
            "target": self.target,
            "alphabet": self.alphabet,
            "index": self.index,
            "attempts": self.attempts,
            "pending": self.pending,
            "rng_state": list(self.rng.getstate()),
        }

    def restore(self, state):
        if state["target"] != self.target or state["alphabet"] != self.alphabet:
            raise ValueError("Saved task target/alphabet does not match this run")
        self.index = state["index"]
        self.attempts = state["attempts"]
        self.pending = state["pending"]
        self.rng.setstate(restore_random_state(state["rng_state"]))


class TreeTask:
    """Full-alphabet spelling over the same two-choice neural readout.

    The decoder still resolves exactly one axis (LEFT / RIGHT / HOLD). To reach
    an alphabet of N symbols, each character is selected by a descent through a
    binary tree: the remaining candidates are split in half, the network picks a
    side, and the surviving half becomes the new candidate set. A character is
    committed only at a leaf, after ceil(log2(N)) consecutive correct choices.

    A choice whose half does not contain the needed character simply re-presents
    the same split (with the halves re-randomised across the keys), so a mistake
    costs one decision and no progress. Nothing is ever appended except the
    symbol left at a leaf, so a wrong turn costs attempts, never a wrong
    character.
    """

    def __init__(self, target, alphabet, rng):
        if not isinstance(target, str) or not target:
            raise ValueError("Target text must be a nonempty string")
        symbols = sorted(set(alphabet))
        if len(symbols) < 2:
            raise ValueError("Alphabet needs at least two distinct characters")
        missing = sorted(set(target) - set(symbols))
        if missing:
            raise ValueError(f"Target characters not in alphabet: {missing!r}")
        self.target = target
        self.alphabet = alphabet
        self.symbols = symbols
        self.rng = rng
        self.index = 0
        self.attempts = 0
        self.path = [list(symbols)]     # path[-1] is the live candidate set
        self.pending = None
        self.committed = False

    @property
    def typed(self):
        return self.target[: self.index]

    @property
    def done(self):
        return self.index >= len(self.target)

    @property
    def depth_required(self):
        n, d = len(self.symbols), 0
        while n > 1:
            n = (n + 1) // 2
            d += 1
        return d

    @property
    def remaining(self):
        return self.path[-1]

    @property
    def depth(self):
        return len(self.path) - 1

    def _restart_descent(self):
        self.path = [list(self.symbols)]
        self.pending = None

    def candidates_view(self):
        """Current left/right symbol groups; a HOLD leaves them unchanged."""
        if self.done:
            raise RuntimeError("Target already complete")
        if self.pending is None:
            half = (len(self.remaining) + 1) // 2
            a, b = self.remaining[:half], self.remaining[half:]
            if self.rng.random() < 0.5:
                left, right = a, b
            else:
                left, right = b, a
            correct = self.target[self.index]
            self.pending = {
                "left": left,
                "right": right,
                "correct_side": "LEFT" if correct in left else "RIGHT",
                "correct_char": correct,
                "depth": self.depth,
                "depth_required": self.depth_required,
            }
        return self.pending

    # the CLI calls .candidates() on either task type
    def candidates(self):
        return self.candidates_view()

    def score(self, action):
        if action not in ("LEFT", "RIGHT", "HOLD"):
            raise ValueError("Unknown action")
        view = self.candidates_view()
        self.committed = False
        if action == "HOLD":
            return "hold"
        chosen = view["left"] if action == "LEFT" else view["right"]
        if view["correct_char"] in chosen:
            self.path.append(list(chosen))
            self.pending = None
            if len(self.path[-1]) == 1:
                self.index += 1
                self.attempts = 0
                self.committed = True
                self._restart_descent()
            return "correct"
        # wrong half: retry this same node; the split is redrawn with the
        # halves re-randomised between the keys, so no progress is lost
        self.attempts += 1
        self.pending = None
        return "incorrect"

    def state(self):
        return {
            "target": self.target,
            "alphabet": self.alphabet,
            "index": self.index,
            "attempts": self.attempts,
            "path": [list(x) for x in self.path],
            "pending": self.pending,
            "rng_state": list(self.rng.getstate()),
        }

    def restore(self, state):
        if state["target"] != self.target or state["alphabet"] != self.alphabet:
            raise ValueError("Saved task target/alphabet does not match this run")
        self.index = state["index"]
        self.attempts = state["attempts"]
        self.path = [list(x) for x in state["path"]]
        self.pending = state["pending"]
        self.rng.setstate(restore_random_state(state["rng_state"]))


def new_rng(seed):
    return random.Random(seed)
