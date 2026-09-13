"""Only RGB and engineered reinforcement enter the network.

Target text, typed progress and correctness never enter here or override the
decoded action; see decoder.py and task.py for where those live instead.
"""

import hashlib

import numpy as np

from .common import annotations
from .visual import VisualMemoryBrain


def balanced_threshold(values):
    """The cut that splits these values as evenly as possible, and lands on none.

    A plain median is the obvious choice and the wrong one here. One DNp20 cell
    per side over a 500 ms window makes the right-minus-left rate a small set of
    even integers, so the median lands *on* a heavily populated value: those
    observations become exact ties, and the mass left above and below the cut is
    not equal. Measured on a live arena run that produced 18 LEFT against 43
    RIGHT with 34 ties -- a bias large enough to drive the paddle into a wall and
    hold it there.

    Choosing among the midpoints between adjacent distinct values instead means
    the cut never coincides with an observation, so there are no ties, and the
    midpoint that minimises |above - below| balances the two directions by
    construction. Computed from this decoder's own past output and nothing else.
    """
    unique = sorted(set(values))
    if len(unique) < 2:
        return (unique[0] if unique else 0.0) - 1e-9
    array = np.asarray(values, dtype=float)
    best, best_gap = None, None
    for lo, hi in zip(unique, unique[1:]):
        cut = (lo + hi) / 2
        gap = abs(int((array > cut).sum()) - int((array < cut).sum()))
        if best_gap is None or gap < best_gap:
            best, best_gap = cut, gap
    return float(best)


class Decoder:
    """Fixed DNp20 mean-rate decoder: RIGHT/LEFT/HOLD from spike counts only.

    DNpe017 spikes are recorded for comparison but never gate the decision;
    enabling a gate would be a new experiment and must not happen silently
    (see docs/model.md).

    Optional baseline centering (baseline_obs > 0) compares the right-minus-left
    rate against the running median of its own last baseline_obs values instead
    of against zero. The two DNp20 cells do not sit at equal rates -- in the
    typing runs the right cell led on 39 of 52 observations -- so an uncentered
    comparison spends nearly every decision on that standing offset. A single
    cell per side firing over 500 ms also quantises the difference to a handful
    of even values, which is why the threshold is a median rather than a mean:
    the mean of a lopsided discrete distribution still lands off the mode and
    leaves a residual bias.

    The window length is a real trade-off, not a free parameter, and it is the
    difference between a paddle that can cross the field and one that cannot. A
    short window forces the recent decisions to be balanced by construction:
    measured on a real 441-observation arena run, a 16-observation window
    produced exactly 211 LEFT and 211 RIGHT and never permitted a run longer
    than three steps in one direction, confining the paddle to a 104 px band. A
    frozen threshold permits runs of nine but lets the offset creep back and
    pins the paddle to a wall. On that same series a 64-observation window is
    the compromise that holds both ends: 198 LEFT against 208 RIGHT, and a
    paddle that reaches both walls.

    What this can and cannot do: it removes a constant offset, so the readout is
    free to respond to the image. It cannot manufacture a correct answer,
    because it is computed only from this decoder's own past output and never
    sees the target, the arena, the outcome or which direction is right. If the
    image does not modulate these two cells, centering yields a coin flip and
    the measured tracking rate lands on 0.5. Both the raw and the centered rate
    are recorded on every observation.
    """

    def __init__(self, ids, annotation, deadband_hz, baseline_obs=0):
        types = annotation.type.fillna("")
        sides = annotation.somaSide.fillna("")
        self.left = np.flatnonzero(types.eq("DNp20") & sides.eq("L"))
        self.right = np.flatnonzero(types.eq("DNp20") & sides.eq("R"))
        self.gate = np.flatnonzero(types.eq("DNpe017"))
        if not len(self.left) or not len(self.right):
            raise RuntimeError("Missing annotated DNp20 outputs")
        self.deadband = deadband_hz
        self.baseline_obs = int(baseline_obs)
        self.history = []
        self.baseline = 0.0
        self.calibrated = False
        self.identities = {
            k: [str(ids[i]) for i in getattr(self, k)]
            for k in ["left", "right", "gate"]
        }

    def decode(self, counts, seconds):
        # Mean rates prevent side population size from creating a built-in bias.
        left = float(np.mean(counts[self.left]) / seconds)
        right = float(np.mean(counts[self.right]) / seconds)
        difference = right - left
        baseline = self.baseline if self.baseline_obs else 0.0
        centered = difference - baseline
        gate = int(counts[self.gate].sum()) if len(self.gate) else 0
        action = (
            "HOLD"
            if abs(centered) < self.deadband
            else "RIGHT"
            if centered > 0
            else "LEFT"
        )
        if self.baseline_obs and not self.calibrated:
            self.history.append(difference)
            del self.history[: -self.baseline_obs]
            self.baseline = balanced_threshold(self.history)
        return {
            "action": action,
            "left_hz": left,
            "right_hz": right,
            "difference_hz": difference,
            "baseline_hz": baseline,
            "centered_hz": centered,
            "gate_spikes": gate,
            "cell_ids": self.identities,
            "source": "malecns",
        }

    def calibrate(self, differences):
        """Fix the decision threshold from a neutral-stimulus calibration block.

        `differences` are this decoder's own outputs while it was shown a
        stimulus in which neither direction is the answer -- the ball sitting
        directly on the paddle. The threshold is the balanced cut through them,
        so the standing offset between the two cells is removed while leaving
        any ball-driven deviation during play intact.

        This is the piece that a threshold fitted to live task data cannot do.
        The offset between these two cells is around 10 Hz; the ball moves them
        by 1 to 7 Hz. A trailing window fitted during play subtracts both, so a
        ball that genuinely sits to one side for a stretch -- exactly when the
        paddle most needs to travel -- is read as bias and cancelled.
        """
        if not differences:
            raise ValueError("Calibration needs at least one observation")
        self.history = list(differences)
        self.baseline = balanced_threshold(self.history)
        self.calibrated = True
        return self.baseline

    def state(self):
        return {
            "baseline": self.baseline,
            "history": list(self.history),
            "baseline_obs": self.baseline_obs,
            "calibrated": self.calibrated,
        }

    def restore(self, state):
        if state["baseline_obs"] != self.baseline_obs:
            raise ValueError("Saved decoder baseline window does not match this run")
        self.baseline = state["baseline"]
        self.history = list(state["history"])
        self.calibrated = bool(state.get("calibrated"))


SPIKE_BUCKETS = 512


class FlyController:
    def __init__(self, settings):
        self.s = settings
        self.brain = VisualMemoryBrain()
        self.brain.weights_frozen = not settings.learning
        annotation = annotations(self.brain.ids)
        self.decoder = Decoder(
            self.brain.ids,
            annotation,
            settings.decoder_deadband_hz,
            settings.decoder_baseline_obs,
        )
        self.id_to_index = {str(cell_id): i for i, cell_id in enumerate(self.brain.ids)}
        self.buckets, self.bucket_regions = self._assign_buckets(annotation)

    def _assign_buckets(self, annotation):
        """Group every neuron into a fixed bucket, tagged L / R / C by its side.

        Per-observation spike counts are reported per bucket so a reader can see
        *which* populations fired, not just how many spikes there were. The
        grouping is anatomical only in its left/right split; bucket membership
        within a side is by cell order, not by position.
        """
        side = annotation.rootSide.fillna("C").to_numpy()
        region = np.where(side == "L", "L", np.where(side == "R", "R", "C"))
        buckets = np.zeros(self.brain.n, dtype=np.int32)
        regions = []
        start = 0
        for tag in ("L", "R", "C"):
            idx = np.flatnonzero(region == tag)
            share = max(1, round(SPIKE_BUCKETS * len(idx) / self.brain.n))
            share = min(share, SPIKE_BUCKETS - start - (2 - "LRC".index(tag)))
            edges = np.linspace(0, len(idx), share + 1).astype(int)
            for b in range(share):
                buckets[idx[edges[b]:edges[b + 1]]] = start + b
            regions.extend([tag] * share)
            start += share
        return buckets, regions

    def observe(self, rgb, reinforcement, population_ids=None):
        if reinforcement not in ("none", "reward", "aversive"):
            raise ValueError("Unknown reinforcement")
        b = self.brain
        counts = np.zeros(b.n, dtype=np.int32)
        wall = 0.0
        remaining = round(self.s.neural_ms / b.dt)
        pulse = round(self.s.pulse_ms / b.dt) if reinforcement != "none" else 0
        delivered = 0
        while remaining:
            n = min(remaining, round(self.s.neural_bin_ms / b.dt))
            if pulse:
                n = min(n, pulse)
            stimulus = (
                (b.circuit[reinforcement], self.s.pulse_current) if pulse else None
            )
            c, elapsed = b.rgb_step(
                rgb, n * b.dt, learning=self.s.learning, stimulation=stimulus
            )
            counts += c
            wall += elapsed
            remaining -= n
            if pulse:
                delivered += n
                pulse -= n
        b.counts[:] = counts
        result = {
            **self.decoder.decode(counts, self.s.neural_ms / 1000),
            "brain_ms": b.sim_ms,
            "compute_seconds": wall,
            "stimulus": reinforcement,
            "stimulus_ms": delivered * b.dt,
            "reward_spikes": int(counts[b.circuit["reward"]].sum()),
            "aversive_spikes": int(counts[b.circuit["aversive"]].sum()),
            "KC_spikes": int(counts[b.circuit["kc"]].sum()),
            "total_spikes": int(counts.sum()),
            "active_neurons": int(np.count_nonzero(counts)),
            "spike_buckets": np.bincount(
                self.buckets, weights=counts, minlength=len(self.bucket_regions)
            ).astype(np.int64).tolist(),
            "spike_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
            "input_sha256": hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),
            "memory": b.memory(),
        }
        if population_ids is not None:
            try:
                indices = [self.id_to_index[str(cell_id)] for cell_id in population_ids]
            except KeyError as exc:
                raise ValueError(f"Population decoder cell is absent: {exc.args[0]}") from None
            seconds = self.s.neural_ms / 1000
            result["population_rates"] = {
                str(cell_id): float(counts[index] / seconds)
                for cell_id, index in zip(population_ids, indices)
            }
        return result

    def save(self, path):
        self.brain.checkpoint(path, compress=self.s.compress_checkpoints)

    def restore(self, path):
        self.brain.restore(path)
