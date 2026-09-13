"""Only RGB and engineered reinforcement enter the network.

Target text, typed progress and correctness never enter here or override the
decoded action; see decoder.py and task.py for where those live instead.
"""

import hashlib

import numpy as np

from .common import annotations
from .visual import VisualMemoryBrain


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
    of even values, which is why the median is used rather than the mean: the
    mean of a lopsided discrete distribution still lands off the mode and leaves
    a residual bias, while the median splits the recent decisions evenly by
    construction.

    What this can and cannot do: it removes the constant offset, so the readout
    is free to respond to the image. It cannot manufacture a correct answer,
    because it is computed only from this decoder's own past output and never
    sees the target, the arena, the outcome or which direction is right. If the
    image does not modulate these two cells, centering yields an even coin flip
    and the measured tracking rate lands on 0.5. Both the raw and the centered
    rate are recorded on every observation.
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
        if self.baseline_obs:
            self.history.append(difference)
            del self.history[: -self.baseline_obs]
            self.baseline = float(np.median(self.history))
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

    def state(self):
        return {
            "baseline": self.baseline,
            "history": list(self.history),
            "baseline_obs": self.baseline_obs,
        }

    def restore(self, state):
        if state["baseline_obs"] != self.baseline_obs:
            raise ValueError("Saved decoder baseline window does not match this run")
        self.baseline = state["baseline"]
        self.history = list(state["history"])


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

    def observe(self, rgb, reinforcement):
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
        return {
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

    def save(self, path):
        self.brain.checkpoint(path)

    def restore(self, path):
        self.brain.restore(path)
