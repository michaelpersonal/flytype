"""Frozen population decoder for graded paddle control.

Calibration may fit neural firing rates to labelled visual probe offsets.
Runtime decoding receives neural rates and its own previous estimate only; it
never reads live game geometry, correctness, score, outcome, or target text.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

# Everything that changes the pixels this decoder was fitted on. A frozen
# linear map from firing rates to pixels is only meaningful under the viewing
# conditions it was fitted under: rendering the same world fixed-view instead
# of egocentric, or widening the paddle, moves every photoreceptor onto
# different pixels while the decoder keeps reporting numbers as if nothing had
# changed. Recorded at calibration, checked at load.
FRAME_CONDITIONS = (
    "egocentric",
    "ego_scale",
    "paddle_width",
    "ball_diameter",
    "field_top",
    "field_bottom",
    "paddle_top",
    "neural_ms",
)


class PopulationMotorDecoder:
    VERSION = 2

    def __init__(
        self,
        *,
        cell_ids,
        means,
        scales,
        weights,
        intercept,
        position_scale,
        motion_scale,
        ridge,
        metadata=None,
    ):
        self.cell_ids = tuple(str(x) for x in cell_ids)
        self.means = np.asarray(means, dtype=float)
        self.scales = np.asarray(scales, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.intercept = float(intercept)
        self.position_scale = float(position_scale)
        self.motion_scale = float(motion_scale)
        self.ridge = float(ridge)
        self.metadata = dict(metadata or {})
        self.previous_offset = None
        self._validate()

    def _validate(self):
        n = len(self.cell_ids)
        if not n or len(set(self.cell_ids)) != n:
            raise ValueError("Population decoder needs unique cell IDs")
        if any(len(x) != n for x in (self.means, self.scales, self.weights)):
            raise ValueError("Population decoder vector lengths do not match")
        values = np.concatenate((
            self.means,
            self.scales,
            self.weights,
            np.asarray([
                self.intercept,
                self.position_scale,
                self.motion_scale,
                self.ridge,
            ]),
        ))
        if not np.isfinite(values).all():
            raise ValueError("Population decoder contains non-finite values")
        if (self.scales <= 0).any() or self.position_scale <= 0 or self.motion_scale <= 0:
            raise ValueError("Population decoder scales must be positive")
        if self.ridge < 0:
            raise ValueError("Population decoder ridge must be nonnegative")

    def decode(self, rates):
        offset = self.estimate(rates)
        motion = 0.0 if self.previous_offset is None else offset - self.previous_offset
        raw_control = offset / self.position_scale + motion / self.motion_scale
        control = math.tanh(raw_control)
        self.previous_offset = offset
        return {
            "offset_px": offset,
            "relative_motion_px": motion,
            "raw_control": raw_control,
            "control": control,
            "cell_count": len(self.cell_ids),
            "decoder_sha256": self.artifact_sha256,
        }

    def estimate(self, rates):
        """Estimate relative horizontal offset without changing temporal state."""
        try:
            values = np.asarray([rates[cell_id] for cell_id in self.cell_ids], dtype=float)
        except KeyError as exc:
            raise ValueError(f"Missing population rate for {exc.args[0]}") from None
        if not np.isfinite(values).all():
            raise ValueError("Population rates must be finite")
        return self.intercept + float(((values - self.means) / self.scales) @ self.weights)

    def check_compatible(self, settings):
        """Refuse to decode frames unlike the ones this decoder was fitted on."""
        recorded = self.metadata.get("frame_conditions")
        if not recorded:
            raise ValueError(
                "Population decoder predates frame-condition recording; "
                "rebuild it with `flytype calibrate-play-decoder`"
            )
        current = frame_conditions(settings)
        drift = {
            key: {"calibrated": recorded.get(key), "run": current[key]}
            for key in FRAME_CONDITIONS
            if recorded.get(key) != current[key]
        }
        if drift:
            raise ValueError(
                "Population decoder was calibrated under different viewing "
                f"conditions: {json.dumps(drift, sort_keys=True)}. Recalibrate "
                "it, or run with the settings it was calibrated under."
            )

    def forget_motion(self):
        """Drop the temporal term across a discontinuity in the world.

        The motion term is a difference between successive offset estimates,
        which only means anything while the ball is travelling. A serve
        teleports it anywhere across the field and a rollover replaces the
        game outright, so carrying the previous estimate across either one
        manufactures a motion of up to the full field width and saturates the
        control on the first observation of the new ball.
        """
        self.previous_offset = None

    def state(self):
        return {"previous_offset": self.previous_offset}

    def restore(self, state):
        value = state.get("previous_offset")
        if value is not None and not math.isfinite(float(value)):
            raise ValueError("Invalid prior population offset")
        self.previous_offset = None if value is None else float(value)

    def artifact(self):
        return {
            "version": self.VERSION,
            "cell_ids": list(self.cell_ids),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "weights": self.weights.tolist(),
            "intercept": self.intercept,
            "position_scale": self.position_scale,
            "motion_scale": self.motion_scale,
            "ridge": self.ridge,
            "metadata": self.metadata,
        }

    @property
    def artifact_sha256(self):
        payload = json.dumps(self.artifact(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def save(self, path):
        Path(path).write_text(json.dumps(self.artifact(), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get("version") != cls.VERSION:
            raise ValueError("Unsupported population decoder version")
        return cls(
            cell_ids=data["cell_ids"],
            means=data["means"],
            scales=data["scales"],
            weights=data["weights"],
            intercept=data["intercept"],
            position_scale=data["position_scale"],
            motion_scale=data["motion_scale"],
            ridge=data["ridge"],
            metadata=data.get("metadata"),
        )


def frame_conditions(settings):
    """The viewing conditions a decoder artifact is only valid under.

    Imported lazily so `motor` stays loadable without pulling in the renderer.
    """
    from .arena import EGO_SCALE
    from .breakout import BALL_D, FIELD_BOTTOM, FIELD_TOP, PADDLE_TOP

    return {
        "egocentric": bool(settings.egocentric),
        "ego_scale": float(EGO_SCALE),
        "paddle_width": int(settings.paddle_width),
        "ball_diameter": int(BALL_D),
        "field_top": int(FIELD_TOP),
        "field_bottom": int(FIELD_BOTTOM),
        "paddle_top": int(PADDLE_TOP),
        "neural_ms": float(settings.neural_ms),
    }


def fit_population_decoder(
    *,
    cell_ids,
    samples,
    offsets,
    ridge=1.0,
    position_scale=64.0,
    motion_scale=48.0,
    metadata=None,
):
    """Fit a frozen ridge decoder from probe rates to relative x offset."""
    x = np.asarray(samples, dtype=float)
    y = np.asarray(offsets, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Probe samples and offsets have incompatible shapes")
    if x.shape[1] != len(cell_ids) or len(x) < 3:
        raise ValueError("Probe matrix does not match the declared population")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Probe data must be finite")
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-9] = 1.0
    z = (x - means) / scales
    centred = y - y.mean()
    weights = np.linalg.solve(z.T @ z + float(ridge) * np.eye(z.shape[1]), z.T @ centred)
    return PopulationMotorDecoder(
        cell_ids=cell_ids,
        means=means,
        scales=scales,
        weights=weights,
        intercept=float(y.mean()),
        position_scale=position_scale,
        motion_scale=motion_scale,
        ridge=ridge,
        metadata=metadata,
    )


class RetinalVectorDecoder:
    """Winner-take-all population vector over mapped photoreceptors.

    The ridge decoder this replaces fitted descending-neuron firing rates to
    the ball's offset and recovered nothing: 158.5 px held-out against 161.6 px
    for shuffled labels. Two things were wrong with it.

    Descending neurons are a motor bottleneck, a few hundred cells carrying
    commands out of the brain. Retinotopic position lives in the eye, where
    thousands of photoreceptors tile the visual field. And a place code cannot
    be read by linear regression: a photoreceptor fires when the ball is over
    *its* patch of screen, so its rate against offset is a bump, not a line,
    and a fit over a handful of training offsets cannot interpolate it.

    The standard readout for a place code is a population vector: weight each
    cell by the screen position it looks at and take the centroid. Those
    positions are anatomical -- they come from the connectome's retinotopy via
    the retina adapter -- so they are not fitted to the game. Restricting the
    vote to the most strongly driven cells matters, because the ball is small:
    over the whole population its ~40 photoreceptors are swamped by background
    and paddle, giving r = 0.87 and 89 px, while the top percentile alone gives
    r = 0.99 and 17.5 px.

    What this is and is not: photoreceptors are the input layer, so this reads
    the retinal image rather than any computation the network performs on it.
    It is a genuine neural readout -- LIF photoreceptor spikes, decoded by
    anatomical position -- and it is honestly a sensory one. It should not be
    described as the network deciding anything.
    """

    VERSION = 1

    def __init__(self, *, cell_ids, positions, baseline, quantile, slope,
                 intercept, position_scale, motion_scale, metadata=None):
        self.cell_ids = tuple(str(x) for x in cell_ids)
        self.positions = np.asarray(positions, dtype=float)
        self.baseline = np.asarray(baseline, dtype=float)
        self.quantile = float(quantile)
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.position_scale = float(position_scale)
        self.motion_scale = float(motion_scale)
        self.metadata = dict(metadata or {})
        self.previous_offset = None
        n = len(self.cell_ids)
        if not n or len(self.positions) != n or len(self.baseline) != n:
            raise ValueError("Retinal decoder vector lengths do not match")
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("Retinal decoder quantile must lie in (0, 1)")
        if self.slope == 0.0 or not np.isfinite(self.slope):
            raise ValueError("Retinal decoder slope must be finite and nonzero")

    def centroid(self, rates):
        """Screen column the driven photoreceptors point at, or None if dark."""
        try:
            values = np.asarray([rates[c] for c in self.cell_ids], dtype=float)
        except KeyError as exc:
            raise ValueError(f"Missing photoreceptor rate for {exc.args[0]}") from None
        if not np.isfinite(values).all():
            raise ValueError("Photoreceptor rates must be finite")
        driven = np.maximum(values - self.baseline, 0.0)
        if not driven.any():
            return None
        cut = np.quantile(driven, self.quantile)
        weights = np.where(driven >= cut, driven, 0.0)
        total = weights.sum()
        if total <= 0:
            return None
        return float((weights * self.positions).sum() / total)

    def estimate(self, rates):
        """Ball offset in world px, or None when nothing is driven."""
        centre = self.centroid(rates)
        if centre is None:
            return None
        return (centre - self.intercept) / self.slope

    def decode(self, rates):
        offset = self.estimate(rates)
        if offset is None:
            # Nothing visible: hold rather than invent a direction.
            return {"offset_px": None, "relative_motion_px": 0.0,
                    "raw_control": 0.0, "control": 0.0,
                    "cell_count": len(self.cell_ids),
                    "decoder_sha256": self.artifact_sha256}
        motion = 0.0 if self.previous_offset is None else offset - self.previous_offset
        raw = offset / self.position_scale + motion / self.motion_scale
        self.previous_offset = offset
        return {"offset_px": offset, "relative_motion_px": motion,
                "raw_control": raw, "control": math.tanh(raw),
                "cell_count": len(self.cell_ids),
                "decoder_sha256": self.artifact_sha256}

    def forget_motion(self):
        self.previous_offset = None

    def check_compatible(self, settings):
        PopulationMotorDecoder.check_compatible(self, settings)

    def state(self):
        return {"previous_offset": self.previous_offset}

    def restore(self, state):
        value = state.get("previous_offset")
        self.previous_offset = None if value is None else float(value)

    def artifact(self):
        return {"version": self.VERSION, "kind": "retinal_vector",
                "cell_ids": list(self.cell_ids),
                "positions": self.positions.tolist(),
                "baseline": self.baseline.tolist(),
                "quantile": self.quantile, "slope": self.slope,
                "intercept": self.intercept,
                "position_scale": self.position_scale,
                "motion_scale": self.motion_scale, "metadata": self.metadata}

    @property
    def artifact_sha256(self):
        payload = json.dumps(self.artifact(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def save(self, path):
        Path(path).write_text(json.dumps(self.artifact(), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get("kind") != "retinal_vector" or data.get("version") != cls.VERSION:
            raise ValueError("Not a supported retinal vector decoder artifact")
        return cls(cell_ids=data["cell_ids"], positions=data["positions"],
                   baseline=data["baseline"], quantile=data["quantile"],
                   slope=data["slope"], intercept=data["intercept"],
                   position_scale=data["position_scale"],
                   motion_scale=data["motion_scale"],
                   metadata=data.get("metadata"))


def load_motor_decoder(path):
    """Load whichever decoder kind the artifact holds."""
    data = json.loads(Path(path).read_text())
    if data.get("kind") == "retinal_vector":
        return RetinalVectorDecoder.load(path)
    return PopulationMotorDecoder.load(path)
