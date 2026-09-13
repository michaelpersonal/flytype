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
