"""Fit a frozen neural population decoder on separate visual probe frames.

Calibration is the only place labelled offsets are allowed to touch neural
data, and it happens once, offline, with plasticity frozen and no
reinforcement. The artifact it writes is then read-only at runtime: the
decoder sees firing rates and its own previous estimate, never a live ball
coordinate, score, outcome or correct direction.

Four properties of this substrate shape the probe design, all measured on this
host rather than assumed (see docs/validation.md):

* The network needs roughly four observations to reach steady state. On an
  unchanging frame total spikes ran 185k, 196k, 318k, 419k and only then held
  at 419k +/- 0.5%. Probes recorded before that ramp settles are dominated by
  it, so a warm-up block is discarded before anything is kept.
* Single observations are noisy. Repeating one frame moved the DNp20
  difference over several Hz, comparable to the ball's own effect, so every
  condition is repeated and the split is taken across conditions.
* The ball reaches +/- 284 px from the paddle, not the +/- 120 px an earlier
  probe set covered. Beyond the fitted range a linear decoder extrapolates.
* The retina samples vertically very unevenly, so a fit at one ball height
  does not transfer. Conditions span height as well as offset, and held-out
  error is reported per height band.
"""

import dataclasses
import math
from pathlib import Path

import numpy as np

from .arena import render_arena
from .breakout import (
    BALL_D,
    FIELD_BOTTOM,
    FIELD_TOP,
    WIDTH,
    BreakoutGame,
    new_game_rng,
)
from .motor import fit_population_decoder, frame_conditions
from .neural.common import annotations


def descending_population(controller):
    """Return declared bilateral descending cells in retained graph order."""
    table = annotations(controller.brain.ids)
    types = table.type.fillna("")
    sides = table.somaSide.fillna("")
    mask = types.str.startswith("DN") & sides.isin(("L", "R"))
    return [str(cell_id) for cell_id in controller.brain.ids[mask.to_numpy()]]


def reachable_offsets(settings):
    """Widest ball-centre minus paddle-centre the arena can actually produce."""
    half_ball = BALL_D / 2
    half_paddle = settings.paddle_width / 2
    return (
        half_ball - (WIDTH - half_paddle),
        (WIDTH - half_ball) - half_paddle,
    )


def reachable_heights():
    """Top and bottom the ball's own top edge can occupy inside the field."""
    return float(FIELD_TOP), float(FIELD_BOTTOM - BALL_D)


def probe_conditions(settings, offset_count, height_count):
    """The (offset, height) grid the decoder is fitted over."""
    low, high = reachable_offsets(settings)
    top, bottom = reachable_heights()
    offsets = np.linspace(low, high, int(offset_count))
    heights = (
        np.linspace(top, bottom, int(height_count))
        if height_count > 1
        else np.asarray([(top + bottom) / 2])
    )
    return [
        {"offset": float(offset), "height": float(height)}
        for height in heights
        for offset in offsets
    ]


def _select_features(samples, offsets, cell_ids, limit):
    x = np.asarray(samples, dtype=float)
    y = np.asarray(offsets, dtype=float)
    xc = x - x.mean(axis=0)
    yc = y - y.mean()
    denominator = np.linalg.norm(xc, axis=0) * np.linalg.norm(yc)
    correlation = np.divide(
        np.abs(xc.T @ yc),
        denominator,
        out=np.zeros(x.shape[1], dtype=float),
        where=denominator > 0,
    )
    count = min(int(limit), x.shape[1])
    selected = np.argsort(correlation, kind="stable")[-count:]
    return selected, [cell_ids[i] for i in selected]


def _fit_and_score(rates, labels, trial_condition, training, cell_ids,
                   feature_count, ridge, position_scale, motion_scale):
    """Select, fit and validate once. Splits by condition, never by trial.

    Repeats of one condition are near-duplicates, so letting some land in
    training and others in validation would score the fit on frames it had
    effectively already seen. The mask is over conditions and every trial
    follows its own condition.
    """
    training_trials = training[trial_condition]
    y = labels[trial_condition]
    selected, selected_ids = _select_features(
        rates[training_trials], y[training_trials], cell_ids, feature_count
    )
    decoder = fit_population_decoder(
        cell_ids=selected_ids,
        samples=rates[training_trials][:, selected],
        offsets=y[training_trials],
        ridge=ridge,
        position_scale=position_scale,
        motion_scale=motion_scale,
    )
    held_out = ~training_trials
    predicted = np.asarray([
        decoder.estimate(dict(zip(selected_ids, row[selected])))
        for row in rates[held_out]
    ])
    expected = y[held_out]
    return decoder, selected, selected_ids, predicted, expected


def _scores(predicted, expected):
    mae = float(np.mean(np.abs(predicted - expected))) if len(expected) else None
    correlation = None
    if len(expected) > 1 and np.std(predicted) > 0 and np.std(expected) > 0:
        correlation = float(np.corrcoef(predicted, expected)[0, 1])
        if not math.isfinite(correlation):
            correlation = None
    return mae, correlation


def calibrate_play_decoder(
    settings,
    out,
    *,
    dataset_info,
    offset_count=17,
    height_count=5,
    repeats=3,
    warmup=8,
    feature_count=32,
    ridge=4.0,
    position_scale=64.0,
    motion_scale=48.0,
    controller=None,
    progress=None,
):
    """Run non-reinforced MaleCNS probes and save a frozen decoder artifact."""
    if settings.fixture:
        raise ValueError("Population calibration requires MaleCNS, not fixture mode")
    if offset_count < 5:
        raise ValueError("Population calibration needs at least 5 offsets")
    if height_count < 1 or repeats < 1 or warmup < 0:
        raise ValueError("Invalid probe grid")
    if feature_count < 2:
        raise ValueError("Population calibration needs at least 2 cells")

    calibration_settings = dataclasses.replace(
        settings,
        frozen=True,
        fixture=False,
        fast=False,
        shuffle_feedback=False,
    )
    if controller is None:
        from .neural.controller import FlyController

        controller = FlyController(calibration_settings)

    population = descending_population(controller)
    if not population:
        raise RuntimeError("No bilateral descending-neuron population was found")

    game = BreakoutGame(calibration_settings, new_game_rng(settings.seed))
    conditions = probe_conditions(calibration_settings, offset_count, height_count)

    def probe_frame(offset, height):
        view = dict(game.view())
        view["ball_x"] = game.paddle_center - BALL_D / 2 + float(offset)
        view["ball_y"] = float(height)
        return render_arena(view, egocentric=calibration_settings.egocentric)

    frames = [probe_frame(c["offset"], c["height"]) for c in conditions]

    # Settle the network before anything is recorded. The stimulus is the
    # neutral one -- ball centred on the paddle -- so the warm-up cannot bias
    # any direction, and none of it is kept.
    neutral = probe_frame(0.0, (FIELD_TOP + FIELD_BOTTOM - BALL_D) / 2)
    total = warmup + len(conditions) * repeats
    done = 0
    for _ in range(warmup):
        result = controller.observe(neutral, "none")
        done += 1
        if progress is not None:
            progress(done, total, result.get("compute_seconds"))

    # Interleave every repeat of every condition in one seeded random order, so
    # any residual drift across the block spreads evenly over conditions rather
    # than loading onto whichever ones happen to run early.
    rng = np.random.default_rng(settings.seed ^ 0xC411B)
    order = rng.permutation(len(conditions) * repeats)
    rates = np.zeros((len(order), len(population)), dtype=float)
    trial_condition = np.zeros(len(order), dtype=int)
    for slot, trial in enumerate(order):
        condition = int(trial % len(conditions))
        result = controller.observe(
            frames[condition], "none", population_ids=population
        )
        values = result["population_rates"]
        rates[slot] = [values[cell_id] for cell_id in population]
        trial_condition[slot] = condition
        done += 1
        if progress is not None:
            progress(done, total, result.get("compute_seconds"))

    labels = np.asarray([c["offset"] for c in conditions], dtype=float)
    heights = np.asarray([c["height"] for c in conditions], dtype=float)
    held_out_conditions = rng.permutation(len(conditions))[: max(1, len(conditions) // 4)]
    training = np.ones(len(conditions), dtype=bool)
    training[held_out_conditions] = False

    decoder, selected, selected_ids, predicted, expected = _fit_and_score(
        rates, labels, trial_condition, training, population,
        feature_count, ridge, position_scale, motion_scale,
    )
    mae, correlation = _scores(predicted, expected)

    # Null control: the identical pipeline -- same rates, same selection, same
    # ridge, same split -- fitted to a permuted condition-to-offset mapping.
    # Whatever it scores is what this many free parameters extract from noise,
    # and the real result only means something above it.
    shuffled_labels = labels[rng.permutation(len(conditions))]
    _, _, _, null_predicted, null_expected = _fit_and_score(
        rates, shuffled_labels, trial_condition, training, population,
        feature_count, ridge, position_scale, motion_scale,
    )
    null_mae, null_correlation = _scores(null_predicted, null_expected)

    held_trials = ~training[trial_condition]

    def averaged(predicted_values, label_values):
        """Score the held-out conditions after averaging their repeats.

        Averaging cuts per-observation noise by sqrt(repeats) while leaving any
        real signal untouched, so a score that improves sharply here means the
        signal exists but is buried in single-observation noise, and a longer
        integration window would expose it. Runtime still decodes one
        observation at a time -- this is a diagnostic, never a decoding path.

        It has a substantial noise floor of its own at this many held-out
        conditions, which is why the shuffled control is scored the same way
        and reported beside it. Neither number means anything alone.
        """
        rows, wanted = [], []
        for condition in np.flatnonzero(~training):
            selected = predicted_values[trial_condition[held_trials] == condition]
            if len(selected):
                rows.append(float(np.mean(selected)))
                wanted.append(float(label_values[condition]))
        return _scores(np.asarray(rows), np.asarray(wanted))

    averaged_mae, averaged_correlation = averaged(predicted, labels)
    null_averaged_mae, null_averaged_correlation = averaged(
        null_predicted, shuffled_labels
    )

    # What a decoder that learned nothing scores, for the same held-out set.
    constant_mae = float(np.mean(np.abs(expected - labels[training].mean())))

    by_height = {}
    for height in sorted(set(heights.tolist())):
        mask = heights[trial_condition][held_trials] == height
        if mask.any():
            band_mae, band_r = _scores(predicted[mask], expected[mask])
            by_height[f"{height:.0f}"] = {"mae_px": band_mae, "correlation": band_r}

    decoder.metadata.update({
        "source": "malecns visual offset probes",
        "candidate_rule": "type starts with DN and somaSide is L or R",
        "candidate_cells": len(population),
        "selected_cells": len(selected_ids),
        "conditions": len(conditions),
        "offset_count": int(offset_count),
        "height_count": int(height_count),
        "repeats": int(repeats),
        "warmup_discarded": int(warmup),
        "observations": total,
        "offset_range_px": list(reachable_offsets(calibration_settings)),
        "height_range_px": list(reachable_heights()),
        "training_conditions": int(training.sum()),
        "validation_conditions": int((~training).sum()),
        "reinforcement": "none",
        "frozen_during_calibration": True,
        "dataset": dataset_info,
        "frame_conditions": frame_conditions(calibration_settings),
        "validation_mae_px": mae,
        "validation_correlation": correlation,
        "shuffled_control_mae_px": null_mae,
        "shuffled_control_correlation": null_correlation,
        "predict_the_mean_mae_px": constant_mae,
        "condition_averaged_mae_px": averaged_mae,
        "condition_averaged_correlation": averaged_correlation,
        "condition_averaged_shuffled_mae_px": null_averaged_mae,
        "condition_averaged_shuffled_correlation": null_averaged_correlation,
        "validation_by_ball_height": by_height,
    })
    decoder.save(Path(out))
    # The probe rates cost real connectome time to collect; keeping them beside
    # the artifact means any later question about this fit -- a different
    # regulariser, more features, a nonlinear model -- can be answered offline
    # instead of re-running the graph.
    np.savez_compressed(
        Path(out).with_suffix(".probes.npz"),
        rates=rates,
        trial_condition=trial_condition,
        offsets=labels,
        heights=heights,
        training=training,
        cell_ids=np.asarray(population),
    )
    return decoder
