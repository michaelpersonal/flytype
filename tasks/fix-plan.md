# Fix plan: continuous MaleCNS FlyBreak

Owner: this working session. Supersedes the unchecked boxes in `todo.md`.

Everything below runs on the real MaleCNS graph. `--fixture` remains an
explicitly-labelled synthetic demo and is never a fallback. "Testable without
the graph" below means the *unit test* does not have to load 166,700 neurons,
not that the code path avoids MaleCNS.

## Measured facts this plan is built on

Recorded 2026-09-12 on this host, MaleCNS v1.0 retained graph.

| Fact | Value | Why it matters |
|---|---|---|
| Observation cost | ~0.8-1.4 s | 300 probes is ~5 min, not hours. The 21-probe budget was never a compute limit. |
| Warm-up transient | 185k -> 196k -> 318k -> 419k spikes over 4 observations on a **constant** frame, then 419k +/- 0.5% | The current calibration records from observation 1. ~19% of its probes are dominated by a ramp unrelated to the ball. |
| Candidate DN cells | 1,326 bilateral `type.startswith("DN")` | 32 weights are currently fitted on 16 training probes. |
| Trial noise, identical frame | right-left = -2.0 / +8.0 / +4.0 Hz | Per-observation noise is comparable to the ball signal. Conditions must be repeated and averaged. |
| Reachable ball-to-paddle offset | +/- 284 px | Probes only span +/- 120 px. The rest is extrapolation. |
| Ball visibility, 16 px in field y=12..104 | mean 22.8 photoreceptors, blind at 1.7% of positions, 10th pct 4 | The 24->16 px shrink did NOT re-break visibility. `BALL_D` stays 16. |

## Order and dependencies

Phase 1 must precede Phase 2: item 1 changes the decoder artifact schema, so
calibrating first would mean calibrating twice. Phase 3 needs a real artifact
from Phase 2. Phase 5 needs real numbers from Phases 2-3.

## Phase 1 - plumbing correctness (unit-testable without loading the graph)

- [x] **1. Bind the decoder to its calibration conditions.** Record
  `egocentric`, `ego_scale`, `paddle_width`, `ball_diameter`, `field_top`,
  `field_bottom`, `paddle_top`, `neural_ms` in the artifact metadata; add
  `check_compatible(settings)` raising on any mismatch at load.
  Closes: `--fixed-view` silently using an egocentric-fitted decoder;
  `--paddle-width` changing the frame but not the decoder.
- [x] **2. Reset the motion term on discontinuities.** `previous_offset = None`
  on episode rollover and whenever the ball is re-served (`last_event ==
  "lost"`). Today a new serve produces a spurious motion term up to +/-284/48.
- [x] **3. Disambiguate the two action fields.** `event["action"]` stays the
  motor action; add `event["dnp20_action"]`; state in provenance that DNp20 is
  telemetry and does not gate control.
- [x] **4. Record the renderer variant per run** so `build_breakout_frames.py`
  reproduces either loop's frames. `_play_loop` passes `action=`/`stimulus=`
  to `render_arena`; `ContinuousPlaySession` does not.

Not doing: unifying `flytype play` into `ContinuousPlaySession`. It is the
highest-risk change available and the legacy DNp20 episodes in
`docs/validation.md` depend on that loop. Revisit only on request.

## Phase 2 - calibration redesign, then actually run it

- [x] **5. Warm-up block.** Discard the first 8 observations before recording
  any probe. Single biggest lever.
- [x] **6. Cover the real range.** Probe offsets across +/- 284 px.
- [x] **7. Vary ball height.** Grid of (offset x height) across y=12..88 so the
  fit is forced to be height-invariant; report held-out error per height band.
- [x] **8. Repeat and average each condition.** Repeat count sized from the
  measured per-cell d'.
- [x] **9. Shuffled-label null control.** Refit the identical pipeline on
  permuted offsets using the same recorded rates; report its held-out
  correlation next to the real one. No extra compute. Non-negotiable before
  any number goes in a doc.
- [x] **10. Run it and commit the artifact** so the numbers are reproducible.

## Phase 3 - verify the real path

- [x] **11.** A `FLYTYPE_FULL_TEST=1` test driving one real MaleCNS
  observation through `ContinuousPlaySession`, asserting `source == "malecns"`.
  `todo.md` cites this verification; it does not exist.
- [x] **12.** One real `flytype web` launch; confirm a snapshot reaches the
  browser; compare tracking rate against the measured 0.498 chance baseline
  with `tools/analyze_play.py`.

## Phase 4 - presentation only (zero lines changed in `arena.py`)

- [x] **13.** Scale the browser canvas to the `field_top..field_bottom` band
  instead of all 180 rows. Removes the ~42% dead void under the paddle.
- [x] **14.** Close the right-rail gap; reposition the event toast.
- [x] **15.** Label the retinal panel: bricks are **not** drawn in the
  egocentric frame, so the model never sees them. Currently the most
  misleading element on screen for a recording.

## Phase 5 - documentation

- [x] **16.** Correct `breakout.py:13`, `README.md:90`, `docs/validation.md:79`
  (all still say field `y=12..64` and a 10 px ball). Replace the
  unreproducible r=0.558 / MAE 83.85 px with what Phase 2 produces, alongside
  its shuffled-label control.

## Findings that came out of doing this

1. **The population decoder does not decode.** 433 real observations, held-out
   MAE 158.5 px against 161.6 px for shuffled labels and 166.5 px for a decoder
   that always predicts the mean. Averaging 5 repeats per condition (equivalent
   to 2.5 s of integration) moved it to 156.9 px, so this is not a
   noise-limited result and a longer observation window will not rescue it.
   The previously published 0.558 / 83.85 px is withdrawn in docs/validation.md.
2. **The paddle performs significantly worse than chance, not merely at
   chance.** 305 real observations: tracking 0.371 over 256 scored moves,
   -4.12 SE, p < 0.0001, paddle confined to the right-hand 41% of the field.
   Over the same observations the DNp20 telemetry was balanced (156 L / 147 R)
   while the population decoder went 93 / 212, so the bias is in the decoder's
   fitted normalisation, not in the network.
3. **A linear readout may be the wrong model rather than the wrong pathway.**
   Earlier d' probes of the DNp20 pair separated far-left from far-right at
   d' = 3.32 at +/-250 px while sitting near zero at +/-180 px. That
   non-monotonic shape is unrepresentable by any linear decoder. Probe rates
   are now saved to `play-population.probes.npz` so a nonlinear readout can be
   tried offline with no further MaleCNS time.
4. **The web loop costs ~2.5 s per observation, and it is neural compute.**
   Measured 2.0 s of compute during play against 0.64 s for the same 500 ms
   window during calibration: play runs with plasticity on and a reinforcement
   pulse on nearly every observation, calibration runs frozen and unreinforced.
   Writing and hashing the 7 MB checkpoint is ~4 ms of it, so the checkpoint
   cadence is not worth changing. The 0.18 s default tick is unreachable in
   MaleCNS mode regardless.
5. **Calibration regime != play regime.** The decoder is fitted frozen and
   unreinforced and deployed plastic and reinforced, so its per-cell z-scoring
   is applied to a rate distribution it never saw. Measured as a standing
   +276 px bias on the offset estimate, present at full size from observation 1
   and reaching +655 px when the ball can never be more than 284 px away.
   Recorded and surfaced on the page, not corrected: re-centring at runtime
   would mean the decoder was reading something other than neural firing.

## Guardrails

- Full suite green between every item; never batch two items into one commit.
- Phases 1 and 4 must not touch the sensory path: `git diff --stat` on
  `flytype/arena.py` and the geometry constants in `flytype/breakout.py` must
  be empty before those commits.
- Any geometry change re-runs the photoreceptor-coverage measurement.
- No claim about steering goes in a doc unless pooled play beats the measured
  0.498 chance level on an exact binomial test.
