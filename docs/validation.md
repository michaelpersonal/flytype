# Validation status

Recorded during implementation on 2026-09-11 in a real macOS environment with
network access. **The full MaleCNS v1.0 release was actually downloaded and
compiled** (not skipped as unavailable): 166,700 neurons, 25,582,938 directed
connections, 124,177,617 synaptic contacts, matching the committed checksum
locks exactly.

Current local result (2026-09-12): **88 passed, 6 skipped**; the skipped tests
are the opt-in full-connectome ones, which pass under `FLYTYPE_FULL_TEST=1`.
A real MaleCNS run (below) produced stimulus frames, neural-only decoded
actions, delayed reinforcement, valid checkpoints, and resumed correctly from
committed state.

The brick-breaker sections below were revised on 2026-09-12 after re-measuring
the geometry and re-running the population calibration with warm-up, full
offset range, repeated conditions and matched null controls. **Two figures
previously reported here are withdrawn** and marked as such in place: the
population decoder's 0.558 correlation / 83.85 px error, and a 0.545 tracking
rate for one egocentric episode.

| Check | Observed result | What it does not establish |
| --- | --- | --- |
| Rebuild from checksum-verified released files | 166,700 neurons, 25,582,938 connections and 124,177,617 contacts; every compiled array matches its lock | Completeness or physiological accuracy of the reconstruction |
| Unit tests (task/decoder/feedback/state/fixture CLI) | Target advances only on correct committed selection; incorrect choices never alter accepted text; distractors never equal the correct character; seeded candidate placement is deterministic; SPACE renders correctly but persists as a literal space; decoder output depends only on supplied counts and configured deadband; reinforcement delayed exactly one observation; fixture events tagged `"source": "fixture"`, real events `"source": "malecns"`; resume restores target index, RNG state, pending reinforcement and (real mode) brain checkpoint; incompatible resume configuration is rejected; completion requires exact target equality | Behavior of any run not covered by these specific assertions |
| Full-network sensory/feedback test | A white RGB frame activates KCs; reward pulses spike PAM11; aversive pulses spike PPL101; eligible synapses change under reward and stay exactly unchanged under `--frozen` | Accurate fly retinal responses or an acquired character-selection association |
| Full-network checkpoint test | Checkpoint restore reproduces committed weights and simulation cursor exactly | Retention of a useful strategy or conditioning equivalent to a biological experiment |
| Fixture end-to-end run | `run --fixture --seed 42 --steps 2000` completed `follow @michaelzsguo on x` in 27 observations with zero unscored attempts | A property of the real MaleCNS decoder; the fixture decoder is a scripted stand-in |
| Real MaleCNS run, `--seed 42` | See below | A realistic model of fly behavior, spelling ability, or unbiased character selection |

## Real MaleCNS run

```sh
python -m flytype run --target 'follow @michaelzsguo on x' --seed 42 --out runs/michael-follow
```

The first 5 observations of this run (before any reinforcement had been
delivered) show:

| Observation | Right−Left DNp20 (Hz) | DNpe017 spikes (logged only) | Action | Outcome |
| --- | --- | --- | --- | --- |
| 1 | +10.0 | 29 | RIGHT | incorrect |
| 2 | +4.0 | 11 | RIGHT | incorrect |
| 3 | +6.0 | 5 | RIGHT | incorrect |
| 4 | +4.0 | 0 | RIGHT | correct |
| 5 | +8.0 | 0 | RIGHT | incorrect |

Every observation in this run decoded RIGHT: the network's spontaneous
right-vs-left DNp20 bias exceeded the 0.1 Hz deadband and stayed positive
across all five observations, even though DNpe017 spike counts (logged, never
gating) varied and even dropped to 0. This is the directional-bias limitation
predicted in [the model](model.md): because each character's correct side is
independently randomized, a fixed one-sided decoder still completes the
sentence over enough attempts (roughly half of randomized placements put the
correct character on the biased side), but it is not evidence that the
network is "choosing" correctly.

This same run was then continued (`--steps 0`) to completion. It finished
`follow @michaelzsguo on x` (25 characters) after **53 total observations**:
51 RIGHT, 1 LEFT, 1 HOLD; 25 correct, 27 incorrect, 1 hold — almost exactly
2.1 attempts per accepted character, consistent with a strong RIGHT bias
combined with independently randomized correct-side placement. No character
hit `--max-attempts-per-character` (default 200); the run never halted.
`flytype status --out runs/michael-follow` reports `"done": true` and the
exact typed text.

## Brick-breaker task (`flytype play`)

The paddle task was added to ask a question the typing task cannot answer: the
typing display re-randomizes which side is correct on every attempt, so a fixed
directional bias still completes the sentence. A paddle has to go to a
*particular* place, so a biased or uninformative readout simply loses.

**Chance level is measured, not assumed.** `tracking_rate` is the share of
non-HOLD moves that went the way the ball actually lay, excluding moves made
while the paddle was already within half a step of it. Over 200 matched
episodes a random agent scores **0.498** (387 ticks, 2.75 paddle hits, 2.88
bricks); an always-RIGHT agent scores **0.051**.
`tests/test_breakout.py::test_random_play_tracks_at_chance` pins the definition.

**Two structural findings came out of building it.**

*The simulated eye does not see most of the frame.* Of 3,335 mapped R1–R6 and
811 mapped R8 photoreceptors, roughly two thirds land in the top third of the
frame and the bottom-left quadrant receives none. A 6 px ball in a conventional
full-frame arena drew a mean of **1.9** photoreceptors and was completely
invisible at 28 of 42 sampled positions. This also applies to the typing
display, whose key band extends well below the sampled region.

Re-measured 2026-09-12 for the current geometry — field `y=12..104`, 16 px
circular ball — over every ball position the arena can produce, on a 6 px grid:

| ball | mean photoreceptors on the ball | positions with none | 10th percentile |
| --- | --- | --- | --- |
| 6 px, full-height field (original) | 1.9 | 67% | 0 |
| 16 px, `y=12..104` (current) | **22.8** | **1.7%** | 4 |
| 24 px, `y=12..104` (previous) | 50.5 | 0.2% | 10 |

The ball is comfortably visible at its current size. The 10th-percentile figure
is the cost of the smaller ball: in the worst decile of positions it lands on
about four photoreceptors, so this table is worth regenerating whenever the
geometry moves.

*A mean-rate readout cannot compute a relative position.* Probing the pathway
directly — same paddle geometry, ball pinned either side of it, paddle parked at
three world positions — the right-minus-left DNp20 rate separated ball-left from
ball-right by **+0.00 Hz (d′ = 0.00)** under a fixed view. The readout does
carry the ball's *absolute* screen position (+2.58 Hz between the far-left and
far-right edges of the frame), but the task needs its position *relative to the
paddle*, and two cells summed into one number cannot subtract one from the
other. Rendering the field scrolled so the paddle sits at the frame centre
(`--egocentric`) makes those two quantities the same quantity; the same probe
then separates the conditions by **+1.89 Hz**, rising to d′ = 3.03 when the ball
is far from the paddle and falling to roughly zero when it is close.

**Legacy DNp20 episodes.** These were recorded under an **earlier arena** — an
80 px paddle and the old `y=12..64` field — and the fixed-view run predates the
`egocentric` setting existing at all. They are kept because they are what the
two-cell readout did, but they are not reproducible with the current geometry
and must not be compared against anything measured after it changed. Recounted
2026-09-12 from the event logs still in `runs/` with `tools/analyze_play.py`:

| view | ticks | scored moves | tracking | exact two-sided p |
|---|---|---|---|---|
| fixed (`runs/breakout-fixedview`) | 257 | 179 | 0.475 | 0.55 |
| egocentric (`runs/breakout`) | 193 | 137 | 0.482 | 0.73 |

Neither is distinguishable from chance, and both sit slightly below it. **No
claim is made that the network plays this game.** An earlier revision of this
document reported 0.545 for the egocentric episode; that run directory has
since been overwritten and the figure is not reproducible, which is why the
table above is recounted from files rather than carried forward.

`--decoder-baseline-obs N` centers the decision on the running median of the
decoder's own last N right-minus-left values. Without it the raw difference is
almost always positive — 39 of 52 observations in the typing run — and the
paddle pins to one wall. The median is a function of the decoder's own past
output only; it never sees the arena, the outcome or the correct direction, so
it can remove a standing offset but cannot supply an answer, and if the image
carries no side information the result is a fair coin (which is exactly what
the fixed-view episode shows). Exact ties decode as HOLD, which is why roughly
a third of observations do not move the paddle at all.

## Reproduce

```sh
# Frozen population decoder for the continuous site, with its own controls
python -m flytype calibrate-play-decoder --out calibration/play-population.json \
    --seed 7 --repeats 5
# The continuous site itself, real MaleCNS
python -m flytype web --out runs/web-malecns
# Legacy two-cell DNp20 paddle
python -m flytype play --out runs/breakout --seed 7 --decoder-baseline-obs 16 --egocentric
python tools/analyze_play.py runs/breakout runs/breakout-fixedview
python -m pytest -q
FLYTYPE_FULL_TEST=1 python -m pytest -q tests/test_neural_integration.py
python -m flytype prepare
python -m flytype verify
python -m flytype run --fixture --seed 42 --steps 2000 --out runs/check-fixture
python -m flytype run --target 'follow @michaelzsguo on x' --seed 42 --out runs/michael-follow
```

All runtime evidence stays local in `runs/`; it is not uploaded with this
report. Prices, spike patterns and outcomes will differ machine to machine and
run to run even with the same seed if the MaleCNS source files, kernel source,
or decoder configuration differ from what `provenance.json` records for the
run.

## Continuous population-decoder calibration

The continuous website drives the paddle from a frozen population decoder: a
ridge fit from the firing rates of 32 selected bilateral descending neurons
(chosen from 1,326 candidates) to the ball's horizontal offset from the paddle.
Calibration is the only place labelled offsets touch neural data. It runs
offline with plasticity frozen and no reinforcement, and the artifact is
read-only at runtime.

### What the substrate does, measured before designing the probes

| Property | Measurement | Consequence for the probe design |
|---|---|---|
| Startup transient | On an **unchanging** frame, total spikes ran 185k → 196k → 318k → 419k, then held at 419k ± 0.5% | The network needs ~4 observations to settle. The first 8 are now discarded before anything is recorded. |
| Per-observation noise | Repeating one frame moved the DNp20 difference over several Hz — comparable to the ball's own effect | Every condition is repeated, and the train/validation split is taken **across conditions**, never across repeats of one. |
| Reachable offset | ±284 px, from the arena geometry | Probes span the full range. An earlier probe set covered ±120 px, so most of the operating range was extrapolation. |
| Retinal sampling in y | Strongly uneven | Conditions span ball height as well as offset, and error is reported per height band. |
| Observation cost | ~0.6–1.4 s | A few hundred probes costs minutes, so the small probe count was never a compute constraint. |

### Result

`calibrate-play-decoder --seed 7 --repeats 5`, 433 real MaleCNS observations,
85 conditions (17 offsets × 5 heights), 64 training and 21 held-out conditions:

| Measure | Value | Matched control |
|---|---|---|
| Held-out MAE | **158.5 px** | 161.6 px with shuffled labels; **166.5 px** predicting the mean |
| Held-out correlation | 0.229 | −0.063 with shuffled labels |
| MAE, repeats averaged per condition | 156.9 px | 161.6 px with shuffled labels |
| Correlation, repeats averaged | 0.406 | −0.103 with shuffled labels |

**The decoder does not recover ball position.** It beats a decoder that has
learned nothing and always predicts the mean by 5%, and it beats its own
shuffled labels by 2%. The correlation looks more encouraging than the error
does, but this pipeline produces a held-out correlation near **0.42 from pure
noise** — verified on synthetic data in
`tests/test_play_calibration.py::test_pure_noise_does_not_look_like_a_signal`
— so 0.229 is at the floor, and 0.406 on 21 held-out conditions is t = 1.94,
p ≈ 0.07, not significant.

**This is not a noise problem.** Averaging five repeats per condition cuts
per-observation noise by √5 while leaving any real signal intact. It moved the
error from 158.5 px to 156.9 px. Had the signal been present but buried, that
average — equivalent to 2.5 s of integration — would have exposed it. A longer
`--neural-ms` will therefore not fix this.

What it does *not* establish: that the pathway carries no offset information at
all. Earlier direct d′ probes of the DNp20 pair separated far-left from
far-right at d′ = 3.32 at ±250 px while sitting near zero at ±180 px — a
**non-monotonic** relationship, which a linear decoder cannot represent whatever
its regularisation. A nonlinear readout over the same recorded rates is the
open question, and the probe rates are saved beside the artifact
(`play-population.probes.npz`) so it can be asked without further MaleCNS time.

An earlier revision of this document reported correlation **0.558** and MAE
**83.85 px** from a 21-probe calibration. Those numbers are withdrawn: that fit
spanned only ±120 px (a narrower range mechanically lowers MAE), recorded from
the first observation with no warm-up, fitted 32 weights to 16 training probes,
and had no control to compare against. No artifact from it survives.

Fixture mode is synthetic and never substitutes for a failed real calibration;
in fixture mode the control is scripted from the ball position and the page
says so.

### The decoder driving a real game

`python -m flytype web --out runs/web-malecns`, 305 real MaleCNS observations
across 4 episodes, stopped cleanly through the browser control API. Scored with
`tools/analyze_play.py` against the measured chance level of 0.5:

| Measure | Value |
|---|---|
| Scored moves | 256 of 305 observations |
| Tracking rate | **0.371** |
| Distance from chance | −4.12 SE, exact two-sided binomial **p < 0.0001** |
| Paddle moves | 93 LEFT against 212 RIGHT |
| Field visited | paddle_x 155..264 of 0..264 |
| Offset estimates outside ±284 px | **30%**, reaching +828 px |

The paddle does not merely fail to track the ball — it tracks it **significantly
worse than random**, and confines itself to the right-hand 41% of the field.
This is the standing offset bias of the previous section acting on the control:
a persistently positive offset estimate is a persistent RIGHT command.

The decisive detail is the telemetry. Over the same 305 observations the
two-cell DNp20 readout, recorded but never gating, came out very nearly
balanced — 156 LEFT against 147 RIGHT — while the population decoder driving
the paddle went 93 against 212. The network is not biased to one side. The
decoder's fitted normalisation is, because it is being applied to rates from a
regime it was not fitted on. The two decoders agreed on 68% of observations,
which is also why they are now recorded separately as `action` and
`dnp20_action`.

Episode rollover, archival, resume and the loopback control API all behaved
correctly throughout; those parts of the site work. What does not work is the
decode.

### Why nothing downstream of the lamina fires

Attempted 2026-09-13: make T4/T5 fire. It did not succeed, and the reason is
specific enough to be worth recording. `tools/probe_propagation.py` reproduces
all of it.

With the shipped parameters a cell rests at -52 mV and fires at -45 mV, so
sustained input must exceed **+7** for the cell to spike. Measured net input:

| population | net input per cell | state |
|---|---|---|
| Mi1 (ON-pathway relay) | **-41.38** | clamped below its own resting potential |
| T4c | +3.14 | below threshold |

Mi1 gates the whole downstream optic lobe, and it is held 41 units below rest,
with excitation +32.6 against inhibition -74.0. Its single largest input is
L1 at -22.05 per cell.

**The proximate cause is the transmitter-to-sign proxy.** `transmitter_signs`
maps glutamate to inhibitory everywhere, and its own docstring says "This is
NOT receptor physiology." In the fly, L1 -> Mi1 is the excitatory ON-pathway
synapse; the sign is set by the postsynaptic receptor, which the annotation
does not carry. Flipping every lamina -> medulla inhibitory edge to excitatory
moves Mi1 from -41.38 to +2.72 -- still under the +7 it needs.

**Global knobs cannot fix a balanced circuit.** T4c receives +59,174
excitatory against -53,594 inhibitory. Scaling the synaptic gain scales both,
and adding a tonic background wakes T4's inhibitors (Mi9, Mi4, CT1, Tlp13/14)
as readily as its drivers. Measured across 22 configurations spanning synaptic
gain 1.0-4.0, tonic 0-8, the lamina sign flip, and moving as well as static
stimuli:

* T4/T5 never exceeded 5% of cells firing, and were at 0.0% under every stock
  setting, with a moving ball as well as a static one.
* **Every configuration that made them fire produced zero cells encoding ball
  position.** Forcing activity does not restore information.

No parameter set was adopted. A more active brain that encodes nothing is worse
than an honestly silent one, because it looks like progress.

**What would actually be required**, none of which is a parameter change:
receptor-level synapse signs rather than a transmitter proxy; graded
transmission for photoreceptors, lamina and much of the medulla, which are
non-spiking in vivo; and per-cell-type excitability instead of one rest and one
threshold for all 166,700 cells. The compiled manifest already says as much:
"Photoreceptors and lamina are graded in vivo. This experiment uses an explicit
LIF proxy, low-pass luminance drive and tonic lamina current; it is not
validated fly vision."

This is also why the retinal population vector is the readout that works, and
why reading further downstream is not available: there is nothing downstream to
read.

### The readout was wrong, not the network

Everything above says the frozen ridge decoder over descending neurons cannot
read the ball. That result stands, and the reason turned out to be the choice
of population and the choice of readout, not the simulation.

A whole-brain probe (15 offsets x 6 repeats, 50 ms window) settles it. Per-cell
linear correlation with offset finds nothing anywhere: max |r| = 0.359 across
all 166,700 neurons against 0.376 for shuffled labels. But that is the wrong
statistic. A photoreceptor fires when the ball is over **its** patch of screen,
so its rate against offset is a bump, not a line, and linear correlation is
near zero for a perfectly informative cell. An F-test, which detects any
dependence on offset whatever its shape, finds **805 cells** above the shuffled
maximum (max F 25.5 against 5.0).

The information is there. Reading it needs the decoder that matches a place
code: a **population vector**, weighting each photoreceptor by the screen
column it looks at -- a position taken from the connectome's retinotopy through
the retina adapter, not fitted to anything -- and taking the centroid.

| readout | held-out MAE | r | control |
|---|---|---|---|
| ridge over 32 descending neurons | 158.5 px | 0.23 | 161.6 px shuffled |
| population vector, all photoreceptors | 88.6 px | 0.87 | shuffled r 0.12 |
| population vector, top 1% driven cells | **17.5 px** | **0.99** | shuffled r 0.11 |

Restricting the vote matters because the ball is small: across the whole
population its ~40 photoreceptors are swamped by background and paddle.

**Driving the game with it, the paddle tracks the ball.** Over 502 real MaleCNS
observations: tracking **0.735** on 283 scored moves, **+7.91 SE above chance,
exact two-sided p < 0.0001**; 30 paddle hits, 12 of 14 bricks broken, one ball
lost. The offset estimate's standing bias is gone (mean +11 px against +276),
the paddle uses the whole field (0..264, against 155..264), and control
saturation falls from 73% to 32%. Fitted in the live pipeline at one ball
height the decoder measures r = 0.81 and MAE 97 px, weaker than the 17.5 px
probe figure because retinal sampling is uneven vertically and the live ball
spans heights the fit did not cover.

**The honest caveat, which matters.** Photoreceptors are the input layer. This
reads the retinal image, not a computation the network performs on it. It is a
genuine neural readout -- LIF photoreceptor spikes decoded by anatomical
position -- and it is a sensory one. It shows the simulated eye sees the ball
and that a standard population-vector readout of that eye can steer a paddle.
It is not evidence that the network decides, learns, or plays.

### The cause: the decoder is deployed outside its calibration regime

Calibration runs with plasticity **frozen** and reinforcement **off**. The
continuous site runs with plasticity on, and reinforcement fires on nearly
every observation — 57 of the first 58 in `runs/web-malecns` (15 reward, 42
aversive). Reinforcement pulses drive PAM11/PPL101 and shift firing
network-wide, so the per-cell means and scales the decoder z-scores with were
fitted on a rate distribution it never sees during play.

The symptom is unmistakable. Over the first 58 real observations the offset
estimate averaged **+276 px** and reached **+655 px**, when the ball can never
be more than 284 px from the paddle. The bias is present at full size from
observation 1 and does not grow, so it is not plasticity drifting during play.

**The regime is not what causes the bias, though.** Running the site frozen and
unreinforced — matching calibration exactly — left the bias in place and
slightly worse: mean **+318 px**, with 66% of estimates outside the reachable
range against 30% before. Matching the regime is still the right default,
because it removes an uncontrolled variable, but the honest reading is simpler:
a decoder carrying no signal produces an essentially arbitrary linear
projection of the current rates, and there is no reason for that projection to
be centred.

This is recorded and shown on the page, not corrected. Re-centring the estimate
at runtime would mean the decoder was reading something other than neural
firing, which is the one thing it must not do. The legitimate fixes are to
calibrate under the same regime the site runs in, or to run the site frozen;
both are experiment-design choices rather than code changes, so neither is
made here. It is not the reason the decoder fails to *decode* — it scores at the noise
floor on held-out probes collected in its own regime — but it is the reason the
paddle performs measurably worse than random rather than merely randomly. Any
future calibration must control for it.

Before claiming learned character selection, implement the held-out replay,
multi-seed comparison, frozen-weight and shuffled-feedback controls, and
prespecified statistical analysis described in [the model](model.md). This
repository currently provides a functioning experimental loop, not that
empirical result.
