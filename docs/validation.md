# Validation status

Recorded during implementation on 2026-09-11 in a real macOS environment with
network access. **The full MaleCNS v1.0 release was actually downloaded and
compiled** (not skipped as unavailable): 166,700 neurons, 25,582,938 directed
connections, 124,177,617 synaptic contacts, matching the committed checksum
locks exactly.

Final local result: **46 tests passed**, including the opt-in full-connectome
tests (`FLYTYPE_FULL_TEST=1`). A real MaleCNS run (below) produced stimulus
frames, neural-only decoded actions, delayed reinforcement, valid checkpoints,
and resumed correctly from committed state.

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
invisible at 28 of 42 sampled positions. Confining the playfield to `y=12..64`
and enlarging the ball to 10 px raised this to a mean of **25.5** over 2,284
real ball positions, blind at 4.7%. This also applies to the typing display,
whose key band extends well below the sampled region.

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

**Episodes run so far** (MaleCNS, seed 7, `--decoder-baseline-obs 16`):

| view | ticks | scored moves | tracking | paddle hits | balls lost |
|---|---|---|---|---|---|
| fixed (`runs/breakout-fixedview`) | 257 | 182 | 0.473 | 1 | 3 |
| egocentric (`runs/breakout`) | 177 | 110 | 0.545 | 0 | 3 |

Neither is distinguishable from chance on its own: the egocentric episode is
0.94 SE above 0.5, exact two-sided binomial p = 0.39. **No claim is made that
the network plays this game.** Use `tools/analyze_play.py` to pool episodes and
get the exact test; additional seeds are the only way to separate a small real
effect from noise, and the effect predicted by the probe above is small.

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

Before claiming learned character selection, implement the held-out replay,
multi-seed comparison, frozen-weight and shuffled-feedback controls, and
prespecified statistical analysis described in [the model](model.md). This
repository currently provides a functioning experimental loop, not that
empirical result.
