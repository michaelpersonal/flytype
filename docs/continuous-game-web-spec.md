# Spec: Continuous FlyBreak Website

Status: draft for review  
Prepared: 2026-09-12

## Assumptions

1. This is a local website bound to `127.0.0.1`, not a public or cloud service.
2. The default launch uses the checksum-verified local MaleCNS dataset and
   retained graph. Fixture mode exists only behind an explicit `--fixture` flag,
   remains visibly labelled synthetic output, and is never selected as a
   fallback when MaleCNS cannot start.
3. "Continuous" means the server starts a fresh seeded episode after a wall is
   cleared or all balls are lost, until the operator pauses or stops it.
4. Neural decisions remain observation-bound. Each committed observation emits
   one continuous signed control in `[-1, 1]`; the browser may visually tween
   between committed states, but it may not invent controls or conceal latency.
5. The existing scratchpad console supplies the visual direction, but the
   maintained website source and assets move into this repository.
6. The taller geometry uses the current `y=12..104` field, a paddle at `y=92`,
   and the new 16 px circular ball, subject to measured retinal-visibility
   validation before it is described as suitable for MaleCNS control.

## Objective

Build a local web application that continuously runs FlyBreak episodes through
a sensory → retained MaleCNS graph → frozen population decoder → velocity loop.
The page must make a recording-ready game presentation while truthfully showing
whether each decision came from MaleCNS or fixture mode.

The operator can start, pause after the current committed observation, resume,
and stop. A completed episode is persisted before the next episode begins.

## Tech Stack

- Python 3.11+ and the existing `flytype` package.
- Python standard-library HTTP server initially; no new runtime dependency.
- Plain HTML, CSS, Canvas, and browser JavaScript maintained under
  `flytype/web/`.
- JSON status/control endpoints plus Server-Sent Events for committed state
  updates.
- Existing atomic state and alternating neural-checkpoint machinery.
- A versioned population-decoder artifact fitted on separate probe frames and
  checksum-locked into run provenance.

## Commands

```sh
# Fast, explicitly synthetic presentation mode
python -m flytype web --fixture --out runs/web-fixture --seed 7

# Real local MaleCNS mode
python -m flytype web --out runs/web-malecns --seed 7

# Explicitly rebuild the frozen decoder from MaleCNS probe observations
python -m flytype calibrate-play-decoder --out calibration/play-population.json

# Custom port; always loopback-only
python -m flytype web --fixture --port 8765 --out runs/web-fixture

# Tests
.venv/bin/pytest -q
```

The command prints the exact local URL. It never binds a non-loopback address.

## Project Structure

```text
flytype/
├── breakout.py          # geometry, exact circle collision, episode state
├── arena.py             # 320×180 sensory rendering
├── motor.py             # frozen neural position/motion decoder
├── play_session.py      # one persisted neural/game observation at a time
├── web_server.py        # local HTTP/SSE lifecycle and controls
└── web/
    ├── index.html       # recording-oriented semantic layout
    ├── app.css          # responsive 16:9 presentation styling
    └── app.js           # canvas rendering, controls, SSE updates
tests/
├── test_breakout.py
├── test_play_session.py
└── test_web_server.py
```

The scratchpad HTML becomes a reference artifact, not production source.

## Code Style

Python state transitions are explicit and return immutable browser payloads:

```python
def advance(self) -> dict:
    """Commit exactly one neural observation and return its public snapshot."""
    before = self.game.view()
    frame = render_arena(before, egocentric=self.settings.egocentric)
    neural = self.controller.observe(frame, self.feedback.pending)
    motor = self.motor.decode(neural["population_rates"])
    outcome = self.game.step_control(motor["control"])
    self._commit(outcome, neural)
    return self.public_snapshot()
```

- Existing formatting and naming conventions apply.
- Target text and correctness never enter `flytype/neural/`.
- Browser payloads contain presentation data, not controller access.
- Calibration may read labelled probe offsets. The frozen runtime decoder reads
  neural rates and its own previous estimate only; it never reads live ball
  coordinates, correctness, score, or outcome.
- Frontend CSS uses the existing dark graphite, teal, amber, and red console
  palette with responsive grid/flex children constrained against overflow.

## Testing Strategy

- Unit: exact circle/rectangle contacts, fractional paddle control, geometry
  bounds, population-decoder serialization, episode reset seeds, pause/stop
  state transitions, and public snapshot schema.
- Calibration: deterministic train/validation probes report held-out offset
  error and correlation and freeze neuron IDs, weights, normalization, gains,
  dataset identity, and source identity in the decoder artifact.
- Integration: fixture server runs multiple episodes, persists each committed
  tick, resumes after restart, and streams monotonic observation identifiers.
- HTTP: server rejects non-loopback binding, malformed controls, and unsupported
  methods; static paths cannot escape the packaged web root.
- Browser: at 1440×810 and 1920×1080, no overflow or clipped status text;
  start/pause/resume works; source mode is always visible.
- Full repository: `.venv/bin/pytest -q`.
- Optional real integration: `FLYTYPE_FULL_TEST=1` exercises one MaleCNS
  observation without requiring it in the default suite.

## Boundaries

### Always

- Use only rendered RGB as neural sensory input and only the frozen population
  decoder's signed control as the live action. DNp20 remains visible telemetry.
- Persist game state, feedback state, provenance, and alternating neural
  checkpoint before publishing a committed update or beginning another episode.
- Refuse resume on settings, geometry, decoder, dataset, or source mismatch.
- Label every snapshot and visible session as `malecns` or `fixture`.
- Keep detailed calibration and validation caveats in `docs/`.

### Ask first

- Add a third-party web framework or frontend build tool.
- Bind beyond `127.0.0.1`, add remote access, or add authentication.
- Change a frozen decoder after calibration, the sensory adapter, reinforcement
  populations, or plasticity rule.
- Delete or migrate an existing run directory.

### Never

- Script MaleCNS-labelled decisions, use an LLM to choose actions, correct the
  decoded control, or override it with hidden ball position.
- Select neurons or fit weights using gameplay correctness, outcomes, or score.
- Let browser-rendered presentation graphics replace the 320×180 frame actually
  supplied to the neural controller.
- Claim a biological fly, understanding, intent, pleasure, pain, consciousness,
  or validated learning.
- Accept credentials or post, follow, like, or otherwise act on X.

## Success Criteria

1. `python -m flytype web --fixture ...` opens a playable local page and runs at
   least three consecutive episodes without manual restart.
2. `python -m flytype web ...` uses only real MaleCNS observations and a
   checksum-locked population decoder, displaying honest compute latency.
3. Start, pause-after-current-step, resume, and stop are deterministic and safe.
4. A crash or restart resumes the last committed episode state; a completed
   episode is archived before the next seeded episode is initialized.
5. The visible ball is 16 px in new runs, stays circular at every viewport, and
   bounces only on true circle contact with the visible paddle or brick.
6. Replay/event payloads include ball, paddle, field, and brick geometry, so a
   viewer cannot silently render a different revision.
7. The world arena is dominant; neural offset/motion estimate, signed control,
   DNp20 telemetry, retinal input, CNS activity, and paddle evidence form one
   vertical rail at recording widths.
8. Fixture and MaleCNS modes are visually unmistakable and recorded in every
   event, snapshot, checkpoint provenance, and episode summary.
9. Held-out probe decoding and retinal coverage of the taller geometry are
   measured. The UI does not claim steering unless pooled play beats chance.
10. Paddle displacement is proportional to committed control in `[-1, 1]`;
    successive neural offset estimates supply a disclosed motion term. No live
    game coordinate enters that calculation.
11. The default test suite passes with no new external runtime dependency.

## Status against these criteria, 2026-09-12

Criteria 1-8, 10 and 11 are met. Criterion 5 is met with a 16 px ball, whose
retinal coverage was measured rather than assumed (mean 22.8 photoreceptors,
blind at 1.7% of positions; see [validation](validation.md)).

**Criterion 9 is met in the negative, which is the outcome that matters.**
Held-out probe decoding and retinal coverage were measured. The population
decoder does not beat its own shuffled-label control or a decoder that always
predicts the mean, so under the second half of criterion 9 the UI does not
claim steering: it displays the decoder's held-out error against both controls
and states plainly that the offset is not a demonstrated read of ball position.

Two defects found while measuring are recorded in [validation](validation.md)
rather than papered over: the decoder is calibrated frozen and unreinforced but
deployed plastic and reinforced, which puts a standing ~+276 px bias on the
estimate; and the default 0.18 s tick is unreachable in MaleCNS mode, where an
observation costs ~2 s of neural compute. Persisting the 7 MB checkpoint and
hashing it accounts for about 4 ms of that and is not worth changing.

Measured back to back after letting the network settle, the regime does not
affect the cost at all: frozen/unreinforced 2.00 s, frozen/reinforced 2.01 s,
plastic/unreinforced 2.03 s, all at ~431,000 spikes per observation. The only
lever on neural cost is the observation window, which scales it linearly
(500 ms -> 2.14 s, 250 ms -> 1.07 s on this host). An earlier revision of this
document attributed the cost to plasticity and reinforcement; that was measured
before the network had settled and is withdrawn.

## Resolved Decision

The website defaults to real MaleCNS. `--fixture` is explicit and is used only
for tests or a deliberately synthetic demonstration; failure to load MaleCNS
stops with an error instead of falling back to fixture output.
