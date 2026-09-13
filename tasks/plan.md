# Implementation Plan: Continuous FlyBreak Website

## Overview

Add a loopback-only web command that continuously advances a MaleCNS population-
decoded Breakout loop, persists every committed observation, starts new
deterministic episodes without resetting neural state, and streams truthful
snapshots to a recording-oriented browser UI. Real MaleCNS remains the default;
fixture mode requires `--fixture`.

Specification: `docs/continuous-game-web-spec.md`

## Architecture Decisions

- One worker thread exclusively mutates the game, feedback schedule, decoder,
  and neural controller. HTTP threads never call neural or physics code.
- A `ContinuousPlaySession` exposes `advance()`, `snapshot()`, pause/resume, and
  stop. `advance()` does not publish until event, checkpoint, and state are
  durably committed.
- Neural state persists across episode boundaries. Only `BreakoutGame` resets,
  using `base_seed + episode_index`; the completed episode is archived before
  the next episode is made visible.
- The server uses Python's standard-library `ThreadingHTTPServer` and SSE. This
  avoids adding a framework dependency to a small local-only service.
- The server binds only `127.0.0.1`. Mutating requests require an unguessable
  per-process token delivered to the same-origin page.
- The browser renders world geometry from the server snapshot and displays the
  exact 320×180 retinal frame used for the corresponding neural decision.
- A calibration-only pipeline fits relative horizontal ball offset from a
  declared descending-neuron population on probe frames. Runtime derives motion
  from successive neural estimates and maps offset plus motion to continuous
  signed paddle control. The frozen artifact is provenance-locked.
- Smoothness is presentational only: the frontend may tween between two
  committed positions, but actions and outcomes update only on server events.
- MaleCNS startup failure is fatal. There is no automatic fixture fallback.

## Dependency Graph

```text
geometry + population decoder
          │
          ▼
single-step persisted play session
          │
          ├──────────────┐
          ▼              ▼
continuous episode   HTTP/SSE contract
coordinator              │
          └───────┬──────┘
                  ▼
        recording-oriented UI
                  │
                  ▼
       browser QA + documentation
```

## Task List

### Phase 1: Truthful game/session foundation

- [ ] Task 1: Lock geometry and exact collision behavior.
- [ ] Task 2: Build and freeze the population position/motion decoder.
- [ ] Task 3: Extract one persisted graded-control observation into a session.
- [ ] Task 4: Add continuous episode rollover and resume.

### Checkpoint: Foundation

- [ ] Focused physics/session tests pass.
- [ ] Existing `flytype play` behavior remains compatible.
- [ ] State cannot be published before its checkpoint is committed.
- [ ] Review persisted files from a two-episode fixture run.

### Phase 2: Local website vertical slice

- [ ] Task 5: Add the loopback HTTP/SSE command and controls.
- [ ] Task 6: Build the live recording-oriented frontend.

### Checkpoint: End-to-end fixture path

- [ ] Launch command prints a working local URL.
- [ ] Browser shows live fixture decisions with an unavoidable FIXTURE label.
- [ ] Pause, resume, stop, episode rollover, and restart-resume work.
- [ ] No horizontal or vertical overflow at recording viewports.

### Phase 3: MaleCNS integration and evidence

- [ ] Task 7: Verify real-mode calibration/startup and document evidence.
- [ ] Task 8: Complete regression, browser, and documentation checks.

### Checkpoint: Complete

- [ ] `.venv/bin/pytest -q` passes.
- [ ] Optional full integration completes one MaleCNS observation.
- [ ] Real mode never falls back to fixture mode.
- [ ] All success criteria in the specification are accounted for.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| A MaleCNS observation is much slower than a visual frame | High | Show compute state and last committed observation; never generate substitute actions |
| HTTP requests race neural/checkpoint writes | High | Single-writer worker; immutable snapshots published only after commit |
| Episode rollover loses the last state | High | Archive summary and commit rollover state before starting the next tick |
| Geometry becomes visually different from physics | High | Carry complete geometry in every snapshot/event and share exact circle collision code assumptions |
| A webpage on another origin controls localhost | Medium | Loopback-only bind, random control token, same-origin assets, no permissive CORS |
| Taller field reduces retinal signal | High | Measure coverage and document it; do not claim steering without pooled evidence |
| Existing uncommitted work is overwritten | High | Patch only overlapping lines deliberately and preserve unrelated changes |

## Open Questions

None. MaleCNS is the confirmed default; fixture mode is explicit only.
