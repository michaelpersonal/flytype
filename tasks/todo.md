# Continuous FlyBreak Task List

> **Superseded 2026-09-12 by [`fix-plan.md`](fix-plan.md).** The acceptance
> boxes below were never ticked and three of them did not hold when checked:
>
> * Task 2 claimed "existing `flytype play` uses the same session code". It did
>   not and still does not; `cli._play_loop` remains a separate loop. That is
>   now a deliberate, documented choice rather than an unmet claim.
> * Task 6 cited `FLYTYPE_FULL_TEST=1 pytest tests/test_neural_integration.py`
>   as verifying one real MaleCNS observation reaching the browser. No such
>   test existed. One does now.
> * Task 6's "geometry coverage evidence" and the decoder quality figures in
>   `docs/validation.md` were not reproducible: no calibration artifact
>   survived. Both have been re-measured and the old figures withdrawn.
>
> Kept for the record of what was attempted. `fix-plan.md` carries current
> state.

Implementation status as originally written: the geometry, persisted continuous
session, loopback website, responsive console, and real MaleCNS smoke path are
implemented. The full suite passes (`75 passed, 5 skipped`). The remaining open
research question is decoder quality: the default held-out probe result is
documented in `docs/validation.md` and is not presented as validated gameplay.

## Task 1: Lock geometry and exact collision behavior

**Description:** Finish the smaller circular ball and ensure render, physics,
state, event, and replay geometry cannot drift.

**Acceptance criteria:**
- [ ] New runs use a 16 px circular ball in the taller field.
- [ ] Paddle and brick contacts use circle/rectangle intersection.
- [ ] Complete geometry appears in game views, saved geometry, and replay data.

**Verification:**
- [ ] `.venv/bin/pytest -q tests/test_breakout.py`
- [ ] Render a catch and corner miss for visual inspection.

**Dependencies:** None  
**Files likely touched:** `flytype/breakout.py`, `flytype/arena.py`,
`tools/build_breakout_frames.py`, `tests/test_breakout.py`  
**Estimated scope:** Medium

## Task 2: Extract one persisted observation into a session object

**Description:** Move the `play` loop's single-observation transaction into a
reusable object without changing the sensory/neural/action boundary.

**Acceptance criteria:**
- [ ] `advance()` performs exactly one RGB observation and decoded action.
- [ ] Checkpoint, event, state, and public snapshot share one observation ID.
- [ ] Existing `flytype play` uses the same session code.

**Verification:**
- [ ] `.venv/bin/pytest -q tests/test_play_session.py tests/test_breakout.py`
- [ ] Existing fixture CLI reproducibility test passes.

**Dependencies:** Task 1  
**Files likely touched:** `flytype/play_session.py`, `flytype/cli.py`,
`flytype/state.py`, `tests/test_play_session.py`  
**Estimated scope:** Medium

## Task 3: Add continuous episode rollover and resume

**Description:** Preserve the controller across deterministic game resets and
persist the episode index plus completed summaries.

**Acceptance criteria:**
- [ ] A completed episode is archived before episode `N+1` starts.
- [ ] Seeds are deterministic and episode IDs are monotonic.
- [ ] Restart resumes the last committed episode/controller state exactly.

**Verification:**
- [ ] Fixture test runs and resumes across at least three episodes.
- [ ] Corrupt or mismatched state is refused.

**Dependencies:** Task 2  
**Files likely touched:** `flytype/play_session.py`, `flytype/state.py`,
`tests/test_play_session.py`  
**Estimated scope:** Medium

## Task 4: Add loopback HTTP/SSE command and controls

**Description:** Serve packaged assets, immutable state, committed SSE updates,
and authenticated local pause/resume/stop controls.

**Acceptance criteria:**
- [ ] `flytype web` defaults to MaleCNS; `--fixture` is explicit.
- [ ] The server binds only `127.0.0.1` and prints its URL.
- [ ] Controls are safe while an observation is computing.

**Verification:**
- [ ] `.venv/bin/pytest -q tests/test_web_server.py`
- [ ] Cross-origin/simple malformed control requests are rejected.

**Dependencies:** Task 3  
**Files likely touched:** `flytype/web_server.py`, `flytype/cli.py`,
`flytype/__main__.py`, `tests/test_web_server.py`  
**Estimated scope:** Medium

## Task 5: Build the live recording-oriented frontend

**Description:** Move the approved console direction into maintained static
assets driven by SSE snapshots rather than replay files.

**Acceptance criteria:**
- [ ] Arena dominates; diagnostics are a vertical rail at recording widths.
- [ ] Source mode, compute state, action, outcome, and episode are always clear.
- [ ] Start/pause/resume/stop work without reloading.

**Verification:**
- [ ] Fixture browser run advances live and rolls episodes.
- [ ] Visual QA at 1440×810 and 1920×1080 has no clipped text or overflow.

**Dependencies:** Task 4  
**Files likely touched:** `flytype/web/index.html`, `flytype/web/app.css`,
`flytype/web/app.js`, `pyproject.toml`  
**Estimated scope:** Medium

## Task 6: Verify real-mode startup and geometry evidence

**Description:** Exercise the default MaleCNS path and document what the taller
field and smaller ball do to retinal sampling without claiming successful play.

**Acceptance criteria:**
- [ ] Default web startup verifies and loads MaleCNS or fails explicitly.
- [ ] One real observation reaches the browser with `source=malecns`.
- [ ] Geometry coverage evidence and caveats are documented.

**Verification:**
- [ ] `FLYTYPE_FULL_TEST=1 .venv/bin/pytest -q tests/test_neural_integration.py`
- [ ] Inspect one real browser snapshot and matching event/checkpoint IDs.

**Dependencies:** Tasks 4-5  
**Files likely touched:** `tests/test_neural_integration.py`,
`docs/validation.md`, `docs/model.md`  
**Estimated scope:** Medium

## Task 7: Complete regression and operator documentation

**Description:** Run the complete suite, tighten failure behavior, and document
the two launch modes without expanding the README.

**Acceptance criteria:**
- [ ] Full tests pass and no new runtime dependency is introduced.
- [ ] README contains only concise launch instructions.
- [ ] Detailed web/session/calibration caveats live under `docs/`.

**Verification:**
- [ ] `.venv/bin/pytest -q`
- [ ] `git diff --check`
- [ ] Fresh local fixture launch and stop from documented commands.

**Dependencies:** Tasks 1-6  
**Files likely touched:** `README.md`, `docs/continuous-game-web-spec.md`,
`tasks/todo.md`  
**Estimated scope:** Small
