# FlyType MVP — Implementation and Setup Specification

Status: implementation handoff  
Prepared: 2026-09-11  
Target upstream: `nftechie/stonkfly` commit `78ef3e05ab0fa086032098558d893667068944a0`

## 1. Objective

Build a local, reproducible demonstration in which the MaleCNS v1.0 fruit-fly
connectome simulation selects characters until it produces this exact text:

```text
follow @michaelzsguo on x
```

The demonstration must make the closed loop easy to understand:

```text
two visible character choices
        ↓
MaleCNS visual input and neural simulation
        ↓
fixed left/right neural decoder
        ↓
selected character
        ↓
reward or aversive dopamine stimulus on the next observation
```

This is a local simulation. It does not require a hosted model, GPU, Coinbase,
an X account, or any external API after the MaleCNS data has been downloaded.

## 2. Required Claim Language

Use this description in the README and UI:

> A MaleCNS connectome simulation selects characters to reproduce a supplied
> target sentence through a fixed visual and neural interface.

Do not claim that:

- a biological fly is being used;
- the fly understands language or Michael's X account;
- the fly independently chose the sentence;
- the model has learned spelling;
- dopamine activity establishes pleasure, pain, consciousness, or intent;
- useful learning has occurred without the controls described in section 12.

The target sentence is supplied by the program. The connectome controls the
left/right character selections through an engineered decoder.

## 3. Scope

### In scope

- Start from the Stonkfly repository and retain its MaleCNS neural engine.
- Remove all market, Coinbase, wallet, order, and monetary-risk functionality.
- Render a 320×180 RGB stimulus containing the current text and two character
  choices.
- Decode MaleCNS activity into `LEFT`, `RIGHT`, or `HOLD`.
- Append a character only when the selected choice matches the next target
  character.
- Deliver reward or aversive reinforcement during the next neural observation.
- Persist events, progress, configuration, neural checkpoints, and the latest
  stimulus frame locally.
- Provide deterministic fixture mode for fast testing without the full graph.
- Provide a real MaleCNS mode using the checksum-verified graph.
- Support interruption and safe resume from the last committed character.

### Non-goals

- Posting, following, liking, or performing any action on X.
- Generating arbitrary prose or Python code.
- A full virtual keyboard in the first release.
- An LLM or language model anywhere in the selection loop.
- A biological-neuron interface.
- A cloud service, public website, livestream, or hosted inference API.
- Female FlyWire support.
- Demonstrating statistically significant learning.
- Video production or social-post packaging.

## 4. Dataset Decision

Use **MaleCNS v1.0**, not the female FlyWire connectome.

Reasons:

- Stonkfly already imports and checksum-locks MaleCNS v1.0.
- Its retained graph has 166,700 neurons and 25,582,938 directed connections.
- Its retinal projection and neuron annotations are MaleCNS-specific.
- Its dopamine and memory circuits are already identified and tested.
- Switching to the female graph would require a new importer, retinal mapping,
  neuron correspondence work, graph locks, decoder validation, and scientific
  revalidation. That is a separate project.

Do not silently substitute another dataset or copy MaleCNS neuron mappings onto
a female graph.

## 5. Machine Requirements and Storage

Minimum development environment:

- macOS or Linux;
- Python 3.11;
- a C++17 compiler (`clang++` or `g++`);
- 16 GB RAM recommended;
- 10 GB free disk space minimum before preparation;
- 20 GB free disk space recommended for source data, normalized data, compiled
  arrays, checkpoints, logs, and temporary preparation files;
- network access for the initial dataset download.

The three upstream MaleCNS Feather inputs total approximately 1.11 GB as of
2026-09-11. Preparation creates additional normalized and compiled copies, so
the implementation must check available space rather than budgeting only for
the downloads.

No GPU is required. No process needs to stay running when an experiment is not
active.

## 6. Repository and Licensing Strategy

Create the implementation as a separate code repository. Suggested location:

```text
/Users/zhisongguo/code/flytype
```

Start from the pinned Stonkfly commit:

```bash
git clone https://github.com/nftechie/stonkfly.git flytype
cd flytype
git checkout 78ef3e05ab0fa086032098558d893667068944a0
```

Then rename the Python package from `stonkfly` to `flytype` and preserve useful
Git history where practical.

Requirements:

- Retain Stonkfly's MIT copyright and license notice.
- Attribute MaleCNS v1.0 and comply with its CC BY license.
- Keep a `THIRD_PARTY.md` file with upstream source, pinned commit, dataset URL,
  and scientific references.
- Do not commit downloaded MaleCNS data, compiled graph arrays, experiment runs,
  neural checkpoints, or temporary files.

## 7. Package Layout

The final repository should approximately contain:

```text
flytype/
├── __init__.py
├── __main__.py
├── cli.py
├── config.py
├── data.py
├── task.py
├── display.py
├── decoder.py
├── feedback.py
├── state.py
└── neural/
    ├── brain.py
    ├── circuit.py
    ├── common.py
    ├── connectome.py
    ├── controller.py
    ├── datasets.json
    ├── kernel.cpp
    ├── prepare.py
    ├── rule.py
    ├── sensory.py
    ├── state.py
    ├── transmitters.py
    ├── visual.py
    └── lock JSON files
tests/
├── test_task.py
├── test_decoder.py
├── test_feedback.py
├── test_state.py
├── test_fixture_run.py
└── test_neural_integration.py
```

Delete the inherited trading modules after their useful generic logic has been
extracted:

- `actions.py`
- `broker.py`
- `market.py`
- `risk.py`
- trading-specific ledger code

Remove `coinbase-agentkit`, `coinbase-advanced-py`, and `python-dotenv` unless a
remaining non-secret configuration use is explicitly justified.

## 8. User-Visible Behavior

The default target is exactly:

```text
follow @michaelzsguo on x
```

Each observation displays:

- a small `TARGET` label and the complete target sentence;
- a `TYPED` line showing accepted characters plus a cursor;
- one candidate character in a large left key;
- one candidate character in a large right key;
- the latest decoded action;
- the latest reinforcement state: `NONE`, `REWARD`, or `AVERSIVE`.

For every unresolved target character:

1. Construct one correct candidate and one distractor.
2. Randomly assign the candidates to left and right using the run's seeded RNG.
3. Render the frame.
4. Advance the neural simulation.
5. Decode `LEFT`, `RIGHT`, or `HOLD` using only neural spike counts.
6. On `HOLD`, record the observation and do not score an attempt.
7. On a correct selection, append the character and schedule `REWARD` for the
   next observation.
8. On an incorrect selection, append nothing and schedule `AVERSIVE` for the
   next observation.
9. Generate a fresh distractor and randomized placement for the next attempt.
10. Atomically persist state and the neural checkpoint.

The distractor alphabet for the default task is:

```text
abcdefghijklmnopqrstuvwxyz @
```

The distractor must differ from the correct character. Spaces must be displayed
as a visible `SPACE` key while the accepted output still contains an actual
space.

On completion, print and persist:

```text
follow @michaelzsguo on x
```

The application must never send that text to X or any other network service.

## 9. Neural Decoder

Reuse the MaleCNS DNp20 populations currently identified by Stonkfly.

For each observation, calculate:

```text
difference_hz = mean(right_DNp20_hz) - mean(left_DNp20_hz)
```

Default mapping:

```text
difference_hz > deadband   → RIGHT
difference_hz < -deadband  → LEFT
otherwise                  → HOLD
```

Start with a small, declared deadband such as `0.1 Hz`. It must be configurable
and recorded in the run manifest. Do not use target correctness, target text, or
reward state to override the neural proposal or choose a replacement action.

DNpe017 gating may be logged for comparison, but it is not required for the MVP
selection decision. Adding a gate is a future experiment and must not be enabled
silently.

The fixture decoder may emit scripted actions solely in `--fixture` mode. Every
event must clearly record whether its source was `fixture` or `malecns`.

## 10. Reinforcement Timing

Reuse Stonkfly's identified circuits and candidate memory rule:

- correct selection → stimulate PAM11 reward cells;
- incorrect selection → stimulate PPL101 aversive cells;
- `HOLD` or initial observation → no reinforcement.

Reinforcement resulting from observation `N` must be delivered during
observation `N+1`. Do not retroactively modify the spikes that produced the
selection.

Use the inherited defaults unless a recorded calibration justifies a change:

- neural window: 500 ms;
- neural bin: at most 10 ms;
- reinforcement pulse: 200 ms;
- pulse current: inherited Stonkfly value;
- learning enabled by default in normal MaleCNS runs;
- `--frozen` disables plasticity for a control run.

Correctness may control only the next reinforcement stimulus. It must not edit
synaptic weights directly.

## 11. Persistence and Outputs

Default run directory:

```text
runs/default/
```

Required artifacts:

```text
runs/default/
├── config.json
├── provenance.json
├── state.json
├── events.jsonl
├── latest-input.png
├── final.txt
├── brain-0.npz
├── brain-1.npz
└── worker.lock
```

Requirements:

- Use a single-worker lock per run directory.
- Store RNG state or enough deterministic state to resume candidate placement.
- Commit the neural checkpoint and accepted text before advancing to another
  character.
- Use alternating checkpoint slots so a crash cannot destroy the last committed
  state.
- Store SHA-256 hashes for checkpoints and source/config provenance.
- Refuse resume if the checkpoint hash, target, decoder configuration, dataset,
  or source signature differs.
- `events.jsonl` must include observation number, target index, candidates and
  their sides, neural rates, decoded action, correctness, delivered stimulus,
  scheduled next stimulus, accepted text, compute time, and mode.
- Write `final.txt` only after exact target completion.

No event or checkpoint file may contain account credentials because this project
must not accept account credentials at all.

## 12. Controls and Interpretation

The MVP demonstrates a closed sensorimotor loop, not useful language learning.
Because the correct candidate is repeatedly presented against a distractor and
its side is randomized, the phrase may complete even with a fixed directional
bias.

Support these experiment variants:

- normal plasticity;
- `--frozen` weights;
- `--shuffle-feedback`, which independently shuffles reward/aversive feedback;
- deterministic fixture mode for software validation.

Report at least:

- total observations;
- committed left/right selections;
- holds;
- correct and incorrect attempts;
- attempts per accepted character;
- completion time;
- changed plastic-edge count;
- decoder directional bias.

Do not label a normal run as learning evidence. A later learning study would need
multiple seeds, independent initial states, held-out trials, frozen-weight and
shuffled-feedback controls, and a prespecified statistical analysis.

## 13. CLI Contract

Required commands:

```bash
python -m flytype prepare
python -m flytype verify
python -m flytype run
python -m flytype run --fixture --seed 42
python -m flytype run --frozen --seed 42
python -m flytype run --shuffle-feedback --seed 42
python -m flytype status --out runs/default
```

Required `run` options:

```text
--target TEXT
--out PATH
--seed INTEGER
--steps INTEGER
--max-attempts-per-character INTEGER
--neural-ms FLOAT
--decoder-deadband-hz FLOAT
--fixture
--frozen
--shuffle-feedback
--fast
```

Rules:

- Default target: `follow @michaelzsguo on x`.
- `--steps 0` means continue until completion or a stop condition.
- `--fast` is permitted only in fixture/local simulation mode and removes wall
  delays; it must not alter neural time.
- Exceeding `--max-attempts-per-character` must halt transparently rather than
  inserting the correct character.
- A `STOP` file in the run directory must stop cleanly after the current atomic
  observation commit.

## 14. Setup Procedure

### macOS prerequisites

Verify tools:

```bash
python3.11 --version
clang++ --version
df -h .
```

If the compiler is missing, the human operator installs Xcode Command Line
Tools. The implementation agent must not automate OS-level installation.

### Python environment

```bash
cd /Users/zhisongguo/code/flytype
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

### Data location

Support a dedicated absolute data path:

```bash
export FLYTYPE_DATA=/Users/zhisongguo/data/flytype-malecns-v1
python -m flytype prepare
python -m flytype verify
```

If `FLYTYPE_DATA` is unset, use `<repo>/data`. Never search parent directories
for datasets or credentials. Print the resolved data directory before downloads
begin.

Before downloading, `prepare` must:

1. Resolve the target data directory.
2. Check that at least 10 GB is available.
3. Print the expected source download size and conservative workspace budget.
4. Download to `.partial` files.
5. Verify the committed source hashes.
6. Atomically rename verified downloads.
7. Build and verify the normalized graph and native kernel.

An interrupted download must be safe to rerun.

### Smoke tests

```bash
python -m pytest -q
python -m flytype run --fixture --seed 42 --steps 2000
FLYTYPE_FULL_TEST=1 python -m pytest -q tests/test_neural_integration.py
```

Then run the real demonstration:

```bash
python -m flytype run \
  --target 'follow @michaelzsguo on x' \
  --seed 42 \
  --out runs/michael-follow
```

## 15. Implementation Phases

### Phase 1 — Fork hygiene and dependency removal

- Pin and record the upstream Stonkfly commit.
- Rename the package and environment variable to `flytype` / `FLYTYPE_DATA`.
- Remove Coinbase and trading modules and dependencies.
- Preserve the neural tests before changing behavior.
- Add ignore rules for all downloaded/generated data and run artifacts.

### Phase 2 — Character task and fixture mode

- Implement target state, candidate generation, seeded side randomization, and
  correct/incorrect scoring.
- Implement fixture decoder and the complete CLI loop without MaleCNS.
- Add deterministic unit and end-to-end tests.

### Phase 3 — Visual adapter and real neural decoder

- Render legible 320×180 frames using Pillow.
- Connect frames to the inherited `VisualMemoryBrain`.
- Implement the fixed DNp20 decoder and detailed event logging.
- Confirm that target state never enters the neural decoder.

### Phase 4 — Reinforcement and checkpointing

- Deliver delayed PAM11/PPL101 stimulation.
- Connect inherited plasticity, frozen control, and shuffled-feedback control.
- Implement alternating checkpoints, hashes, locking, and strict resume.

### Phase 5 — Calibration and documentation

- Run fixture and full-graph integration tests.
- Check whether the default frame produces nonzero visual and DNp20 activity.
- If the decoder remains entirely `HOLD`, adjust only declared sensory display
  parameters or the prespecified deadband. Never tune based on which side is
  correct.
- Record calibration decisions in `docs/calibration.md`.
- Finish README, scientific caveats, licensing, and reproducible demo commands.

## 16. Verification Requirements

### Unit tests

Tests must establish that:

- the target advances only on a correct committed selection;
- incorrect choices do not alter accepted text;
- distractors are never equal to the correct character;
- seeded candidate placement is deterministic;
- spaces render as `SPACE` but persist as literal spaces;
- decoder output depends only on supplied neural counts and configured deadband;
- reinforcement is delayed exactly one observation;
- fixture mode is unmistakably identified in every event;
- resume restores target index, RNG behavior, pending reinforcement, and brain
  checkpoint;
- incompatible resume configuration is rejected;
- completion requires exact target equality.

### Integration tests

The full-graph test must verify:

- MaleCNS release and graph checksums;
- 166,700 retained neurons;
- 25,582,938 retained directed connections;
- visual stimulation produces neural activity;
- reward stimulation produces PAM11 spikes;
- aversive stimulation produces PPL101 spikes;
- checkpoint restore reproduces the committed state;
- frozen mode does not change plastic weights.

### Manual visual check

Open `latest-input.png` and confirm that:

- target and typed text are readable;
- the two choices are visually separated and balanced;
- left/right placement is unambiguous;
- `SPACE` is understandable;
- no text is clipped at 320×180;
- the display does not leak which candidate is correct through color, brightness,
  size, or position styling.

## 17. Done When

The implementation is complete only when all of the following are true:

1. `python -m pytest -q` exits successfully.
2. `python -m flytype verify` reports MaleCNS v1.0, 166,700 neurons,
   25,582,938 directed connections, and verified arrays.
3. `python -m flytype run --fixture --seed 42 --steps 2000` produces
   `final.txt` containing exactly `follow @michaelzsguo on x`.
4. `FLYTYPE_FULL_TEST=1 python -m pytest -q tests/test_neural_integration.py`
   exits successfully on a prepared machine.
5. A real MaleCNS run produces stimulus frames, neural-only decoded actions,
   delayed reinforcement events, valid checkpoints, and resumable state without
   invoking trading, X, an LLM, or a scripted action override.
6. A real MaleCNS run completes the target or halts transparently at its declared
   attempt limit; it must never silently force a correct character.
7. `latest-input.png` passes the manual visual checks in section 16.
8. The README contains setup, disk/RAM guidance, exact run commands, the required
   claim language, limitations, upstream attribution, and license information.
9. The repository contains no downloaded MaleCNS files, credentials, account
   identifiers, neural checkpoints, or run logs in Git history.

## 18. Agent Handoff Instructions

The implementing agent should begin by reading:

1. this specification;
2. the workspace or target-repository `AGENTS.md` files;
3. Stonkfly's `AGENTS.md`, `README.md`, `docs/model.md`, and
   `docs/validation.md` at the pinned commit;
4. `stonkfly/neural/controller.py`, `brain.py`, `visual.py`, `prepare.py`, and
   the neural tests.

The agent may refine internal module boundaries, but must preserve the scope,
neural-only action rule, dataset decision, safety boundaries, claim language,
verification commands, and `Done When` contract above.

