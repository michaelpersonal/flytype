# FlyType

A MaleCNS connectome simulation selects characters to reproduce a supplied
target sentence through a fixed visual and neural interface.

**How it works:** a locally rendered 320×180 frame shows the target sentence,
the text typed so far, and two candidate characters (one correct, one
distractor) in a left key and a right key. It stimulates 3,335 brightness
inputs and 811 R8 color inputs in the retained **MaleCNS v1.0 graph: 166,700
neurons, 25.6 million connections**. A fixed neural readout — mean right minus
left DNp20 firing against a configured deadband — decodes LEFT, RIGHT or HOLD.
A correct selection appends the character; an incorrect one does not.

Correct selections schedule a reward pulse into 15 identified PAM11 dopamine
cells on the next observation; incorrect selections schedule an aversive pulse
into two PPL101 cells. A candidate memory rule changes existing KC-to-MBON
connections. These are engineered reinforcement signals, **not modeled pain
receptors**, and the network's understanding of language, spelling, or
Michael's X account is not modeled or claimed. [Model and evidence](docs/model.md).

FlyType never posts, follows, likes, or otherwise acts on X or any other
network service, and never accepts or stores account credentials.

## Run it

Python 3.11+, a C++17 compiler, macOS/Linux. Allow at least 10 GB free disk
(20 GB recommended) and network access for the ~1.1 GB MaleCNS dataset; 16 GB
RAM recommended. No GPU required.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m flytype prepare
python -m flytype verify
```

`prepare` prints the resolved data directory before downloading. Set
`FLYTYPE_DATA=/absolute/path` to use a dedicated data location instead of
`<repo>/data`; an interrupted download is safe to rerun.

```sh
python -m pytest -q
python -m flytype run --fixture --seed 42 --steps 2000
FLYTYPE_FULL_TEST=1 python -m pytest -q tests/test_neural_integration.py
python -m flytype run --target 'follow @michaelzsguo on x' --seed 42 --out runs/michael-follow
python -m flytype status --out runs/michael-follow
```

`--fixture` uses a deterministic scripted stand-in instead of MaleCNS, for
fast offline testing; every event it produces is tagged `"source": "fixture"`,
and every real-MaleCNS event is tagged `"source": "malecns"`. Ctrl-C or a
`STOP` file in the run directory stops cleanly after the current committed
observation; rerunning the same `--out` resumes from it. `--frozen` disables
plasticity for a control run; `--shuffle-feedback` independently randomizes
which reinforcement (reward/aversive) follows a scored attempt. Local state,
the latest stimulus frame, event log and resumable neural checkpoints live
under the run's `--out` directory (default `runs/default/`); nothing is
uploaded anywhere.

## Brick breaker

The same two-choice readout can drive something other than a keyboard. `play`
puts the decoded LEFT/RIGHT/HOLD on a brick-breaker paddle: one observation is
one tick, the paddle takes one step in the decoded direction, then the ball
moves. The network's only input is the rendered arena; it never receives the
ball position, the score or which direction is correct, and the paddle is never
nudged toward the ball.

```sh
python -m flytype play --out runs/breakout --seed 7 --decoder-baseline-obs 16
python tools/build_breakout_frames.py runs/breakout out/arena-frames.json
python tools/build_arena_audio.py out/arena-frames.json out/arena-clicks.wav
```

Two things about this task are worth knowing before reading a score.

**The playfield is a letterbox because the eye is.** The retina adapter maps
3,335 R1–R6 and 811 R8 photoreceptors across the frame very unevenly: roughly
two thirds land in the top third, and the bottom-left quadrant receives none at
all. A paddle drawn at a conventional screen bottom is invisible to this eye. The
playfield is therefore confined to `y=12..64`, which puts a mean of 25
photoreceptors on the ball instead of 1.9.

**`--decoder-baseline-obs N` centers the decision on the readout's own running
median.** The two DNp20 cells do not rest at equal rates, so the raw difference
is almost always positive and an uncentered readout pins the paddle to one wall.
The median is computed from the decoder's own past output only — never from the
arena, the outcome or the correct direction — so it removes the standing offset
without being able to supply an answer; exact ties decode as HOLD. `0` (the
default, and what every typing run used) disables it.

The run writes `summary.json`, whose `tracking_rate` is the share of non-HOLD
moves that went the way the ball actually lay, excluding moves made while the
paddle was already under it. **Chance is 0.5**, and `build_breakout_frames.py`
measures matched random and always-RIGHT agents on the same physics so the
episode can be read against them rather than against an impression.

## Limitations

The MVP demonstrates a closed sensorimotor loop, not evidence of learning: the
correct candidate is repeatedly shown against a distractor with its side
independently randomized each attempt, so the target sentence can complete
even under a fixed directional decoder bias with no plasticity at all. See
[docs/model.md](docs/model.md) for what is and is not modeled,
[docs/validation.md](docs/validation.md) for the checks actually performed
(including an observed directional bias in a real run), and
[docs/calibration.md](docs/calibration.md) for how sensory/decoder parameters
were chosen.

## License and attribution

MIT License (see `LICENSE`). Forked from
[nftechie/stonkfly](https://github.com/nftechie/stonkfly) at commit
`78ef3e05ab0fa086032098558d893667068944a0`; trading, Coinbase and AgentKit
code was removed and a character-selection task was added in its place. See
[THIRD_PARTY.md](THIRD_PARTY.md) for full upstream and dataset attribution,
including the MaleCNS v1.0 CC BY license.
