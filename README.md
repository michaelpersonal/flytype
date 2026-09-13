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

The legacy `play` command puts the decoded LEFT/RIGHT/HOLD on a brick-breaker
paddle. The recording-ready continuous website uses a separate frozen
MaleCNS population decoder: each neural observation produces a signed control
in `[-1, 1]`, with a disclosed position/motion estimate.

```sh
python -m flytype play --out runs/breakout --seed 7 --decoder-baseline-obs 16
python tools/build_breakout_frames.py runs/breakout out/arena-frames.json
python tools/build_arena_audio.py out/arena-frames.json out/arena-clicks.wav
```

Run the continuous local website with the verified MaleCNS dataset:

```sh
python -m flytype web --out runs/web-malecns
```

It prints a loopback URL, persists state and checkpoints under the run
directory, and rolls into a fresh episode after a loss or clear. Use
`--fixture` only for an explicitly synthetic fast demo.

**On pace.** One observation is one tick, and a tick costs about two seconds of
MaleCNS compute on this host, so the game advances in real time no faster than
the connectome runs. Three things affect what you see, and only the last one
changes the neural cost:

* The site runs at twice the legacy `play` pace (`--ball-descent-ticks 6`
  with `--paddle-speed 52`), so the ball crosses the field in ~11 s rather than
  ~22 s. A tracker can still clear the wall and a blind agent still cannot;
  `tests/test_breakout.py` pins that. These are not frame conditions, so
  changing them needs no recalibration.
* The browser interpolates between committed states across the measured
  observation interval, so the ball travels continuously instead of jumping and
  waiting. Presentation only: committed positions are unchanged and nothing is
  extrapolated past them.
* `--neural-ms` is the dominant lever and scales the cost linearly. Measured on
  this host, per observation and as the time for the ball to cross the field:

  | window | compute / observation | field crossing |
  | --- | --- | --- |
  | 500 ms | 2.02 s | 12.1 s |
  | 250 ms | 1.01 s | 6.1 s |
  | 150 ms | 0.60 s | 3.6 s |
  | 100 ms | 0.41 s | 2.4 s |

  It is a frame condition, so changing it requires re-running
  `calibrate-play-decoder` at the same window; `web` refuses a decoder fitted
  at a different one. A shorter window means fewer spikes per rate estimate,
  which would be a real cost if the readout carried a signal. It does not:
  recalibrated at 100 ms the held-out error is 161.8 px against 159.6 px for
  shuffled labels, statistically the same as at 500 ms. A watchable demo
  therefore costs nothing measurable here:

  ```sh
  python -m flytype calibrate-play-decoder \
      --out calibration/play-population-100ms.json --neural-ms 100 --repeats 5
  python -m flytype web --out runs/live --neural-ms 100 \
      --motor-decoder calibration/play-population-100ms.json
  ```

Plasticity and reinforcement do **not** affect the cost: measured back to back
after the network settles, every regime costs 2.00-2.03 s per observation.

The continuous site drives the paddle from a **frozen population decoder**, not
the two-cell DNp20 readout — that is recorded on every observation as
`dnp20_action` and never touches the control. The decoder is fitted once,
offline, on labelled visual probes:

```sh
python -m flytype calibrate-play-decoder --out calibration/play-population.json
```

`web` builds this automatically on first launch if it is missing, which costs a
few hundred real MaleCNS observations (a few minutes) before the page opens.
The artifact records the exact viewing conditions it was fitted under — view,
paddle width, ball size, field bounds, observation window — and `web` refuses
to run if any of them differ, because a decoder fitted on paddle-centred frames
reports confident nonsense on any other framing.

**Read the reported error before reading the game.** Calibration prints its
held-out probe error beside two controls: the same pipeline fitted to shuffled
labels, and a decoder that always predicts the mean. On the current geometry it
does not beat either of them, so the site says so on screen and no claim of
steering is made. See [docs/validation.md](docs/validation.md).

Two things about this task are worth knowing before reading a score.

**The playfield is a letterbox because the eye is.** The retina adapter maps
3,335 R1–R6 and 811 R8 photoreceptors across the frame very unevenly: roughly
two thirds land in the top third, and the bottom-left quadrant receives none at
all. A paddle drawn at a conventional screen bottom is invisible to this eye. The
playfield is therefore confined to `y=12..104`, which puts a mean of 22.8
photoreceptors on the 16 px ball instead of the 1.9 a conventional full-frame
arena managed, and leaves it invisible at 1.7% of positions instead of 67%.

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
