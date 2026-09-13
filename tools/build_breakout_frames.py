"""Build the arena replay's frames.json from a completed `flytype play` run.

Reconstructs the exact frame shown at each observation, pairs it with that
observation's recorded neural statistics and outcome, and measures a matched
random-action baseline on the same physics so the episode can be read against
chance rather than against an impression.

    python tools/build_breakout_frames.py runs/breakout out/arena-frames.json
"""

import base64
import io
import json
import random
import statistics
import sys
from pathlib import Path

from PIL import Image

from flytype.arena import render_arena
from flytype.breakout import BreakoutGame, new_game_rng
from flytype.config import Settings

BASELINE_EPISODES = 200


def _png(frame):
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def baselines(settings):
    """Run the same arena under action policies that cannot see the ball.

    These are the numbers the fly's episode has to beat to mean anything. The
    random policy is the honest chance level; always-RIGHT is what a decoder
    with an uncorrected side bias produces.
    """
    out = {}
    for name, pick in (
        ("random", lambda r: r.choice(["LEFT", "RIGHT", "HOLD"])),
        ("always_right", lambda r: "RIGHT"),
    ):
        runs = []
        for seed in range(BASELINE_EPISODES):
            game = BreakoutGame(settings, new_game_rng(seed + 10_000))
            rng = random.Random(seed * 7919 + 11)
            while not game.done and game.tick < 3000:
                game.step(pick(rng))
            runs.append(game.summary())
        out[name] = {
            "episodes": len(runs),
            "ticks": statistics.mean(r["ticks"] for r in runs),
            "paddle_hits": statistics.mean(r["paddle_hits"] for r in runs),
            "bricks_broken": statistics.mean(r["bricks_broken"] for r in runs),
            "tracking_rate": statistics.mean(
                r["tracking_rate"] or 0.0 for r in runs
            ),
            "cleared": sum(r["cleared"] for r in runs) / len(runs),
        }
    return out


def summarize(events, config):
    """Recompute the episode summary from the event log.

    The run writes summary.json when it stops, but rebuilding from events means
    a run still in progress reports the same numbers on the same definitions.
    """
    toward = sum(1 for e in events if e["event"] == "toward")
    away = sum(1 for e in events if e["event"] == "away")
    moves = toward + away
    last = events[-1] if events else None
    total = config["brick_rows"] * config["brick_columns"]
    return {
        "ticks": last["tick"] if last else 0,
        "score": last["score"] if last else 0,
        "bricks_broken": sum(1 for e in events if e["event"] == "brick"),
        "bricks_left": last["bricks_left"] if last else total,
        "paddle_hits": sum(1 for e in events if e["event"] == "caught"),
        "balls_lost": sum(1 for e in events if e["event"] == "lost"),
        "lives_left": last["lives"] if last else config["lives"],
        "cleared": bool(last and last["bricks_left"] == 0),
        "moves_toward_ball": toward,
        "moves_away_from_ball": away,
        "scored_moves": moves,
        "holds": sum(1 for e in events if e["action"] == "HOLD"),
        "tracking_rate": (toward / moves) if moves else None,
        "tracking_chance_level": 0.5,
        "complete": bool(last and (last["lives"] <= 0 or last["bricks_left"] == 0)),
    }


def build(run_dir: Path, out_path: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    settings = Settings(**{k: v for k, v in config.items() if k in Settings.__dataclass_fields__})
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    provenance = json.loads((run_dir / "provenance.json").read_text())
    summary = summarize(events, config)
    ego = bool(config.get("egocentric"))

    frames = []
    if events:
        first = events[0]["before"]
        frames.append({
            "observation": 0,
            "before": first, "after": first,
            "action": "NONE", "outcome": "idle", "event": "start",
            "stimulus": "none",
            "difference_hz": None, "baseline_hz": None, "centered_hz": None,
            "left_hz": None, "right_hz": None,
            "kc_spikes": None, "total_spikes": None, "active_neurons": None,
            "spike_buckets": None, "brain_ms": 0.0,
            "source": events[0]["source"],
            "png": _png(render_arena(first, egocentric=ego)),
        })

    last_action, last_stimulus = "NONE", "NONE"
    for e in events:
        n = e.get("neural", {})
        frames.append({
            "observation": e["observation"],
            "before": e["before"],
            "after": e["after"],
            "action": e["action"],
            "outcome": e["outcome"],
            "event": e["event"],
            "stimulus": e.get("delivered_stimulus", "none"),
            "difference_hz": n.get("difference_hz"),
            "baseline_hz": n.get("baseline_hz"),
            "centered_hz": n.get("centered_hz"),
            "left_hz": n.get("left_hz"),
            "right_hz": n.get("right_hz"),
            "kc_spikes": n.get("KC_spikes"),
            "total_spikes": n.get("total_spikes"),
            "active_neurons": n.get("active_neurons"),
            "spike_buckets": n.get("spike_buckets"),
            "brain_ms": n.get("brain_ms"),
            "source": e["source"],
            "png": _png(render_arena(
                e["before"], action=last_action, stimulus=last_stimulus,
                egocentric=ego,
            )),
        })
        last_action = e["action"]
        last_stimulus = e.get("delivered_stimulus", "none").upper()

    payload = {
        "mode": config.get("fixture") and "fixture" or "malecns",
        "seed": config["seed"],
        "geometry": {
            "paddle_width": config["paddle_width"],
            "paddle_speed": config["paddle_speed"],
            "ball_speed": config["ball_speed"],
            "descent_ticks": config["ball_descent_ticks"],
            "rows": config["brick_rows"],
            "columns": config["brick_columns"],
            "lives": config["lives"],
            "egocentric": ego,
        },
        "decoder": {
            "deadband_hz": config["decoder_deadband_hz"],
            "baseline_obs": config["decoder_baseline_obs"],
            "neurons": 2,
        },
        "bucket_regions": provenance.get("dataset", {}).get("spike_bucket_regions"),
        "neurons": 166700,
        "connections": 25582938,
        "summary": summary,
        "baselines": baselines(settings),
        "frames": frames,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    return payload


if __name__ == "__main__":
    data = build(Path(sys.argv[1]), Path(sys.argv[2]))
    print(json.dumps({
        "frames": len(data["frames"]),
        "mode": data["mode"],
        "summary": data["summary"],
        "baselines": data["baselines"],
        "bytes": Path(sys.argv[2]).stat().st_size,
    }, indent=2))
