"""Pool `flytype play` episodes and test the tracking rate against chance.

A single episode has too few scored moves to separate a small effect from
noise, so this pools whatever run directories it is given and reports an exact
two-sided binomial test against p = 0.5 -- the rate an agent that cannot see
the ball produces by construction (tests/test_breakout.py checks that).

    python tools/analyze_play.py runs/breakout runs/breakout-s*
"""

import json
import math
import sys
from pathlib import Path


def binomial_two_sided(k, n, p=0.5):
    """Exact two-sided binomial p-value, by summing equally-or-less likely tails."""
    if n == 0:
        return 1.0
    def pmf(i):
        return math.comb(n, i) * p**i * (1 - p) ** (n - i)
    observed = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= observed * 1.0000001))


def read(run_dir: Path):
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    config = json.loads((run_dir / "config.json").read_text())
    toward = sum(1 for e in events if e["event"] == "toward")
    away = sum(1 for e in events if e["event"] == "away")
    return {
        "run": run_dir.name,
        "seed": config["seed"],
        "egocentric": bool(config.get("egocentric")),
        "baseline_obs": config["decoder_baseline_obs"],
        "ticks": events[-1]["tick"] if events else 0,
        "toward": toward,
        "away": away,
        "holds": sum(1 for e in events if e["action"] == "HOLD"),
        "paddle_hits": sum(1 for e in events if e["event"] == "caught"),
        "bricks": sum(1 for e in events if e["event"] == "brick"),
        "balls_lost": sum(1 for e in events if e["event"] == "lost"),
        "left": sum(1 for e in events if e["action"] == "LEFT"),
        "right": sum(1 for e in events if e["action"] == "RIGHT"),
    }


def main():
    runs = [read(Path(a)) for a in sys.argv[1:]]
    if not runs:
        raise SystemExit(__doc__)

    groups = {}
    for r in runs:
        groups.setdefault("egocentric" if r["egocentric"] else "fixed view", []).append(r)

    print(f"{'run':24s} {'view':11s} {'ticks':>6s} {'toward':>7s} {'away':>5s} "
          f"{'rate':>6s} {'hits':>5s} {'L/R':>9s}")
    for r in runs:
        n = r["toward"] + r["away"]
        rate = r["toward"] / n if n else float("nan")
        print(f"{r['run']:24s} {'ego' if r['egocentric'] else 'fixed':11s} "
              f"{r['ticks']:6d} {r['toward']:7d} {r['away']:5d} {rate:6.3f} "
              f"{r['paddle_hits']:5d} {r['left']:4d}/{r['right']:<4d}")

    print()
    for name, group in groups.items():
        k = sum(r["toward"] for r in group)
        n = k + sum(r["away"] for r in group)
        rate = k / n if n else float("nan")
        se = math.sqrt(0.25 / n) if n else float("nan")
        p = binomial_two_sided(k, n)
        print(f"{name}: {len(group)} episode(s), {n} scored moves, "
              f"tracking {rate:.3f} ({(rate - 0.5) / se:+.2f} SE from chance), "
              f"exact two-sided p = {p:.4f}")
        print(f"  {'above chance' if p < 0.05 and rate > 0.5 else 'below chance' if p < 0.05 else 'not distinguishable from chance'} "
              f"at the 5% level")


if __name__ == "__main__":
    main()
