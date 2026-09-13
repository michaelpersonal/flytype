"""Why a population does or does not fire: net synaptic input against threshold.

    python tools/probe_propagation.py            # every visual stage
    python tools/probe_propagation.py Mi1 T4c    # what drives these types

A cell fires when sustained input exceeds its threshold minus its resting
potential -- 7 mV with the shipped parameters. This prints the net input each
population actually receives, split into excitation and inhibition, so a silent
population can be told apart from a clamped one. They need different fixes: a
silent cell is missing drive, a clamped cell is being held down by a balanced
circuit that no global gain can rescue, because scaling raises both sides.
"""

import re
import sys
from collections import defaultdict

import numpy as np

STAGES = [
    ("photoreceptor", (r"R1-R6$", r"R7[dpy]?$", r"R8[py]?$")),
    ("lamina", (r"L[1-5]$", r"C2$", r"C3$", r"T1$", r"Lawf\d")),
    ("medulla", (r"Mi\d", r"Tm\d", r"TmY", r"Dm\d", r"Pm\d")),
    ("T4/T5", (r"T[45][a-d]$",)),
    ("lobula", (r"LC\d", r"LPLC\d", r"LPi")),
    ("descending", (r"DN",)),
]


def main(argv):
    from flytype.neural.common import annotations
    from flytype.neural.visual import VisualMemoryBrain

    brain = VisualMemoryBrain()
    table = annotations(brain.ids)
    types = table["type"].fillna("").astype(str).to_numpy()
    gap = -45.0 - float(brain.rest[0])

    def select(patterns):
        return np.flatnonzero([any(re.match(p, x) for p in patterns) for x in types])

    def incoming(rows):
        target = set(rows.tolist())
        per_source = defaultdict(lambda: [0, 0.0])
        for i in range(brain.n):
            for e in range(brain.ptr[i], brain.ptr[i + 1]):
                if brain.post[e] in target:
                    entry = per_source[types[i] or "(untyped)"]
                    entry[0] += 1
                    entry[1] += float(brain.weight[e])
        return per_source

    print(f"resting {brain.rest[0]:.0f} mV, threshold -45 mV "
          f"-> sustained input must exceed {gap:.0f} to fire\n")

    if argv:
        for name in argv:
            rows = np.flatnonzero(types == name)
            if not len(rows):
                print(f"{name}: no such type\n")
                continue
            sources = incoming(rows)
            n = len(rows)
            net = sum(w for _, w in sources.values()) / n
            exc = sum(w for _, w in sources.values() if w > 0) / n
            inh = sum(w for _, w in sources.values() if w < 0) / n
            state = "clamped" if net < 0 else "under threshold" if net < gap else "drivable"
            print(f"=== {name} ({n} cells): net {net:+.2f} per cell -- {state}")
            print(f"    excitation {exc:+.1f}, inhibition {inh:+.1f}")
            for source, (edges, weight) in sorted(
                sources.items(), key=lambda kv: -abs(kv[1][1])
            )[:8]:
                print(f"    {source[:14]:14s} {edges:7d} edges {weight / n:+8.2f} per cell")
            print()
        return

    print(f"{'stage':16s} {'cells':>8s} {'net input / cell':>18s}  state")
    for name, patterns in STAGES:
        rows = select(patterns)
        if not len(rows):
            continue
        sources = incoming(rows)
        net = sum(w for _, w in sources.values()) / len(rows)
        state = ("clamped below rest" if net < 0
                 else "below threshold" if net < gap else "drivable")
        print(f"{name:16s} {len(rows):8d} {net:18.2f}  {state}")


if __name__ == "__main__":
    main(sys.argv[1:])
