"""Build the replay's frames.json from a completed run directory.

Reconstructs the exact stimulus image shown at each observation, then pairs it
with that observation's recorded neural statistics and outcome. Used to feed
the local replay console; it reads only run artifacts and never re-simulates.

    python tools/build_replay_frames.py runs/tree out/frames.json
"""

import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

from flytype.display import render_frame


def _bucket_regions(run_dir: Path):
    """Per-bucket L/R/C tags, as recorded by the controller at run start."""
    prov = run_dir / "provenance.json"
    if not prov.exists():
        return None
    return json.loads(prov.read_text()).get("dataset", {}).get("spike_bucket_regions")


def build(run_dir: Path, out_path: Path) -> dict:
    config = json.loads((run_dir / "config.json").read_text())
    target = config["target"]
    events = [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]

    frames = []
    typed_before = ""

    if events:
        c0 = events[0]["candidates"]
        frame0 = render_frame(target, "", c0["left"], c0["right"],
                              action="NONE", stimulus="NONE")
        buf0 = io.BytesIO()
        Image.fromarray(frame0).save(buf0, format="PNG", optimize=True)
        frames.append({
            "observation": 0,
            "typed_before": "", "typed_after": "",
            "left": c0["left"], "right": c0["right"],
            "correct_side": c0["correct_side"],
            "depth": c0.get("depth"), "depth_required": c0.get("depth_required"),
            "committed": False,
            "action": "NONE", "outcome": "idle", "stimulus": "none",
            "difference_hz": None, "left_hz": None, "right_hz": None,
            "gate_spikes": None, "kc_spikes": None, "total_spikes": None,
            "active_neurons": None, "spike_buckets": None,
            "brain_ms": 0.0, "source": events[0]["source"],
            "png": "data:image/png;base64," + base64.b64encode(buf0.getvalue()).decode(),
        })

    for e in events:
        c = e["candidates"]
        n = e.get("neural", {})
        stimulus = e.get("delivered_stimulus", "none")
        frame = render_frame(
            target, typed_before, c["left"], c["right"],
            action=e["action"], stimulus=stimulus.upper(),
        )
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="PNG", optimize=True)
        frames.append({
            "observation": e["observation"],
            "typed_before": typed_before,
            "typed_after": e["typed"],
            "left": c["left"],
            "right": c["right"],
            "correct_side": c["correct_side"],
            "depth": c.get("depth"),
            "depth_required": c.get("depth_required"),
            "committed": bool(e.get("committed", e["outcome"] == "correct")),
            "action": e["action"],
            "outcome": e["outcome"],
            "stimulus": stimulus,
            "difference_hz": n.get("difference_hz"),
            "left_hz": n.get("left_hz"),
            "right_hz": n.get("right_hz"),
            "gate_spikes": n.get("gate_spikes"),
            "kc_spikes": n.get("KC_spikes"),
            "total_spikes": n.get("total_spikes"),
            "active_neurons": n.get("active_neurons"),
            "spike_buckets": n.get("spike_buckets"),
            "brain_ms": n.get("brain_ms"),
            "source": e["source"],
            "png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        })
        typed_before = e["typed"]

    payload = {
        "target": target,
        "mode": "fixture" if config.get("fixture") else "malecns",
        "keyboard": config.get("keyboard", "pair"),
        "seed": config["seed"],
        "decoder_deadband_hz": config["decoder_deadband_hz"],
        "alphabet_size": len(set(config["alphabet"])),
        "bucket_regions": _bucket_regions(run_dir),
        "neurons": 166700,
        "connections": 25582938,
        "frames": frames,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    return payload


if __name__ == "__main__":
    run = Path(sys.argv[1])
    out = Path(sys.argv[2])
    data = build(run, out)
    print(json.dumps({
        "frames": len(data["frames"]),
        "keyboard": data["keyboard"],
        "mode": data["mode"],
        "bytes": out.stat().st_size,
    }))
