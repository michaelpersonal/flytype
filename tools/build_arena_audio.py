"""Build the arena replay's audio track, one clip per real game event.

The console uses this WAV as its master clock: observation i occupies the
window [i*hold, (i+1)*hold), so a sound can only exist where the event log says
something happened. Silence is therefore meaningful -- a stretch with no clicks
is a stretch where the paddle moved and hit nothing.

    python tools/build_arena_audio.py out/arena-frames.json out/arena-clicks.wav
"""

import argparse
import json
import wave
from pathlib import Path

import numpy as np

SR = 44100


def _env(n, decay):
    t = np.arange(n) / SR
    return t, np.exp(-t * decay)


def paddle_knock():
    """Low wooden knock: the ball meeting the paddle."""
    t, e = _env(int(SR * 0.10), 46)
    body = np.sin(2 * np.pi * 190 * t * np.exp(-t * 5)) * e
    click = np.random.default_rng(5).standard_normal(len(t)) * np.exp(-t * 420)
    sig = 0.75 * body + 0.18 * click
    return sig * np.minimum(1.0, t / 0.0012)


def brick_break():
    """Bright two-tone ping for a brick coming off the wall."""
    t, _ = _env(int(SR * 0.16), 0)
    out = np.zeros(len(t))
    for i, f in enumerate((1320.0, 1980.0)):
        tt = np.maximum(t - 0.03 * i, 0)
        out += 0.34 * np.sin(2 * np.pi * f * tt) * np.exp(-tt * 24) * (t >= 0.03 * i)
    return out


def ball_lost():
    """Descending tone for a dropped ball."""
    t, e = _env(int(SR * 0.42), 7.5)
    sweep = np.sin(2 * np.pi * (300 - 200 * np.minimum(t / 0.4, 1.0)) * t)
    return 0.55 * sweep * e


def paddle_step():
    """Very soft tick marking an observation that actually moved the paddle."""
    t, e = _env(int(SR * 0.03), 260)
    rng = np.random.default_rng(11)
    return 0.10 * rng.standard_normal(len(t)) * e


def build(frames, hold, path: Path):
    total = int(SR * (len(frames) * hold + 1.2))
    track = np.zeros(total)
    clips = {
        "caught": paddle_knock(),
        "brick": brick_break(),
        "lost": ball_lost(),
    }
    step = paddle_step()
    counts = {}
    for i, f in enumerate(frames):
        at = int(SR * i * hold)
        if f["action"] in ("LEFT", "RIGHT"):
            end = min(total, at + len(step))
            track[at:end] += step[: end - at]
            counts["move"] = counts.get("move", 0) + 1
        clip = clips.get(f["event"])
        if clip is None:
            continue
        start = at + int(SR * 0.02)
        end = min(total, start + len(clip))
        track[start:end] += clip[: end - start]
        counts[f["event"]] = counts.get(f["event"], 0) + 1
    peak = float(np.max(np.abs(track)) or 1.0)
    track = np.clip(track / max(peak, 1.0), -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((track * 32767).astype("<i2").tobytes())
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--hold", type=float, default=0.30)
    a = ap.parse_args()
    frames = json.loads(a.frames.read_text())["frames"]
    counts = build(frames, a.hold, a.out)
    print(json.dumps({
        "observations": len(frames),
        "seconds": round(len(frames) * a.hold, 1),
        "events": counts,
        "bytes": a.out.stat().st_size,
    }))


if __name__ == "__main__":
    main()
