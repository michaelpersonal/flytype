"""Record the replay console to an MP4 with a keypress track synced to the run.

Steps the local page one observation at a time via ?obs=N (headless Chrome),
then builds an audio track by placing a click at each observation's moment and
muxes the two with ffmpeg. Deterministic: no screen recording, no dropped
frames, and the audio lines up with the actual decoded keypresses.

    python tools/record_replay.py --url http://localhost:8934/flytype-replay.html \
        --frames <frames.json> --out flytype.mp4 --hold 0.85
"""

import argparse
import io
import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SR = 44100


def shot(url, obs, path, width, height, chrome):
    subprocess.run(
        [chrome, "--headless", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
         f"--screenshot={path}", f"--window-size={width},{height}",
         "--virtual-time-budget=7000", f"{url}?obs={obs}"],
        check=True, capture_output=True,
    )


def click(dull=False, seed=3):
    n = int(SR * 0.065)
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n) * np.exp(-t * (150 if dull else 260))
    reson = np.sin(2 * np.pi * (1100 if dull else 2600) * t) * np.exp(-t * (90 if dull else 150))
    body = np.sin(2 * np.pi * (95 if dull else 165) * t * np.exp(-t * 6)) * np.exp(-t * (55 if dull else 42))
    sig = 0.32 * noise * reson + 0.55 * body
    sig *= np.minimum(1.0, t / 0.0015)
    return sig * (0.7 if dull else 0.9)


def chime():
    n = int(SR * 0.30)
    t = np.arange(n) / SR
    out = np.zeros(n)
    for i, f in enumerate((784.0, 1174.7)):
        tt = np.maximum(t - 0.075 * i, 0)
        out += 0.34 * np.sin(2 * np.pi * f * tt) * np.exp(-tt * 13) * (t >= 0.075 * i)
    return out


def build_audio(frames, hold, path):
    total = int(SR * (len(frames) * hold + 1.0))
    track = np.zeros(total)
    clip_ok, clip_bad, clip_commit = click(False), click(True, 7), chime()
    for i, f in enumerate(frames):
        if f["action"] == "HOLD":
            continue
        start = int(SR * (i * hold + 0.06))
        clip = clip_bad if f["outcome"] == "incorrect" else clip_ok
        end = min(total, start + len(clip))
        track[start:end] += clip[: end - start]
        if f.get("committed"):
            cs = int(SR * (i * hold + 0.12))
            ce = min(total, cs + len(clip_commit))
            track[cs:ce] += clip_commit[: ce - cs]
    peak = np.max(np.abs(track)) or 1.0
    track = np.clip(track / max(peak, 1.0), -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((track * 32767).astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--frames", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--hold", type=float, default=0.85, help="seconds per observation")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--chrome", default=CHROME)
    ap.add_argument("--workdir", type=Path, default=Path("/tmp/flytype-capture"))
    a = ap.parse_args()

    frames = json.loads(a.frames.read_text())["frames"]
    work = a.workdir
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    for i in range(len(frames)):
        shot(a.url, i, work / f"f{i:05d}.png", a.width, a.height, a.chrome)
        if (i + 1) % 10 == 0 or i == len(frames) - 1:
            print(f"captured {i + 1}/{len(frames)}", flush=True)

    audio = work / "track.wav"
    build_audio(frames, a.hold, audio)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", f"{1 / a.hold:.6f}",
         "-i", str(work / "f%05d.png"), "-i", str(audio),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
         "-c:a", "aac", "-b:a", "160k", "-shortest", str(a.out)],
        check=True, capture_output=True,
    )
    print(json.dumps({
        "video": str(a.out),
        "frames": len(frames),
        "seconds": round(len(frames) * a.hold, 1),
        "bytes": a.out.stat().st_size,
    }))


if __name__ == "__main__":
    main()
