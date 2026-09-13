"""Run-directory persistence: locking, provenance, atomic state, event log.

No event or checkpoint file may contain account credentials, because this
project accepts none. Everything here is local JSON/JSONL/npz; no network
calls happen in this module.
"""

import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path

from .neural.common import save_json

PACKAGE_ROOT = Path(__file__).parent


class WorkerLockHeld(SystemExit):
    pass


def acquire_lock(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    handle = (out / "worker.lock").open("a")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise WorkerLockHeld(f"A worker already owns {out}") from None
    return handle


def release_lock(handle):
    handle.close()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_signature() -> dict:
    return {
        str(path.relative_to(PACKAGE_ROOT)): file_sha256(path)
        for path in sorted(PACKAGE_ROOT.rglob("*"))
        if path.suffix in (".py", ".cpp")
    }


# Fields that identify "the same run" for resume purposes. Deliberately
# excludes per-invocation controls (steps, max_attempts_per_character, fast)
# so e.g. `--steps 3` followed by `--steps 0` on the same --out still resumes.
LOCKED_SETTINGS_FIELDS = (
    "target",
    "alphabet",
    "seed",
    "neural_ms",
    "neural_bin_ms",
    "pulse_ms",
    "pulse_current",
    "decoder_deadband_hz",
    "decoder_baseline_obs",
    "fixture",
    "frozen",
    "shuffle_feedback",
)

# Additional identity fields for a `play` run: the arena geometry decides what
# the network sees and how hard the episode is, so changing it starts a new run
# rather than resuming an old one.
LOCKED_ARENA_FIELDS = (
    "paddle_width",
    "paddle_speed",
    "ball_speed",
    "ball_descent_ticks",
    "brick_rows",
    "brick_columns",
    "lives",
    "egocentric",
)


def build_provenance(settings, dataset_info, mode, task="type"):
    all_settings = dataclasses.asdict(settings)
    source = source_signature()
    fields = LOCKED_SETTINGS_FIELDS
    if task == "play":
        fields = fields + LOCKED_ARENA_FIELDS
    resume_lock = {
        "task": task,
        "identity_settings": {k: all_settings[k] for k in fields},
        "dataset": dataset_info,
        "source_sha256": source,
    }
    provenance = {
        "settings": all_settings,
        "dataset": dataset_info,
        "mode": mode,
        "task": task,
        "decoder": (
            "DNp20 mean right-minus-left firing vs a configured deadband selects "
            "LEFT/RIGHT/HOLD. DNpe017 spikes are logged, not gated. Engineered "
            "fixed mapping; never overridden by target text or correctness."
        ),
        "reinforcement": (
            "Correct selection schedules a PAM11 reward pulse and incorrect "
            "selection a PPL101 aversive pulse, delivered one observation later. "
            "Not modeled pain, pleasure, consciousness or validated learning."
        ),
        "claim": (
            "A MaleCNS connectome simulation selects characters to reproduce a "
            "supplied target sentence through a fixed visual and neural "
            "interface."
            if task == "type"
            else "A MaleCNS connectome simulation drives a brick-breaker paddle "
            "through the same fixed two-choice readout. The paddle moves only "
            "where the decoded action says; no tick corrects it toward the "
            "ball. Completing or failing an episode is not evidence of "
            "learning, understanding or play."
        ),
        "source_sha256": source,
        "resume_lock": resume_lock,
    }
    # Only the resume-lock subset gates resume; refuses to resume if target,
    # decoder configuration, dataset or source signature differs (section 11).
    provenance["provenance_sha256"] = hashlib.sha256(
        json.dumps(resume_lock, sort_keys=True).encode()
    ).hexdigest()
    return provenance


def write_config(out: Path, settings):
    save_json(out / "config.json", dataclasses.asdict(settings))


def write_provenance(out: Path, provenance: dict):
    save_json(out / "provenance.json", provenance)


def load_state(out: Path):
    path = out / "state.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def commit_state(
    out: Path,
    *,
    provenance_sha256: str,
    observation_count: int,
    task_state: dict,
    feedback_state: dict,
    fixture_decoder_state,
    checkpoint,
    halted,
    decoder_state=None,
):
    save_json(
        out / "state.json",
        {
            "provenance_sha256": provenance_sha256,
            "observation_count": observation_count,
            "task": task_state,
            "feedback": feedback_state,
            "fixture_decoder": fixture_decoder_state,
            "decoder": decoder_state,
            "checkpoint": checkpoint,
            "halted": halted,
        },
    )


def validate_resume(saved_state: dict, provenance_sha256: str):
    if saved_state["provenance_sha256"] != provenance_sha256:
        raise ValueError(
            "Run target, decoder configuration, dataset or source signature "
            "changed since this run directory was created; use a new --out "
            "directory or restore the original configuration."
        )


def checkpoint_path(out: Path, slot: int) -> Path:
    return out / f"brain-{slot}.npz"


def verify_checkpoint(path: Path, expected_sha256: str):
    if file_sha256(path) != expected_sha256:
        raise ValueError("Checkpoint integrity mismatch: " + str(path))


def append_event(out: Path, event: dict):
    with (out / "events.jsonl").open("a") as f:
        f.write(json.dumps(event, allow_nan=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_final(out: Path, text: str):
    (out / "final.txt").write_text(text)


def stop_requested(out: Path) -> bool:
    return (out / "STOP").exists()
