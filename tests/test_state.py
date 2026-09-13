import json

import pytest

from flytype.config import Settings
from flytype.state import (
    WorkerLockHeld,
    acquire_lock,
    append_event,
    build_provenance,
    commit_state,
    file_sha256,
    load_state,
    release_lock,
    stop_requested,
    validate_resume,
    verify_checkpoint,
    write_config,
    write_final,
    write_provenance,
)


def test_worker_lock_is_single_owner(tmp_path):
    handle = acquire_lock(tmp_path)
    try:
        with pytest.raises(WorkerLockHeld):
            acquire_lock(tmp_path)
    finally:
        release_lock(handle)
    # Released; a fresh acquire now succeeds.
    handle2 = acquire_lock(tmp_path)
    release_lock(handle2)


def test_provenance_signature_is_deterministic():
    settings = Settings(seed=1)
    dataset_info = {"mode": "fixture"}
    p1 = build_provenance(settings, dataset_info, "fixture")
    p2 = build_provenance(settings, dataset_info, "fixture")
    assert p1["provenance_sha256"] == p2["provenance_sha256"]


def test_provenance_signature_changes_with_target():
    dataset_info = {"mode": "fixture"}
    p1 = build_provenance(Settings(target="abc", alphabet="abc"), dataset_info, "fixture")
    p2 = build_provenance(Settings(target="acb", alphabet="abc"), dataset_info, "fixture")
    assert p1["provenance_sha256"] != p2["provenance_sha256"]


def test_provenance_signature_changes_with_decoder_deadband():
    dataset_info = {"mode": "fixture"}
    p1 = build_provenance(Settings(decoder_deadband_hz=0.1), dataset_info, "fixture")
    p2 = build_provenance(Settings(decoder_deadband_hz=0.2), dataset_info, "fixture")
    assert p1["provenance_sha256"] != p2["provenance_sha256"]


def test_config_and_provenance_round_trip(tmp_path):
    settings = Settings(seed=5)
    write_config(tmp_path, settings)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["seed"] == 5

    provenance = build_provenance(settings, {"mode": "fixture"}, "fixture")
    write_provenance(tmp_path, provenance)
    saved_provenance = json.loads((tmp_path / "provenance.json").read_text())
    assert saved_provenance["provenance_sha256"] == provenance["provenance_sha256"]


def test_commit_and_load_state_round_trip(tmp_path):
    assert load_state(tmp_path) is None
    commit_state(
        tmp_path,
        provenance_sha256="abc",
        observation_count=3,
        task_state={"target": "ab", "index": 1},
        feedback_state={"pending": "reward"},
        fixture_decoder_state=None,
        checkpoint=None,
        halted=None,
    )
    state = load_state(tmp_path)
    assert state["observation_count"] == 3
    assert state["task"]["index"] == 1


def test_validate_resume_rejects_signature_mismatch():
    with pytest.raises(ValueError):
        validate_resume({"provenance_sha256": "old"}, "new")
    validate_resume({"provenance_sha256": "same"}, "same")  # does not raise


def test_verify_checkpoint_detects_tampering(tmp_path):
    path = tmp_path / "brain-0.npz"
    path.write_bytes(b"hello")
    good = file_sha256(path)
    verify_checkpoint(path, good)  # does not raise
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError):
        verify_checkpoint(path, good)


def test_append_event_and_final_and_stop(tmp_path):
    assert not stop_requested(tmp_path)
    (tmp_path / "STOP").write_text("")
    assert stop_requested(tmp_path)

    append_event(tmp_path, {"observation": 1, "source": "fixture"})
    append_event(tmp_path, {"observation": 2, "source": "fixture"})
    lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert [json.loads(line)["observation"] for line in lines] == [1, 2]

    write_final(tmp_path, "follow @michaelzsguo on x")
    assert (tmp_path / "final.txt").read_text() == "follow @michaelzsguo on x"
