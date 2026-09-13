"""Single-worker character-selection run loop.

Neural decoding never reads target text or correctness; this module is the
only place that connects a decoded LEFT/RIGHT/HOLD action to task scoring.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from .config import FIXTURE_WALL_DELAY_SECONDS, Settings
from .decoder import FixtureDecoder
from .display import render_frame
from .feedback import FeedbackSchedule
from .state import (
    WorkerLockHeld,
    acquire_lock,
    append_event,
    build_provenance,
    checkpoint_path,
    file_sha256,
    load_state,
    release_lock,
    stop_requested,
    validate_resume,
    write_config,
    write_final,
    write_provenance,
)
from .task import Task, TreeTask, new_rng


def _build_parser():
    p = argparse.ArgumentParser(prog="flytype")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("verify")

    run = sub.add_parser("run")
    run.add_argument("--target", default=Settings.target)
    run.add_argument("--out", type=Path, default=Path("runs/default"))
    run.add_argument("--seed", type=int, default=Settings.seed)
    run.add_argument("--steps", type=int, default=Settings.steps)
    run.add_argument(
        "--max-attempts-per-character",
        type=int,
        default=Settings.max_attempts_per_character,
    )
    run.add_argument("--neural-ms", type=float, default=Settings.neural_ms)
    run.add_argument(
        "--decoder-deadband-hz", type=float, default=Settings.decoder_deadband_hz
    )
    run.add_argument(
        "--decoder-baseline-obs",
        type=int,
        default=Settings.decoder_baseline_obs,
        help="compare the DNp20 difference against a running mean of its own "
             "last N values instead of against zero; 0 disables",
    )
    run.add_argument(
        "--keyboard",
        choices=["pair", "tree"],
        default=Settings.keyboard,
        help="pair: one correct character vs one distractor. "
             "tree: full alphabet via a binary descent on the same L/R readout",
    )
    run.add_argument("--fixture", action="store_true")
    run.add_argument("--frozen", action="store_true")
    run.add_argument("--shuffle-feedback", action="store_true")
    run.add_argument("--fast", action="store_true")

    play = sub.add_parser(
        "play", help="drive a brick-breaker paddle with the same L/R readout"
    )
    play.add_argument("--out", type=Path, default=Path("runs/breakout"))
    play.add_argument("--seed", type=int, default=Settings.seed)
    play.add_argument("--steps", type=int, default=Settings.steps)
    play.add_argument("--neural-ms", type=float, default=Settings.neural_ms)
    play.add_argument(
        "--decoder-deadband-hz", type=float, default=Settings.decoder_deadband_hz
    )
    play.add_argument(
        "--decoder-baseline-obs", type=int, default=16,
        help="running-mean window for the DNp20 difference; 0 disables centering",
    )
    play.add_argument("--paddle-width", type=int, default=Settings.paddle_width)
    play.add_argument("--paddle-speed", type=int, default=Settings.paddle_speed)
    play.add_argument("--ball-speed", type=float, default=Settings.ball_speed)
    play.add_argument("--ball-descent-ticks", type=int,
                      default=Settings.ball_descent_ticks,
                      help="observations the ball spends falling to the paddle")
    play.add_argument("--brick-rows", type=int, default=Settings.brick_rows)
    play.add_argument("--brick-columns", type=int, default=Settings.brick_columns)
    play.add_argument("--lives", type=int, default=Settings.lives)
    play.add_argument("--egocentric", action="store_true",
                      help="scroll the field so the paddle is always centred "
                           "in the frame the network sees")
    play.add_argument("--fixture", action="store_true")
    play.add_argument("--frozen", action="store_true")
    play.add_argument("--shuffle-feedback", action="store_true")
    play.add_argument("--fast", action="store_true")

    status = sub.add_parser("status")
    status.add_argument("--out", type=Path, default=Path("runs/default"))
    return p


def cmd_prepare():
    from .data import prepare

    prepare()


def cmd_verify():
    from .data import verify

    print(json.dumps(verify()))


def cmd_status(out: Path):
    state = load_state(out)
    if state is None:
        print(json.dumps({"observation_count": 0, "typed": "", "halted": None}))
        return
    print(
        json.dumps(
            {
                "observation_count": state["observation_count"],
                "target_index": state["task"]["index"],
                "typed": state["task"]["target"][: state["task"]["index"]],
                "target": state["task"]["target"],
                "attempts_this_character": state["task"]["attempts"],
                "halted": state["halted"],
                "done": state["task"]["index"] >= len(state["task"]["target"]),
            },
            indent=2,
        )
    )


def cmd_run(args):
    settings = Settings(
        target=args.target,
        seed=args.seed,
        steps=args.steps,
        max_attempts_per_character=args.max_attempts_per_character,
        neural_ms=args.neural_ms,
        decoder_deadband_hz=args.decoder_deadband_hz,
        decoder_baseline_obs=args.decoder_baseline_obs,
        keyboard=args.keyboard,
        fixture=args.fixture,
        frozen=args.frozen,
        shuffle_feedback=args.shuffle_feedback,
        fast=args.fast,
    )
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(out)
    try:
        _run_loop(settings, out)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("Stopped; run state preserved.", flush=True)
    except Exception as e:
        (out / "error.json").write_text(
            json.dumps({"type": type(e).__name__, "reason": str(e)}, indent=2) + "\n"
        )
        print(f"Stopped safely: {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        release_lock(lock)


def _open_substrate(settings):
    """Load whichever decoder this run uses; identical for `run` and `play`."""
    if settings.fixture:
        dataset_info = {
            "mode": "fixture",
            "note": "No MaleCNS graph loaded; decoder is a scripted stand-in.",
        }
        return dataset_info, None, FixtureDecoder(settings.seed ^ 0x9E3779B9)
    from .data import verify
    from .neural.controller import FlyController

    dataset_info = verify()
    controller = FlyController(settings)
    dataset_info = {**dataset_info, "spike_bucket_regions": controller.bucket_regions}
    return dataset_info, controller, None


def _run_loop(settings, out: Path):
    dataset_info, controller, fixture_decoder = _open_substrate(settings)
    mode = "fixture" if settings.fixture else "malecns"
    provenance = build_provenance(settings, dataset_info, mode)
    signature = provenance["provenance_sha256"]

    candidate_rng = new_rng(settings.seed)
    shuffle_rng = new_rng(settings.seed ^ 0x5F3759DF) if settings.shuffle_feedback else None
    task_class = TreeTask if settings.keyboard == 'tree' else Task
    task = task_class(settings.target, settings.alphabet, candidate_rng)
    feedback = FeedbackSchedule(settings.shuffle_feedback, shuffle_rng)

    saved = load_state(out)
    if saved is not None:
        validate_resume(saved, signature)
        task.restore(saved["task"])
        feedback.restore(saved["feedback"])
        observation_count = saved["observation_count"]
        if fixture_decoder is not None and saved.get("fixture_decoder") is not None:
            fixture_decoder.restore(saved["fixture_decoder"])
        if controller is not None and saved.get("checkpoint"):
            cp = saved["checkpoint"]
            path = out / cp["file"]
            from .state import verify_checkpoint

            verify_checkpoint(path, cp["sha256"])
            controller.restore(path)
        if controller is not None and saved.get("decoder"):
            controller.decoder.restore(saved["decoder"])
    else:
        write_config(out, settings)
        write_provenance(out, provenance)
        observation_count = 0

    if task.done:
        write_final(out, settings.target)
        print(json.dumps({"completed": True, "final": settings.target}), flush=True)
        return

    from .state import commit_state
    from PIL import Image

    count = 0
    while not settings.steps or count < settings.steps:
        if stop_requested(out):
            print("STOP file present; stopping cleanly.", flush=True)
            break

        candidates = task.candidates()
        frame = render_frame(
            settings.target,
            task.typed,
            candidates["left"],
            candidates["right"],
        )
        deliver = feedback.pending

        if settings.fixture:
            neural = fixture_decoder.decode(candidates["correct_side"])
        else:
            neural = controller.observe(frame, deliver)

        action = neural["action"]
        outcome = task.score(action)
        _, scheduled = feedback.advance(outcome)

        checkpoint_info = None
        if controller is not None:
            slot = observation_count % 2
            cppath = checkpoint_path(out, slot)
            controller.save(cppath)
            checkpoint_info = {"file": cppath.name, "sha256": file_sha256(cppath)}
        observation_count += 1

        halted_reason = None
        if outcome == "incorrect" and task.attempts > settings.max_attempts_per_character:
            halted_reason = (
                f"Exceeded max attempts ({settings.max_attempts_per_character}) "
                f"for target character index {task.index}"
            )

        event = {
            "observation": observation_count,
            "target_index": task.index,
            "candidates": {
                "left": candidates["left"],
                "right": candidates["right"],
                "correct_side": candidates["correct_side"],
                "depth": candidates.get("depth"),
                "depth_required": candidates.get("depth_required"),
            },
            "committed": getattr(task, "committed", outcome == "correct"),
            "neural": neural,
            "action": action,
            "outcome": outcome,
            "delivered_stimulus": deliver,
            "scheduled_next_stimulus": scheduled,
            "typed": task.typed,
            "compute_seconds": neural.get("compute_seconds"),
            "mode": mode,
            "source": neural["source"],
            "halted": halted_reason,
        }
        append_event(out, event)
        Image.fromarray(frame).save(out / "latest-input.png")
        commit_state(
            out,
            provenance_sha256=signature,
            observation_count=observation_count,
            task_state=task.state(),
            feedback_state=feedback.state(),
            fixture_decoder_state=(
                fixture_decoder.state() if fixture_decoder is not None else None
            ),
            decoder_state=(
                controller.decoder.state() if controller is not None else None
            ),
            checkpoint=checkpoint_info,
            halted=halted_reason,
        )
        print(
            json.dumps(
                {
                    "observation": observation_count,
                    "action": action,
                    "outcome": outcome,
                    "typed": task.typed,
                    "source": neural["source"],
                }
            ),
            flush=True,
        )
        count += 1

        if halted_reason:
            print(f"Halted transparently: {halted_reason}", file=sys.stderr)
            raise SystemExit(1)
        if task.done:
            write_final(out, settings.target)
            print(json.dumps({"completed": True, "final": settings.target}), flush=True)
            break
        if settings.fixture and not settings.fast:
            time.sleep(FIXTURE_WALL_DELAY_SECONDS)


def cmd_play(args):
    settings = Settings(
        seed=args.seed,
        steps=args.steps,
        neural_ms=args.neural_ms,
        decoder_deadband_hz=args.decoder_deadband_hz,
        decoder_baseline_obs=args.decoder_baseline_obs,
        paddle_width=args.paddle_width,
        paddle_speed=args.paddle_speed,
        ball_speed=args.ball_speed,
        ball_descent_ticks=args.ball_descent_ticks,
        brick_rows=args.brick_rows,
        brick_columns=args.brick_columns,
        lives=args.lives,
        egocentric=args.egocentric,
        fixture=args.fixture,
        frozen=args.frozen,
        shuffle_feedback=args.shuffle_feedback,
        fast=args.fast,
    )
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(out)
    try:
        _play_loop(settings, out)
    except SystemExit:
        raise
    except KeyboardInterrupt:
        print("Stopped; episode state preserved.", flush=True)
    except Exception as e:
        (out / "error.json").write_text(
            json.dumps({"type": type(e).__name__, "reason": str(e)}, indent=2) + "\n"
        )
        print(f"Stopped safely: {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(1) from None
    finally:
        release_lock(lock)


def _play_loop(settings, out: Path):
    """One observation, one tick: decode an action, then step the arena.

    The only value that crosses from the game into the network is the rendered
    RGB frame; the only value that crosses back is LEFT/RIGHT/HOLD. Outcome
    scoring happens entirely on this side of that line.
    """
    from PIL import Image

    from .arena import render_arena
    from .breakout import BALL_SIZE, BreakoutGame, new_game_rng
    from .state import commit_state, verify_checkpoint

    dataset_info, controller, fixture_decoder = _open_substrate(settings)
    mode = "fixture" if settings.fixture else "malecns"
    provenance = build_provenance(settings, dataset_info, mode, task="play")
    signature = provenance["provenance_sha256"]

    shuffle_rng = (
        new_rng(settings.seed ^ 0x5F3759DF) if settings.shuffle_feedback else None
    )
    game = BreakoutGame(settings, new_game_rng(settings.seed))
    feedback = FeedbackSchedule(settings.shuffle_feedback, shuffle_rng)

    saved = load_state(out)
    if saved is not None:
        validate_resume(saved, signature)
        game.restore(saved["task"])
        feedback.restore(saved["feedback"])
        observation_count = saved["observation_count"]
        if fixture_decoder is not None and saved.get("fixture_decoder") is not None:
            fixture_decoder.restore(saved["fixture_decoder"])
        if controller is not None and saved.get("checkpoint"):
            cp = saved["checkpoint"]
            path = out / cp["file"]
            verify_checkpoint(path, cp["sha256"])
            controller.restore(path)
        if controller is not None and saved.get("decoder"):
            controller.decoder.restore(saved["decoder"])
    else:
        write_config(out, settings)
        write_provenance(out, provenance)
        observation_count = 0

    if game.done:
        _write_summary(out, game)
        print(json.dumps({"completed": True, **game.summary()}), flush=True)
        return

    last_action, last_stimulus = "NONE", "NONE"
    count = 0
    while not settings.steps or count < settings.steps:
        if stop_requested(out):
            print("STOP file present; stopping cleanly.", flush=True)
            break

        view = game.view()
        frame = render_arena(view, action=last_action, stimulus=last_stimulus,
                             egocentric=settings.egocentric)
        deliver = feedback.pending

        if settings.fixture:
            ball = view["ball_x"] + BALL_SIZE / 2
            toward = "LEFT" if ball < game.paddle_center else "RIGHT"
            neural = fixture_decoder.decode(toward)
        else:
            neural = controller.observe(frame, deliver)

        action = neural["action"]
        outcome = game.step(action)
        _, scheduled = feedback.advance(outcome)
        last_action, last_stimulus = action, deliver.upper()

        checkpoint_info = None
        if controller is not None:
            slot = observation_count % 2
            cppath = checkpoint_path(out, slot)
            controller.save(cppath)
            checkpoint_info = {"file": cppath.name, "sha256": file_sha256(cppath)}
        observation_count += 1

        after = game.view()
        append_event(out, {
            "observation": observation_count,
            "tick": game.tick,
            "before": view,
            "after": after,
            "neural": neural,
            "action": action,
            "outcome": outcome,
            "event": game.last_event,
            "delivered_stimulus": deliver,
            "scheduled_next_stimulus": scheduled,
            "score": game.score,
            "lives": game.lives,
            "bricks_left": game.bricks_left,
            "compute_seconds": neural.get("compute_seconds"),
            "mode": mode,
            "source": neural["source"],
            "halted": None,
        })
        Image.fromarray(frame).save(out / "latest-input.png")
        commit_state(
            out,
            provenance_sha256=signature,
            observation_count=observation_count,
            task_state=game.state(),
            feedback_state=feedback.state(),
            fixture_decoder_state=(
                fixture_decoder.state() if fixture_decoder is not None else None
            ),
            decoder_state=(
                controller.decoder.state() if controller is not None else None
            ),
            checkpoint=checkpoint_info,
            halted=None,
        )
        print(json.dumps({
            "observation": observation_count,
            "action": action,
            "outcome": outcome,
            "event": game.last_event,
            "score": game.score,
            "lives": game.lives,
            "bricks_left": game.bricks_left,
            "source": neural["source"],
        }), flush=True)
        count += 1

        if game.done:
            _write_summary(out, game)
            print(json.dumps({"completed": True, **game.summary()}), flush=True)
            break
        if settings.fixture and not settings.fast:
            time.sleep(FIXTURE_WALL_DELAY_SECONDS)

    _write_summary(out, game)


def _write_summary(out: Path, game):
    """Episode outcome plus the honest baseline it should be read against."""
    summary = {
        **game.summary(),
        "note": (
            "tracking_rate is the share of non-HOLD paddle moves that closed the "
            "horizontal gap to the ball. Chance is 0.5. A rate at or below chance "
            "means the decoder did not track the ball, whatever the score says."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    args = _build_parser().parse_args()
    if args.command == "prepare":
        cmd_prepare()
    elif args.command == "verify":
        cmd_verify()
    elif args.command == "status":
        cmd_status(args.out)
    elif args.command == "run":
        try:
            cmd_run(args)
        except WorkerLockHeld as e:
            raise SystemExit(str(e))
    elif args.command == "play":
        try:
            cmd_play(args)
        except WorkerLockHeld as e:
            raise SystemExit(str(e))


if __name__ == "__main__":
    main()
