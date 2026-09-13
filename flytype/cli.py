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
        help="compare the DNp20 difference against a running median of its "
             "own last N values instead of against zero; 0 disables",
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
        "--decoder-baseline-obs", type=int, default=64,
        help="running-median window for the DNp20 decision threshold; "
             "too short and the paddle cannot cross the field, too long "
             "and the side bias creeps back. 0 disables centering",
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

    calibrate_play = sub.add_parser(
        "calibrate-play-decoder",
        help="fit a frozen MaleCNS population decoder on visual probe frames",
    )
    calibrate_play.add_argument(
        "--out", type=Path, default=Path("calibration/play-population.json")
    )
    calibrate_play.add_argument("--seed", type=int, default=Settings.seed)
    calibrate_play.add_argument("--neural-ms", type=float, default=Settings.neural_ms)
    calibrate_play.add_argument("--offsets", type=int, default=17,
                                help="ball offsets sampled across the reachable range")
    calibrate_play.add_argument("--heights", type=int, default=5,
                                help="ball heights sampled down the field")
    calibrate_play.add_argument("--repeats", type=int, default=3,
                                help="observations per condition; single "
                                     "observations are noisier than the signal")
    calibrate_play.add_argument("--warmup", type=int, default=8,
                                help="observations discarded before recording "
                                     "while the network reaches steady state")
    calibrate_play.add_argument("--features", type=int, default=32)
    # The frame the probes are drawn on must be the frame the run will show,
    # so the geometry that shapes it is settable here too and is recorded in
    # the artifact. `web` refuses a decoder whose conditions do not match.
    calibrate_play.add_argument("--paddle-width", type=int,
                                default=Settings.paddle_width)
    calibrate_play.add_argument("--brick-rows", type=int,
                                default=Settings.brick_rows)
    calibrate_play.add_argument("--brick-columns", type=int,
                                default=Settings.brick_columns)
    calibrate_play.add_argument("--fixed-view", action="store_true")

    web = sub.add_parser(
        "web", help="run the continuous loopback-only FlyBreak website"
    )
    web.add_argument("--out", type=Path, default=Path("runs/web-malecns"))
    web.add_argument("--seed", type=int, default=Settings.seed)
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--tick-seconds", type=float, default=0.18)
    web.add_argument("--neural-ms", type=float, default=Settings.neural_ms)
    web.add_argument("--decoder-deadband-hz", type=float,
                     default=Settings.decoder_deadband_hz)
    web.add_argument("--decoder-baseline-obs", type=int, default=64)
    web.add_argument("--paddle-width", type=int, default=Settings.paddle_width)
    web.add_argument("--paddle-speed", type=int, default=Settings.paddle_speed)
    web.add_argument("--ball-speed", type=float, default=Settings.ball_speed)
    web.add_argument("--ball-descent-ticks", type=int,
                     default=Settings.ball_descent_ticks)
    web.add_argument("--brick-rows", type=int, default=Settings.brick_rows)
    web.add_argument("--brick-columns", type=int, default=Settings.brick_columns)
    web.add_argument("--lives", type=int, default=Settings.lives)
    web.add_argument("--fixed-view", action="store_true")
    web.add_argument("--fixture", action="store_true")
    web.add_argument("--frozen", action="store_true")
    web.add_argument("--shuffle-feedback", action="store_true")
    web.add_argument("--no-open", action="store_true")
    web.add_argument(
        "--motor-decoder",
        type=Path,
        default=Path("calibration/play-population.json"),
    )

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


def _web_settings(args):
    return Settings(
        seed=args.seed,
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
        egocentric=not args.fixed_view,
        fixture=args.fixture,
        frozen=args.frozen,
        shuffle_feedback=args.shuffle_feedback,
        fast=args.fixture,
    )


def cmd_calibrate_play_decoder(args):
    from .data import verify
    from .play_calibration import calibrate_play_decoder

    settings = Settings(
        seed=args.seed,
        neural_ms=args.neural_ms,
        decoder_baseline_obs=0,
        paddle_width=args.paddle_width,
        brick_rows=args.brick_rows,
        brick_columns=args.brick_columns,
        egocentric=not args.fixed_view,
        frozen=True,
    )
    dataset_info = verify()

    def progress(done, total, compute_seconds):
        print(json.dumps({
            "calibrating_population": done,
            "of": total,
            "compute_seconds": compute_seconds,
        }), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    decoder = calibrate_play_decoder(
        settings,
        args.out,
        dataset_info=dataset_info,
        offset_count=args.offsets,
        height_count=args.heights,
        repeats=args.repeats,
        warmup=args.warmup,
        feature_count=args.features,
        progress=progress,
    )
    meta = decoder.metadata
    print(json.dumps({
        "calibrated": True,
        "path": str(args.out),
        "decoder_sha256": decoder.artifact_sha256,
        "selected_cells": len(decoder.cell_ids),
        "observations": meta["observations"],
        "validation_mae_px": meta["validation_mae_px"],
        "validation_correlation": meta["validation_correlation"],
        # The same pipeline fitted to permuted labels. The real numbers only
        # mean something to the extent they beat these.
        "shuffled_control_mae_px": meta["shuffled_control_mae_px"],
        "shuffled_control_correlation": meta["shuffled_control_correlation"],
        # What a decoder that learned nothing would score on the same split.
        "predict_the_mean_mae_px": meta["predict_the_mean_mae_px"],
        # Repeats averaged per held-out condition: if this is much better than
        # the per-observation score, the signal is real but buried in noise.
        "condition_averaged_mae_px": meta["condition_averaged_mae_px"],
        "condition_averaged_correlation": meta["condition_averaged_correlation"],
        "condition_averaged_shuffled_mae_px": meta["condition_averaged_shuffled_mae_px"],
        "condition_averaged_shuffled_correlation":
            meta["condition_averaged_shuffled_correlation"],
        "validation_by_ball_height": meta["validation_by_ball_height"],
        "verdict": _calibration_verdict(meta),
    }), flush=True)


def _calibration_verdict(meta):
    """One line saying whether this fit beat its own controls, in plain words.

    Reported so a reader does not have to interpret four numbers to find out
    whether the decoder decoded anything, and so the answer is recorded in the
    run log rather than reconstructed later.
    """
    mae = meta.get("validation_mae_px")
    shuffled = meta.get("shuffled_control_mae_px")
    blind = meta.get("predict_the_mean_mae_px")
    if mae is None or shuffled is None or blind is None:
        return "not scored"
    if mae < 0.9 * shuffled and mae < 0.9 * blind:
        return (f"decodes offset: {mae:.0f} px against {shuffled:.0f} px "
                f"shuffled and {blind:.0f} px for predicting the mean")
    averaged = meta.get("condition_averaged_mae_px")
    buried = averaged is not None and averaged < 0.9 * min(shuffled, blind)
    return (
        f"NO USABLE SIGNAL: {mae:.0f} px against {shuffled:.0f} px shuffled and "
        f"{blind:.0f} px for predicting the mean. "
        + ("Averaging repeats does recover it, so the signal exists but is "
           "buried in per-observation noise; a longer observation window is "
           "the thing to try."
           if buried else
           "Averaging repeats does not recover it either, so this is not a "
           "noise problem and a longer window will not fix it.")
    )


def cmd_web(args):
    import gc
    import webbrowser

    from .motor import PopulationMotorDecoder
    from .play_calibration import calibrate_play_decoder
    from .play_session import ContinuousPlaySession
    from .web_server import GameRunner, create_server

    settings = _web_settings(args)
    lock = acquire_lock(args.out)
    server = runner = None
    try:
        if not settings.fixture and not args.motor_decoder.exists():
            from .data import verify

            dataset_info = verify()
            print(json.dumps({
                "calibrating": True,
                "reason": "population decoder artifact not found",
                "path": str(args.motor_decoder),
            }), flush=True)
            motor = calibrate_play_decoder(
                settings,
                args.motor_decoder,
                dataset_info=dataset_info,
                progress=lambda done, total, seconds: print(json.dumps({
                    "calibrating_population": done,
                    "of": total,
                    "compute_seconds": seconds,
                }), flush=True),
            )
            gc.collect()
        else:
            motor = (
                None
                if settings.fixture
                else PopulationMotorDecoder.load(args.motor_decoder)
            )
            if motor is not None:
                # Every pixel-level condition, not just the window length: a
                # decoder fitted on egocentric frames reports confident
                # nonsense on fixed-view ones, and nothing downstream can tell.
                motor.check_compatible(settings)

        dataset_info, controller, fixture_decoder = _open_substrate(settings)
        session = ContinuousPlaySession(
            settings=settings,
            out=args.out,
            dataset_info=dataset_info,
            controller=controller,
            fixture_decoder=fixture_decoder,
            motor=motor,
        )
        runner = GameRunner(session, tick_seconds=args.tick_seconds)
        server = create_server(runner, port=args.port)
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        print(json.dumps({"url": url, "source": session.mode}), flush=True)
        runner.start()
        if not args.no_open:
            webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped; continuous game state preserved.", flush=True)
    finally:
        if runner is not None:
            runner.stop()
        if server is not None:
            server.server_close()
        release_lock(lock)


def _play_loop(settings, out: Path):
    """One observation, one tick: decode an action, then step the arena.

    The only value that crosses from the game into the network is the rendered
    RGB frame; the legacy character-play path decodes LEFT/RIGHT/HOLD. Outcome
    scoring happens entirely on this side of that line. The continuous web
    path uses its separate frozen population decoder for signed control.
    """
    from PIL import Image

    from .arena import render_arena
    from .breakout import BALL_D, BreakoutGame, new_game_rng
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

    if (
        controller is not None
        and settings.decoder_baseline_obs
        and not controller.decoder.calibrated
    ):
        _calibrate(controller, game, settings, out, render_arena)

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
            ball = view["ball_x"] + BALL_D / 2
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
            # This loop draws the previous action and stimulus into the frame;
            # ContinuousPlaySession does not. See play_session.py.
            "frame_annotated": True,
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


def _calibrate(controller, game, settings, out: Path, render_arena):
    """Fix the decoder's threshold on a stimulus where neither side is correct.

    The ball is drawn sitting directly on the paddle, so the frame contains no
    left-or-right answer to find; whatever difference the two DNp20 cells show
    is their standing offset, not a response to the task. The calibration block
    is written to calibration.json and its observations are not scored, do not
    move the paddle and do not advance the game.
    """
    view = dict(game.view())
    view["ball_x"] = game.paddle_center - BALL_D / 2
    view["ball_y"] = float(game.wall_bottom + 2)
    frame = render_arena(view, egocentric=settings.egocentric)

    differences = []
    for i in range(settings.decoder_baseline_obs):
        differences.append(controller.observe(frame, "none")["difference_hz"])
        if (i + 1) % 16 == 0:
            print(json.dumps({"calibrating": i + 1,
                              "of": settings.decoder_baseline_obs}), flush=True)
    threshold = controller.decoder.calibrate(differences)
    (out / "calibration.json").write_text(json.dumps({
        "observations": len(differences),
        "stimulus": "ball centred on the paddle; neither direction is correct",
        "egocentric": settings.egocentric,
        "differences_hz": differences,
        "threshold_hz": threshold,
        "note": (
            "Threshold is the balanced cut through the decoder's own output "
            "under a neutral stimulus. No target, outcome or correct direction "
            "entered this block, and none of these observations were scored."
        ),
    }, indent=2) + "\n")
    print(json.dumps({"calibrated": True, "threshold_hz": threshold,
                      "observations": len(differences)}), flush=True)


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
    elif args.command == "calibrate-play-decoder":
        cmd_calibrate_play_decoder(args)
    elif args.command == "web":
        try:
            cmd_web(args)
        except WorkerLockHeld as e:
            raise SystemExit(str(e))


if __name__ == "__main__":
    main()
