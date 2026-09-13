"""Persisted, observation-at-a-time Breakout sessions for CLI and web use."""

import base64
import io
import json
from pathlib import Path

from PIL import Image

from .arena import render_arena
from .breakout import (
    BALL_D,
    BRICK_GAP,
    BRICK_HEIGHT,
    BRICK_TOP,
    FIELD_BOTTOM,
    FIELD_MARGIN,
    FIELD_TOP,
    PADDLE_HEIGHT,
    PADDLE_TOP,
    BreakoutGame,
    new_game_rng,
)
from .feedback import FeedbackSchedule
from .neural.common import save_json
from .state import (
    append_event,
    build_provenance,
    checkpoint_path,
    commit_state,
    file_sha256,
    load_state,
    validate_resume,
    verify_checkpoint,
    write_config,
    write_provenance,
)
from .task import new_rng


def _png_data_url(frame):
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _action_for_control(control):
    return "HOLD" if control == 0 else "RIGHT" if control > 0 else "LEFT"


class ContinuousPlaySession:
    """Own one causal neural/game stream and roll games without resetting it."""

    def __init__(
        self,
        *,
        settings,
        out,
        dataset_info,
        controller=None,
        fixture_decoder=None,
        motor=None,
    ):
        self.settings = settings
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.controller = controller
        self.fixture_decoder = fixture_decoder
        self.motor = motor
        self.mode = "fixture" if settings.fixture else "malecns"
        if settings.fixture:
            if fixture_decoder is None or controller is not None or motor is not None:
                raise ValueError("Fixture session needs only a fixture decoder")
        elif controller is None or motor is None or fixture_decoder is not None:
            raise ValueError("MaleCNS session needs a controller and population decoder")

        identity = dict(dataset_info)
        self.decoder_quality = None
        if motor is not None:
            identity["motor_decoder_sha256"] = motor.artifact_sha256
            # Carried to the browser so the page can state how well this
            # decoder actually decoded on held-out probes, beside its own
            # shuffled-label control. A page that shows a confident offset
            # without showing this is claiming more than the fit supports.
            self.decoder_quality = {
                key: motor.metadata.get(key)
                for key in (
                    "validation_mae_px",
                    "validation_correlation",
                    "shuffled_control_mae_px",
                    "shuffled_control_correlation",
                    "predict_the_mean_mae_px",
                    "offset_range_px",
                )
            }
            # Calibration runs frozen and unreinforced; play usually does not.
            # Reinforcement pulses shift firing network-wide, so the z-scoring
            # fitted during calibration is applied to a different rate
            # distribution during play -- measured as a standing offset of
            # roughly +275 px present from the first observation. Recorded
            # rather than corrected: silently re-centring here would be the
            # decoder reading something other than neural firing.
            self.decoder_quality["calibrated_frozen"] = bool(
                motor.metadata.get("frozen_during_calibration")
            )
            self.decoder_quality["run_frozen"] = bool(settings.frozen)
            self.decoder_quality["calibrated_reinforcement"] = motor.metadata.get(
                "reinforcement"
            )
        self.provenance = build_provenance(
            settings,
            identity,
            self.mode,
            task="play",
            decoder_description=(
                "A frozen population decoder estimates relative ball offset from "
                "selected anatomically descending MaleCNS cells, then combines "
                "successive estimates into continuous control in [-1, 1]. The "
                "runtime decoder never reads live game coordinates, correctness, "
                "score or target text. The two-cell DNp20 readout is recorded on "
                "every observation as `dnp20_action` for comparison and never "
                "gates, corrects or contributes to the control that moves the "
                "paddle."
            ) if motor is not None else None,
            claim_description=(
                "A MaleCNS connectome simulation drives a brick-breaker paddle "
                "through a frozen neural population position/motion decoder. "
                "Control is graded and observation-bound; completing or failing "
                "an episode is not evidence of learning, understanding or play."
            ) if motor is not None else None,
        )
        self.signature = self.provenance["provenance_sha256"]
        shuffle_rng = (
            new_rng(settings.seed ^ 0x5F3759DF) if settings.shuffle_feedback else None
        )
        self.feedback = FeedbackSchedule(settings.shuffle_feedback, shuffle_rng)
        self.episode = 0
        self.observation_count = 0
        self.game = BreakoutGame(settings, new_game_rng(settings.seed))
        self.last_checkpoint = None
        self.last_snapshot = None

        saved = load_state(self.out)
        if saved is None:
            write_config(self.out, settings)
            write_provenance(self.out, self.provenance)
        else:
            self._restore(saved)

    def _restore(self, saved):
        validate_resume(saved, self.signature)
        task = saved.get("task", {})
        if "game" not in task or "episode" not in task:
            raise ValueError("Run directory does not contain a continuous game state")
        self.episode = int(task["episode"])
        self.game = BreakoutGame(
            self.settings, new_game_rng(self.settings.seed + self.episode)
        )
        self.game.restore(task["game"])
        self.feedback.restore(saved["feedback"])
        self.observation_count = int(saved["observation_count"])
        if self.fixture_decoder is not None and saved.get("fixture_decoder") is not None:
            self.fixture_decoder.restore(saved["fixture_decoder"])
        if self.controller is not None and saved.get("checkpoint"):
            self.last_checkpoint = saved["checkpoint"]
            path = self.out / self.last_checkpoint["file"]
            verify_checkpoint(path, self.last_checkpoint["sha256"])
            self.controller.restore(path)
        if self.controller is not None and saved.get("decoder"):
            self.controller.decoder.restore(saved["decoder"])
        if self.motor is not None and saved.get("motor") is not None:
            self.motor.restore(saved["motor"])

    def _task_state(self):
        return {"episode": self.episode, "game": self.game.state()}

    def _commit(self):
        commit_state(
            self.out,
            provenance_sha256=self.signature,
            observation_count=self.observation_count,
            task_state=self._task_state(),
            feedback_state=self.feedback.state(),
            fixture_decoder_state=(
                self.fixture_decoder.state() if self.fixture_decoder is not None else None
            ),
            decoder_state=(
                self.controller.decoder.state() if self.controller is not None else None
            ),
            motor_state=self.motor.state() if self.motor is not None else None,
            checkpoint=self.last_checkpoint,
            halted=None,
        )

    def _rollover_if_done(self):
        if not self.game.done:
            return
        record = {
            "episode": self.episode,
            "ending_observation": self.observation_count,
            "summary": self.game.summary(),
            "source": self.mode,
        }
        path = self.out / "episodes" / f"episode-{self.episode:06d}.json"
        if path.exists() and json.loads(path.read_text()) != record:
            raise ValueError(f"Episode archive mismatch: {path}")
        save_json(path, record)
        self.episode += 1
        self.game = BreakoutGame(
            self.settings, new_game_rng(self.settings.seed + self.episode)
        )
        if self.motor is not None:
            self.motor.forget_motion()
        self._commit()

    def advance(self):
        """Compute, persist, and publish exactly one causal observation."""
        self._rollover_if_done()
        before = self.game.view()
        frame = render_arena(
            before,
            egocentric=self.settings.egocentric,
        )
        deliver = self.feedback.pending

        if self.fixture_decoder is not None:
            ball = before["ball_x"] + BALL_D / 2
            toward = "LEFT" if ball < self.game.paddle_center else "RIGHT"
            neural = self.fixture_decoder.decode(toward)
            control = {"LEFT": -1.0, "RIGHT": 1.0, "HOLD": 0.0}[neural["action"]]
            motor = {
                "offset_px": ball - self.game.paddle_center,
                "relative_motion_px": 0.0,
                "raw_control": control,
                "control": control,
                "cell_count": 0,
                "decoder_sha256": None,
            }
        else:
            neural = self.controller.observe(
                frame,
                deliver,
                population_ids=self.motor.cell_ids,
            )
            rates = neural.pop("population_rates")
            motor = self.motor.decode(rates)
            control = motor["control"]

        action = _action_for_control(control)
        outcome = self.game.step_control(control)
        # A lost ball re-serves anywhere across the field, so the next offset
        # estimate is not a continuation of this one.
        if self.motor is not None and self.game.last_event == "lost":
            self.motor.forget_motion()
        _, scheduled = self.feedback.advance(outcome)
        self.observation_count += 1

        if self.controller is not None:
            slot = (self.observation_count - 1) % 2
            path = checkpoint_path(self.out, slot)
            self.controller.save(path)
            self.last_checkpoint = {"file": path.name, "sha256": file_sha256(path)}

        after = self.game.view()
        event = {
            "episode": self.episode,
            "observation": self.observation_count,
            "before": before,
            "after": after,
            "neural": neural,
            "motor": motor,
            # `action` is the paddle's; `dnp20_action` is what the two-cell
            # DNp20 readout would have chosen from the same observation. In
            # MaleCNS mode these are independent decoders reading one frame and
            # they routinely disagree. Only `action` moved the paddle.
            "action": action,
            "dnp20_action": neural.get("action"),
            "outcome": outcome,
            "event": self.game.last_event,
            "delivered_stimulus": deliver,
            "scheduled_next_stimulus": scheduled,
            # This loop renders the observation frame unannotated; the legacy
            # `flytype play` loop draws the previous action and stimulus below
            # the field. A replay tool cannot reproduce the exact bytes the
            # network saw without knowing which, so the frame says so itself.
            "frame_annotated": False,
            "score": self.game.score,
            "lives": self.game.lives,
            "bricks_left": self.game.bricks_left,
            "mode": self.mode,
            "source": neural["source"],
        }
        append_event(self.out, event)
        Image.fromarray(frame).save(self.out / "latest-input.png")
        self._commit()
        self.last_snapshot = self._snapshot(event, frame)
        return self.last_snapshot

    def _snapshot(self, event, frame):
        return {
            "episode": self.episode,
            "observation": self.observation_count,
            "source": event["source"],
            "status": "committed",
            "action": event["action"],
            "outcome": event["outcome"],
            "event": event["event"],
            "world": event["after"],
            "sensory_world": event["before"],
            "motor": event["motor"],
            "decoder_quality": self.decoder_quality,
            "neural": event["neural"],
            "retinal_png": _png_data_url(frame),
            "geometry": {
                "width": 320,
                # The viewpoint the model is actually shown, so the page can
                # describe its own sensory panel truthfully instead of assuming.
                "egocentric": bool(self.settings.egocentric),
                "field_top": FIELD_TOP,
                "field_bottom": FIELD_BOTTOM,
                "field_margin": FIELD_MARGIN,
                "paddle_top": PADDLE_TOP,
                "paddle_height": PADDLE_HEIGHT,
                "ball_diameter": BALL_D,
                "brick_top": BRICK_TOP,
                "brick_height": BRICK_HEIGHT,
                "brick_gap": BRICK_GAP,
            },
            "summary": self.game.summary(),
        }
