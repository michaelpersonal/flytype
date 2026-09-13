"""Brick-breaker world state, physics and scoring.

Nothing here reads neural state, and neural code never reads this module: the
two only meet at the play-session boundary, which passes a decoded signed
control in [-1, 1] and a plain correct/incorrect/hold outcome out. The paddle
moves only as far as the decoded control says; no tick ever nudges it toward
the ball. The legacy CLI still accepts LEFT/RIGHT/HOLD actions.

Geometry is dictated by the eye, not by taste. The MaleCNS retina adapter maps
its photoreceptors across the frame very unevenly -- of 3,335 mapped R1-R6
cells, roughly 2,000 land in the top third and the bottom-left quadrant gets
none at all. Every object the network has to see therefore lives inside the band
y=12..104, and objects are sized so that at least a handful of photoreceptors
fall on them at any horizontal position. Placing the paddle at a conventional
y=165 would have put it in a region the simulated eye does not sample.

Measured for the current geometry: a 16 px ball inside this band draws a mean
of 22.8 mapped photoreceptors and is invisible at 1.7% of reachable positions,
against 1.9 and 67% for a small ball on a full-height field. The worst decile
still only gets about 4, which is the cost of the smaller ball and the reason
these numbers are worth re-measuring whenever the geometry moves.
"""

import math
import random

from .task import restore_random_state

WIDTH, HEIGHT = 320, 180

# The mapped-photoreceptor band. Measured, not chosen: see docs/model.md.
# Deep enough for a round ball to be round and still have room to fall.
FIELD_TOP = 12
FIELD_BOTTOM = 104

HUD_HEIGHT = 11
BRICK_TOP = 13
BRICK_HEIGHT = 6
BRICK_GAP = 3
# The wall spans the middle of the field rather than wall-to-wall: it leaves
# the ball open lanes down both sides, and a full-width slab of colour in the
# most densely sampled rows of the frame swamps the ball it has to compete with.
WALL_SPAN = 0.66
FIELD_MARGIN = (1 - WALL_SPAN) * WIDTH / 2

PADDLE_TOP = 92
PADDLE_HEIGHT = 9
# The renderer and physics share this diameter. Collision tests use the circle
# itself, not its square bounding box, so a near-corner miss stays a miss.
BALL_D = 16


class BreakoutGame:
    """One brick-breaker episode driven by a signed control stream.

    Each observation is exactly one tick: the paddle takes one step in the
    decoded direction (or stands still on HOLD), then the ball advances,
    bounces and resolves collisions.

    Difficulty is set so the paddle *can* keep up. The ball needs
    `descent_ticks` observations to fall from the brick wall to the paddle, and
    in that time a paddle stepping `paddle_speed` px covers
    paddle_speed * descent_ticks px of the 320 px field. At the defaults that is
    280 px, so a decoder that tracks well is rewarded and one that does not
    cannot bluff its way through. Every one of these numbers is written to
    config.json.
    """

    def __init__(self, settings, rng):
        self.rng = rng
        self.paddle_width = settings.paddle_width
        self.paddle_speed = settings.paddle_speed
        self.ball_speed = settings.ball_speed
        self.descent_ticks = settings.ball_descent_ticks
        self.rows = settings.brick_rows
        self.columns = settings.brick_columns
        self.lives_start = settings.lives

        span = WIDTH - 2 * FIELD_MARGIN
        self.brick_width = (span - (self.columns - 1) * BRICK_GAP) / self.columns
        self.wall_bottom = BRICK_TOP + self.rows * (BRICK_HEIGHT + BRICK_GAP)
        # Vertical rate that spends `descent_ticks` crossing the open field.
        self.ball_vy_magnitude = (
            PADDLE_TOP - BALL_D - self.wall_bottom
        ) / self.descent_ticks

        self.bricks = [True] * (self.rows * self.columns)
        self.paddle_x = (WIDTH - self.paddle_width) / 2
        self.lives = self.lives_start
        self.score = 0
        self.bricks_broken = 0
        self.tick = 0
        self.paddle_hits = 0
        self.misses = 0
        self.toward = 0
        self.away = 0
        self.last_event = "start"
        self.last_control = 0.0
        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self._serve()

    # ---------------------------------------------------------------- state

    @property
    def bricks_left(self):
        return sum(self.bricks)

    @property
    def cleared(self):
        return self.bricks_left == 0

    @property
    def done(self):
        return self.lives <= 0 or self.cleared

    @property
    def paddle_center(self):
        return self.paddle_x + self.paddle_width / 2

    def brick_box(self, i):
        r, c = divmod(i, self.columns)
        x = FIELD_MARGIN + c * (self.brick_width + BRICK_GAP)
        y = BRICK_TOP + r * (BRICK_HEIGHT + BRICK_GAP)
        return x, y, x + self.brick_width, y + BRICK_HEIGHT

    def view(self):
        """Everything a renderer or a log needs; no correctness hints."""
        return {
            "tick": self.tick,
            "paddle_x": self.paddle_x,
            "paddle_width": self.paddle_width,
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "ball_diameter": BALL_D,
            "ball_vx": self.ball_vx,
            "ball_vy": self.ball_vy,
            "bricks": "".join("1" if b else "0" for b in self.bricks),
            "rows": self.rows,
            "columns": self.columns,
            "lives": self.lives,
            "score": self.score,
            "bricks_left": self.bricks_left,
            "last_event": self.last_event,
            "paddle_control": self.last_control,
        }

    # ---------------------------------------------------------------- serve

    def _serve(self):
        """Drop the next ball at a seeded point anywhere across the field.

        Serving above the paddle would park the whole episode wherever the
        paddle happened to be: the paddle returns the ball roughly where it
        caught it, so the action never leaves that spot and most of the field
        goes unused. Serving across the full width makes every life a fresh
        approach the paddle has to travel to, which is also the only way a
        decoder this slow ever gets tested against a long traverse.
        """
        margin = BALL_D / 2 + 4
        self.ball_x = margin + self.rng.random() * (WIDTH - 2 * margin - BALL_D)
        self.ball_y = self.wall_bottom + 2
        self.ball_vx = self.ball_speed * (0.55 + 0.45 * self.rng.random())
        if self.rng.random() < 0.5:
            self.ball_vx = -self.ball_vx
        self.ball_vy = self.ball_vy_magnitude

    # ----------------------------------------------------------------- step

    def tracking_direction(self):
        """Which way the ball lies, or None when the paddle is already under it.

        Scored against the direction rather than against the change in gap:
        a paddle stepping 14 px routinely overshoots a ball it is chasing
        correctly, and gap-change would score that good move as a mistake. The
        dead zone is half a paddle step, so ticks where neither direction is
        meaningfully better go unscored instead of being counted as coin flips.
        """
        gap = (self.ball_x + BALL_D / 2) - self.paddle_center
        if abs(gap) < self.paddle_speed / 2:
            return None
        return "RIGHT" if gap > 0 else "LEFT"

    def step(self, action):
        """Advance one tick from a decoded action; return the scored outcome.

        Returns "correct", "incorrect" or "hold" from this tick's geometry
        alone. Catching the ball, breaking a brick and dropping the ball take
        precedence over the tracking signal.
        """
        if action not in ("LEFT", "RIGHT", "HOLD"):
            raise ValueError("Unknown action")
        return self._step_control({"LEFT": -1.0, "RIGHT": 1.0, "HOLD": 0.0}[action], action)

    def step_control(self, control):
        """Advance one tick using a signed, graded paddle control in [-1, 1]."""
        if isinstance(control, bool) or not isinstance(control, (int, float)):
            raise ValueError("Control must be a finite number in [-1, 1]")
        control = float(control)
        if not math.isfinite(control) or not -1.0 <= control <= 1.0:
            raise ValueError("Control must be a finite number in [-1, 1]")
        action = "HOLD" if control == 0 else "RIGHT" if control > 0 else "LEFT"
        return self._step_control(control, action)

    def _step_control(self, control, action):
        if self.done:
            raise RuntimeError("Episode already over")

        direction = self.tracking_direction()
        self.last_control = control
        self.paddle_x += self.paddle_speed * control
        self.paddle_x = min(max(self.paddle_x, 0.0), WIDTH - self.paddle_width)

        scored = action != "HOLD" and direction is not None
        if scored:
            if action == direction:
                self.toward += 1
            else:
                self.away += 1

        broke, caught, lost = self._advance_ball()
        self.tick += 1

        if lost:
            self.last_event = "lost"
            return "incorrect"
        if broke:
            self.last_event = "brick"
            return "correct"
        if caught:
            self.last_event = "caught"
            return "correct"
        if not scored:
            self.last_event = "hold" if action == "HOLD" else "aligned"
            return "hold"
        if action == direction:
            self.last_event = "toward"
            return "correct"
        self.last_event = "away"
        return "incorrect"

    def _advance_ball(self):
        """Move the ball in sub-steps so it cannot tunnel through a wall."""
        steps = max(1, int(abs(self.ball_vx) // 2) + 1)
        broke = caught = lost = False
        for _ in range(steps):
            self.ball_x += self.ball_vx / steps
            self.ball_y += self.ball_vy / steps

            if self.ball_x <= 0:
                self.ball_x = 0.0
                self.ball_vx = -self.ball_vx
            elif self.ball_x + BALL_D >= WIDTH:
                self.ball_x = WIDTH - BALL_D
                self.ball_vx = -self.ball_vx
            if self.ball_y <= FIELD_TOP:
                self.ball_y = float(FIELD_TOP)
                self.ball_vy = abs(self.ball_vy)

            if self._hit_brick():
                broke = True
            if self._hit_paddle():
                caught = True

            if self.ball_y > FIELD_BOTTOM:
                lost = True
                self.lives -= 1
                self.misses += 1
                if not self.done:
                    self._serve()
                return broke, caught, lost
            if self.cleared:
                return broke, caught, lost
        return broke, caught, lost

    def _hit_brick(self):
        for i, alive in enumerate(self.bricks):
            if not alive:
                continue
            x0, y0, x1, y1 = self.brick_box(i)
            if not self._ball_intersects_box(x0, y0, x1, y1):
                continue
            self.bricks[i] = False
            self.bricks_broken += 1
            self.score += 10 * (self.rows - i // self.columns)
            # Send it back down the field; the wall sits above the open area.
            self.ball_vy = abs(self.ball_vy)
            self.ball_y = max(self.ball_y, y1)
            return True
        return False

    def _hit_paddle(self):
        if self.ball_vy <= 0:
            return False
        bx0, bx1 = self.ball_x, self.ball_x + BALL_D
        by1 = self.ball_y + BALL_D
        if by1 < PADDLE_TOP or self.ball_y > PADDLE_TOP + PADDLE_HEIGHT:
            return False
        if not self._ball_intersects_box(
            self.paddle_x,
            PADDLE_TOP,
            self.paddle_x + self.paddle_width,
            PADDLE_TOP + PADDLE_HEIGHT,
        ):
            return False
        # Classic breakout deflection: where it lands sets the outgoing angle.
        offset = ((bx0 + bx1) / 2 - self.paddle_center) / (self.paddle_width / 2)
        offset = min(max(offset, -1.0), 1.0)
        if abs(offset) < 0.12:
            offset = 0.12 if self.rng.random() < 0.5 else -0.12
        # A high floor on the outgoing angle: a near-vertical return would drop
        # the ball straight back onto the paddle and the rally would never move.
        self.ball_vx = self.ball_speed * (0.70 + 0.30 * abs(offset))
        if offset < 0:
            self.ball_vx = -self.ball_vx
        self.ball_vy = -self.ball_vy_magnitude
        self.ball_y = PADDLE_TOP - BALL_D
        self.paddle_hits += 1
        return True

    def _ball_intersects_box(self, x0, y0, x1, y1):
        """Return whether the rendered circle touches an axis-aligned box."""
        radius = BALL_D / 2
        cx = self.ball_x + radius
        cy = self.ball_y + radius
        nearest_x = min(max(cx, x0), x1)
        nearest_y = min(max(cy, y0), y1)
        dx = cx - nearest_x
        dy = cy - nearest_y
        return dx * dx + dy * dy <= radius * radius

    # ------------------------------------------------------- serialization

    def summary(self):
        moves = self.toward + self.away
        return {
            "ticks": self.tick,
            "score": self.score,
            "bricks_broken": self.bricks_broken,
            "bricks_left": self.bricks_left,
            "paddle_hits": self.paddle_hits,
            "balls_lost": self.misses,
            "lives_left": self.lives,
            "cleared": self.cleared,
            "moves_toward_ball": self.toward,
            "moves_away_from_ball": self.away,
            "scored_moves": moves,
            "tracking_rate": (self.toward / moves) if moves else None,
            "tracking_chance_level": 0.5,
        }

    def _geometry(self):
        return {
            "paddle_width": self.paddle_width,
            "paddle_speed": self.paddle_speed,
            "ball_speed": self.ball_speed,
            "ball_diameter": BALL_D,
            "descent_ticks": self.descent_ticks,
            "rows": self.rows,
            "columns": self.columns,
        }

    def state(self):
        return {
            "bricks": "".join("1" if b else "0" for b in self.bricks),
            "paddle_x": self.paddle_x,
            "ball": [self.ball_x, self.ball_y, self.ball_vx, self.ball_vy],
            "lives": self.lives,
            "score": self.score,
            "bricks_broken": self.bricks_broken,
            "tick": self.tick,
            "paddle_hits": self.paddle_hits,
            "misses": self.misses,
            "toward": self.toward,
            "away": self.away,
            "last_event": self.last_event,
            "last_control": self.last_control,
            "geometry": self._geometry(),
            "rng_state": list(self.rng.getstate()),
        }

    def restore(self, state):
        if state["geometry"] != self._geometry():
            raise ValueError("Saved arena geometry does not match this run")
        if len(state["bricks"]) != len(self.bricks):
            raise ValueError("Saved brick layout does not match this run")
        self.bricks = [c == "1" for c in state["bricks"]]
        self.paddle_x = state["paddle_x"]
        self.ball_x, self.ball_y, self.ball_vx, self.ball_vy = state["ball"]
        self.lives = state["lives"]
        self.score = state["score"]
        self.bricks_broken = state["bricks_broken"]
        self.tick = state["tick"]
        self.paddle_hits = state["paddle_hits"]
        self.misses = state["misses"]
        self.toward = state["toward"]
        self.away = state["away"]
        self.last_event = state["last_event"]
        self.last_control = float(state.get("last_control", 0.0))
        self.rng.setstate(restore_random_state(state["rng_state"]))


def new_game_rng(seed):
    return random.Random(seed ^ 0xB12CE5)
