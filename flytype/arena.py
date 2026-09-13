"""Render one 320x180 RGB arena frame -- the network's entire view of the game.

The renderer is given the world state and nothing else. It has no idea which
way the paddle "should" move, so it cannot mark a preferred direction even by
accident: where the ball sits on screen is the only cue, exactly as it would be
for anything else watching the same display.

Two constraints are physiological rather than aesthetic. The playfield occupies
only the top third of the frame because that is where the mapped R1-R6 and R8
photoreceptors actually sample (see breakout.py). Colors are chosen so the ball
is the brightest object in the field and the paddle carries strong green and
blue, which are the two channels the mapped R8y/R8p subtypes read
(see neural/visual.py).
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .breakout import (
    BALL_D,
    BRICK_GAP,
    BRICK_HEIGHT,
    BRICK_TOP,
    FIELD_BOTTOM,
    FIELD_MARGIN,
    FIELD_TOP,
    HEIGHT,
    HUD_HEIGHT,
    PADDLE_HEIGHT,
    PADDLE_TOP,
    WIDTH,
)

BACKGROUND = (6, 8, 18)
FIELD = (13, 18, 38)
OUT_OF_FIELD = (3, 4, 10)
HUD_TEXT = (150, 168, 210)
WALL = (44, 58, 104)
BALL = (255, 255, 255)
BALL_HALO = (140, 200, 255)
PADDLE = (60, 255, 210)
PADDLE_EDGE = (200, 255, 245)
ROW_COLORS = [
    (255, 92, 120),
    (255, 170, 72),
    (246, 230, 92),
    (126, 214, 128),
    (108, 178, 255),
    (186, 142, 255),
]
# Offsets span +/- one field width; half scale keeps all of it on a 320 px frame.
EGO_SCALE = 0.5

STIMULUS_COLORS = {
    "NONE": (96, 112, 152),
    "REWARD": (0, 190, 122),
    "AVERSIVE": (214, 64, 80),
}


def _font(size):
    return ImageFont.load_default(size=size)


def render_arena(view, action="NONE", stimulus="NONE", egocentric=False):
    """Build the observation frame from a BreakoutGame.view() dictionary.

    action/stimulus annotate what already happened on the previous tick. They
    are drawn below the playfield, outside the sampled band, and never describe
    the current correct direction.

    egocentric draws the paddle at the centre of the frame and the ball at its
    offset from the paddle, at half scale so the entire reachable range of
    offsets (+/- one field width) always fits on screen. This is a change of
    viewpoint, not of content: the same objects come from the same state,
    nothing is added, and which way to move is still only recoverable by finding
    the ball.

    It matters because a mean-rate readout reports roughly "how much light fell
    on the right half" -- an absolute screen position. The task needs the ball's
    position *relative to the paddle*, and two cells cannot subtract one from the
    other; putting the paddle at the origin makes those the same quantity.

    The half scale is not cosmetic. A rigid scroll pushes the ball off the frame
    as soon as the paddle is more than half a field away from it, and a decoder
    that cannot see the ball falls back on its own side bias -- which drives the
    paddle further from the ball, off the edge of its own view, and holds it
    against a wall for the rest of the episode. Compressing the offset keeps the
    ball on screen at every separation the game can produce.
    """
    im = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    d = ImageDraw.Draw(im)

    # Egocentric: paddle pinned to frame centre, ball placed at its offset
    # from the paddle compressed by EGO_SCALE so it never leaves the frame.
    ball_d = view.get("ball_diameter", BALL_D)
    paddle_left = view["paddle_x"]
    ball_left = view["ball_x"]
    if egocentric:
        centre = view["paddle_x"] + view["paddle_width"] / 2
        paddle_left = WIDTH / 2 - view["paddle_width"] / 2
        ball_left = (
            WIDTH / 2
            + (view["ball_x"] + ball_d / 2 - centre) * EGO_SCALE
            - ball_d / 2
        )

    d.rectangle((0, FIELD_TOP, WIDTH - 1, FIELD_BOTTOM), fill=FIELD)
    d.line((0, FIELD_TOP, WIDTH - 1, FIELD_TOP), fill=WALL)
    d.line((0, FIELD_BOTTOM, WIDTH - 1, FIELD_BOTTOM), fill=WALL)
    if not egocentric:
        for edge in (0, WIDTH - 1):
            d.line((edge, FIELD_TOP, edge, FIELD_BOTTOM), fill=WALL)

    font = _font(10)
    d.text((6, 0), f"SCORE {view['score']:05d}", font=font, fill=HUD_TEXT)
    text = f"BRICKS {view['bricks_left']:02d}"
    d.text((WIDTH / 2 - d.textlength(text, font=font) / 2, 0), text,
           font=font, fill=HUD_TEXT)
    lives = "o" * max(0, view["lives"])
    d.text((WIDTH - 8 - d.textlength(lives, font=font), 0), lives,
           font=font, fill=HUD_TEXT)
    assert HUD_HEIGHT <= FIELD_TOP

    # The brick wall is scenery for this decision: the network controls the
    # paddle, not the wall, and in the egocentric view it would scroll with
    # every step the paddle takes. That self-induced motion is a large, bright
    # change in the most densely sampled rows of the frame, and it drove the
    # readout the opposite way from whatever the paddle had just done -- a
    # negative feedback loop with a measured lag-1 autocorrelation of -0.47,
    # which pinned the paddle in place however good the ball signal was. Under
    # the egocentric view the only thing that moves is the ball, relative to a
    # paddle that never moves in frame.
    if not egocentric:
        columns = view["columns"]
        brick_w = (WIDTH - 2 * FIELD_MARGIN - (columns - 1) * BRICK_GAP) / columns
        for i, alive in enumerate(view["bricks"]):
            if alive != "1":
                continue
            r, c = divmod(i, columns)
            x = FIELD_MARGIN + c * (brick_w + BRICK_GAP)
            y = BRICK_TOP + r * (BRICK_HEIGHT + BRICK_GAP)
            d.rectangle((x, y, x + brick_w - 1, y + BRICK_HEIGHT - 1),
                        fill=ROW_COLORS[r % len(ROW_COLORS)])

    px, pw = paddle_left, view["paddle_width"]
    d.rectangle((px, PADDLE_TOP, px + pw - 1, PADDLE_TOP + PADDLE_HEIGHT - 1),
                fill=PADDLE)
    d.line((px, PADDLE_TOP, px + pw - 1, PADDLE_TOP), fill=PADDLE_EDGE)

    bx, by = ball_left, view["ball_y"]
    d.ellipse((bx - 2, by - 2, bx + ball_d + 1, by + ball_d + 1), fill=BALL_HALO)
    d.ellipse((bx, by, bx + ball_d - 1, by + ball_d - 1), fill=BALL)

    # Human-readable strip. It sits below FIELD_BOTTOM, in the part of the
    # frame the mapped photoreceptors barely touch, so it cannot act as a cue.
    small = _font(11)
    d.text((6, FIELD_BOTTOM + 10), f"ACTION {action}", font=small,
           fill=(120, 136, 176))
    text = f"STIMULUS {stimulus.upper()}"
    d.text((6, FIELD_BOTTOM + 26), text, font=small,
           fill=STIMULUS_COLORS.get(stimulus.upper(), STIMULUS_COLORS["NONE"]))
    d.text((6, HEIGHT - 16),
           "playfield confined to the mapped photoreceptor band",
           font=_font(9), fill=(58, 68, 96))

    return np.asarray(im, dtype=np.uint8)
