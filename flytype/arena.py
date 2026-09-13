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
    BALL_SIZE,
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

    egocentric scrolls the whole playfield so the paddle sits at the centre of
    the frame. This is a change of viewpoint, not of content: the same objects
    are drawn from the same state, nothing is added, and which way to move is
    still only recoverable by finding the ball. It matters because a mean-rate
    readout reports roughly "how much light fell on the right half" -- an
    absolute screen position. The task needs the ball's position *relative to
    the paddle*, and two cells cannot subtract one from the other. Putting the
    paddle at the origin makes the quantity the eye can report and the quantity
    the task needs the same quantity. Out-of-field area is drawn as a distinct
    dead band so the walls remain findable.
    """
    im = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    d = ImageDraw.Draw(im)

    shift = 0.0
    if egocentric:
        shift = WIDTH / 2 - (view["paddle_x"] + view["paddle_width"] / 2)

    # Only the real field is live; anything beyond a wall is dead band.
    d.rectangle((0, FIELD_TOP, WIDTH - 1, FIELD_BOTTOM), fill=OUT_OF_FIELD)
    d.rectangle((shift, FIELD_TOP, shift + WIDTH - 1, FIELD_BOTTOM), fill=FIELD)
    d.line((0, FIELD_TOP, WIDTH - 1, FIELD_TOP), fill=WALL)
    d.line((0, FIELD_BOTTOM, WIDTH - 1, FIELD_BOTTOM), fill=WALL)
    for edge in (shift, shift + WIDTH - 1):
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

    columns, rows = view["columns"], view["rows"]
    brick_w = (WIDTH - 2 * FIELD_MARGIN - (columns - 1) * BRICK_GAP) / columns
    for i, alive in enumerate(view["bricks"]):
        if alive != "1":
            continue
        r, c = divmod(i, columns)
        x = shift + FIELD_MARGIN + c * (brick_w + BRICK_GAP)
        y = BRICK_TOP + r * (BRICK_HEIGHT + BRICK_GAP)
        color = ROW_COLORS[r % len(ROW_COLORS)]
        d.rectangle((x, y, x + brick_w - 1, y + BRICK_HEIGHT - 1), fill=color)

    px, pw = view["paddle_x"] + shift, view["paddle_width"]
    d.rectangle((px, PADDLE_TOP, px + pw - 1, PADDLE_TOP + PADDLE_HEIGHT - 1),
                fill=PADDLE)
    d.line((px, PADDLE_TOP, px + pw - 1, PADDLE_TOP), fill=PADDLE_EDGE)

    bx, by = view["ball_x"] + shift, view["ball_y"]
    d.rectangle((bx - 2, by - 2, bx + BALL_SIZE + 1, by + BALL_SIZE + 1),
                fill=BALL_HALO)
    d.rectangle((bx, by, bx + BALL_SIZE - 1, by + BALL_SIZE - 1), fill=BALL)

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
