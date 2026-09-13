"""Render one 320x180 RGB observation frame.

The two candidate keys are drawn from identical code with identical size,
color, border and position band; only their text content differs. This
function never receives which candidate is correct, so it cannot leak that
through styling even by accident.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 320, 180
BACKGROUND = (235, 240, 249)
HEADER = (19, 36, 71)
HEADER_TEXT = (219, 229, 249)
INK = (28, 46, 82)
MUTED = (120, 134, 163)
KEY_FILL = (255, 255, 255)
KEY_BORDER = (170, 184, 214)
KEY_TEXT = (27, 39, 81)
CURSOR = (0, 101, 183)

STIMULUS_COLORS = {
    "NONE": MUTED,
    "REWARD": (0, 128, 74),
    "AVERSIVE": (176, 32, 45),
}


def _font(size):
    return ImageFont.load_default(size=size)


def _fit(draw, text, max_width, size, minimum=8):
    while size > minimum:
        font = _font(size)
        if draw.textlength(text, font=font) <= max_width:
            return font
        size -= 1
    font = _font(minimum)
    while text and draw.textlength(text + "…", font=font) > max_width:
        text = text[:-1]
    return font, text + "…" if text != "" else text


def _draw_fitted(draw, xy, text, max_width, size, fill, minimum=8, anchor=None):
    result = _fit(draw, text, max_width, size, minimum)
    if isinstance(result, tuple):
        font, text = result
    else:
        font = result
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def label(char):
    return "SPACE" if char == " " else char


def _draw_choice(d, box, choice):
    """Render one key's contents: a single character, or a group of candidates.

    Both keys are drawn by this same function with the same colors and metrics,
    so nothing about the rendering can reveal which side is correct.
    """
    inner_w = box[2] - box[0] - 16
    inner_h = box[3] - box[1] - 16
    cx = (box[0] + box[2]) // 2
    cy = (box[1] + box[3]) // 2

    symbols = [choice] if isinstance(choice, str) else list(choice)

    if len(symbols) == 1:
        text = label(symbols[0])
        size = 56 if len(text) == 1 else 26
        font = _font(size)
        while draw_width(d, text, font) > inner_w and size > 10:
            size -= 2
            font = _font(size)
        d.text((cx, cy), text, font=font, fill=KEY_TEXT, anchor="mm")
        return

    # group of candidates: lay them out in a grid that fits the key
    glyphs = ["_" if s == " " else s for s in symbols]
    columns = max(1, min(len(glyphs), int(len(glyphs) ** 0.5 + 0.5) + 2))
    rows = (len(glyphs) + columns - 1) // columns
    size = 22
    while size > 8:
        font = _font(size)
        cw = max(draw_width(d, g, font) for g in glyphs) + 5
        line_h = size + 4
        if cw * columns <= inner_w and line_h * rows <= inner_h:
            break
        size -= 1
    font = _font(size)
    cw = max(draw_width(d, g, font) for g in glyphs) + 5
    line_h = size + 4
    x0 = cx - (cw * columns) / 2
    y0 = cy - (line_h * rows) / 2
    for i, g in enumerate(glyphs):
        r, c = divmod(i, columns)
        d.text((x0 + c * cw + cw / 2, y0 + r * line_h + line_h / 2),
               g, font=font, fill=KEY_TEXT, anchor="mm")


def render_frame(target, typed, left_char, right_char, action="NONE", stimulus="NONE"):
    """Build the observation frame.

    action: last decoded LEFT/RIGHT/HOLD/NONE (display only).
    stimulus: NONE/REWARD/AVERSIVE currently being delivered (display only).
    """
    im = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    d = ImageDraw.Draw(im)

    d.rectangle((0, 0, WIDTH - 1, 15), fill=HEADER)
    _draw_fitted(d, (6, 2), "TARGET", 40, 11, HEADER_TEXT)
    _draw_fitted(d, (52, 2), target, WIDTH - 58, 11, HEADER_TEXT)

    d.rectangle((0, 17, WIDTH - 1, 32), fill=BACKGROUND)
    _draw_fitted(d, (6, 19), "TYPED", 38, 11, MUTED)
    cursor = typed + "_"
    _draw_fitted(d, (48, 19), cursor, WIDTH - 54, 11, INK)

    gap = 8
    top, bottom = 38, 150
    key_w = (WIDTH - gap - 12) // 2
    left_box = (6, top, 6 + key_w, bottom)
    right_box = (WIDTH - 6 - key_w, top, WIDTH - 6, bottom)
    for box, choice in [(left_box, left_char), (right_box, right_char)]:
        d.rounded_rectangle(box, radius=10, fill=KEY_FILL, outline=KEY_BORDER, width=2)
        _draw_choice(d, box, choice)

    d.text((left_box[0] + 6, top + 4), "LEFT", font=_font(9), fill=MUTED)
    right_tag_w = d.textlength("RIGHT", font=_font(9))
    d.text((right_box[2] - 6 - right_tag_w, top + 4), "RIGHT", font=_font(9), fill=MUTED)

    d.rectangle((0, 154, WIDTH - 1, HEIGHT - 1), fill=BACKGROUND)
    _draw_fitted(d, (6, 158), f"ACTION {action}", 150, 11, INK)
    stim = stimulus.upper()
    stim_color = STIMULUS_COLORS.get(stim, MUTED)
    text = f"STIMULUS {stim}"
    w = d.textlength(text, font=_font(11))
    _draw_fitted(d, (WIDTH - 6 - w, 158), text, 160, 11, stim_color)

    return np.asarray(im, dtype=np.uint8)


def draw_width(draw, text, font):
    return draw.textlength(text, font=font)
