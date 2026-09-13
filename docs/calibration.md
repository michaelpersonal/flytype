# Calibration decisions

Recorded on 2026-09-11 against a freshly prepared MaleCNS v1.0 graph (166,700
neurons, 25,582,938 connections) on the implementation machine.

## Decoder deadband

Section 9 of the spec calls for "a small, declared deadband such as 0.1 Hz."
`--decoder-deadband-hz` defaults to **0.1 Hz** and was not changed.

**Check performed:** does the default frame produce nonzero visual and DNp20
activity, or does the decoder degenerate to permanent HOLD?

Running `flytype run --target 'follow @michaelzsguo on x' --seed 42` against
the real graph, the first 27 observations decoded **26 RIGHT and 1 LEFT, zero
HOLD**. Right-minus-left DNp20 rate differences observed in this run ranged
from about 2 Hz to 10 Hz in magnitude — comfortably above the 0.1 Hz deadband
in both directions. This confirms nonzero visual and DNp20 activity reaches
the decoder at the default rendering and timing; no adjustment to the deadband
or to sensory display parameters was necessary or made.

**What this does not show:** the network is strongly biased toward RIGHT
(only one LEFT decision in 27 observations). This is a directional bias in
spontaneous/visually-driven DNp20 activity, not a property tuned by us — no
parameter was adjusted based on which side happened to be correct for any
character, consistent with the constraint in section 9 ("Do not use target
correctness, target text, or reward state to override the neural proposal or
choose a replacement action") and section 15 ("Never tune based on which side
is correct"). The bias is disclosed, not hidden, in
[docs/validation.md](validation.md); a fixed decoder can still complete the
target because each character's correct side is independently randomized left
or right per attempt.

## Sensory display parameters

The 320×180 frame layout, font sizes and TARGET/TYPED/key regions in
`display.py` were chosen only for human legibility (see the manual visual
check in section 16) and were not adjusted based on decoder output or
correctness. No lamina bias, luminance gain or R8 gain constant inherited from
Stonkfly's chart renderer was changed; only the rendered content (character
glyphs and layout rather than a price chart) differs.

## Neural timing

`neural_ms=500`, `neural_bin_ms=10`, `pulse_ms=200`, `pulse_current=20` are the
inherited Stonkfly defaults (themselves inherited from the underlying DOOMFLY
kernel defaults) and were not recalibrated. They are recorded in every run's
`config.json` and `provenance.json` so any future change is auditable.

## What was not attempted

No attempt was made to reduce or eliminate the RIGHT bias observed above (for
example by adjusting the deadband asymmetrically, or by filtering which
DNp20 cells count). Doing so would risk exactly the kind of target-correctness
feedback into decoding that section 9 forbids. The bias is treated as a
disclosed property of this reconstruction and decoder, not a bug to tune away.
