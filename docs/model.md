# What is actually modeled

This is a wiring-constrained spiking-network experiment. The connectome supplies
anatomy, not a complete living fly, calibrated physiology, or a language model.
No LLM selects characters. No target-text check overrides the neural proposal
with a different action.

## Anatomy and dynamics

The [MaleCNS v1.0 release](https://male-cns.janelia.org/download/) provides the
male brain and ventral nerve cord. Import retains every assigned neuronal
superclass, including uncertain classes, while excluding explicit glia and
unresolved segmentation objects. It retains all released edges between those
entries, including weak edges and self-connections: **166,700 nodes,
25,582,938 directed connections, 124,177,617 synaptic contacts**. "Full
retained" describes this inclusion policy; it does not mean every biological
synapse was reconstructed.

The importer verifies SHA-256 source files and every compiled graph array
against committed locks. Transmitter annotations and cell order are checked
too. Neuron IDs remain integers. Source files are downloaded separately under
their upstream license.

The native kernel integrates approximate leaky integrate-and-fire cells at
**0.1 ms**. It uses 20 ms membrane and 5 ms synaptic time constants, a −45 mV
threshold, 1.8 ms transmission delay and 2.2 ms refractory period. Contact
count times 0.275 sets initial synaptic magnitude. ACh is assigned excitation;
GABA, glutamate and histamine inhibition, with explicit positive fallback for
unresolved signs. This is a coarse sign proxy, not receptor-specific
physiology. KC rest is −60 mV with an 8 mV adaptation increment decaying over
200 ms; other cells rest at −52 mV.

Pure dopamine, serotonin and octopamine annotations deliver modulatory traces
along their retained edges instead of generic fast excitation. Only the
specified memory rule consumes selected dopamine activity; most modulatory
effects are unmodeled. Cotransmission and receptors remain incomplete. Keeping
an edge in the graph does not establish that all its biological effects are
reproduced.

The event-driven kernel avoids unnecessary subthreshold updates; it does not
prune the graph or enlarge the integration timestep. This accelerates model
execution, not biological time.

## What the fly sees

A locally rendered 320×180 frame shows the target sentence, the accepted
characters typed so far and cursor, and two candidate characters in a large
left key and right key: one is the character needed next, the other a
distractor drawn from `abcdefghijklmnopqrstuvwxyz @`. The frame is generated
entirely from local program state; it is never a capture of anyone's account
or device and contains no network content.

3,335 mapped R1–R6 cells receive linear-sRGB luminance; 811 mapped R8 cells
receive blue/green proxies. Sample locations are inferred from contacts with
column-annotated visual cells, using overlapping left/right viewports.
Unmapped receptors get no invented optical input. Photoreceptors and lamina
are graded in real flies; using spikes, RGB channels, saturating current and a
12 mV-equivalent lamina bias is an explicit display adapter, not validated
retinal physiology.

Existing R8→aMe12 connections use a net excitatory sign motivated by
[Xiao et al., 2023](https://doi.org/10.1038/s41586-023-06681-6); transferring
that result to these reconstructed cells and contact-count magnitudes remains
an assumption. Display sensitivity (background brightness, key contrast) is a
major confound for how much visual activity actually reaches the network and
has not been calibrated against any ground truth.

Each observation advances a configured amount of neural time (**500 ms by
default**), regardless of elapsed wall time. The same candidate pair is
re-shown across repeated HOLD observations; a fresh distractor and a
re-randomized left/right placement are drawn only after a scored (correct or
incorrect) attempt. The renderer never receives which candidate is correct, so
it cannot leak that through color, size or position.

## How a neural spike becomes a character

Over each observation, mean right DNp20 firing minus mean left DNp20 firing is
decoded as follows:

| Neural measurement | Action |
| --- | --- |
| Difference > configured deadband (default 0.1 Hz) | RIGHT |
| Difference < −deadband | LEFT |
| Otherwise | HOLD |

This is an engineered interface, not a discovery of "typing neurons." The
mapping is fixed and reads only spike counts; it never sees the target
sentence, the typed text, or whether a choice is correct. DNpe017 spikes are
recorded for comparison in every event but never gate the decision — adding a
gate would be a new experiment and must not happen silently. Selected cell IDs
appear in the local event log. Persistent turning-like network bias can
therefore make one physical side (not one character) get selected more often;
because each character's side is independently randomized per attempt, that
bias does not by itself determine which characters get typed.

A HOLD is recorded but never counted as an attempt and never changes the
candidate pair. Only LEFT/RIGHT decisions are scored as correct or incorrect
and can advance the typed text or trigger reinforcement.

## What changes with a correct or incorrect selection

At the next observation, a correct selection schedules a **200 ms, 20
mV-equivalent** artificial current into all **15 PAM11 (α1)** cells; an
incorrect selection schedules the same pulse into the **two PPL101 (γ1pedc)**
cells. A HOLD schedules nothing. Reinforcement from observation *N* is always
delivered during observation *N+1*, never edited into the spikes that produced
observation *N*'s decision, and never used to directly edit a synaptic weight.

`--shuffle-feedback` keeps this timing (a stimulus still follows every scored
attempt) but independently randomizes whether it is reward or aversive,
disconnecting the stimulus identity from actual correctness — a control for
whether apparent "learning" tracks correctness or merely tracks *any*
dopamine pulse.

The candidate memory rule acts on **7,835 existing KC→MBON07/MBON11 edges**. It
adapts a baseline-centered anti-Hebbian rate rule from
[Huang, Luo et al., 2024](https://doi.org/10.1038/s41586-024-07819-w): recent
KC activity followed by dopamine tends to depress eligible connections; the
reverse timing can potentiate them. Actual network spikes supply KC/DAN rates
in bins of at most 10 ms. No target text or correctness value directly edits a
synaptic weight — only the identity of the scheduled stimulus.

The 1-second eligibility traces, 1,800-second memory decay, 50 ms efficacy
filter, gain 0.001 and efficacy bounds of 0.1–2× baseline are declared model
choices. The anatomy-derived DAN-to-MBON contact fractions distribute
modulation within each compartment. They are not measured dopamine
concentrations or receptor kinetics.

The PAM11/MBON07 compartment is motivated by
[Ichinose et al., 2015](https://elifesciences.org/articles/10719). Applying one
rule to both α1 and γ1pedc compartments in this male reconstruction is **our
unvalidated extension**, not a replication of either paper. Real fly dopamine
can have context-dependent effects. "Correct-selection dopamine" and
"incorrect-selection dopamine" are engineered assignments. Pain receptors,
subjective pain, pleasure and consciousness are not modeled or measured.

## What would count as evidence of learning

The implementation can demonstrate that sensory input reaches memory cells,
that selected dopamine cells spike, and that temporal pairing changes eligible
synapses. Those are mechanism checks. Because the correct candidate is
repeatedly presented against a distractor and its side is independently
randomized every attempt, **the target sentence can complete even under a
fixed directional bias with no plasticity at all** — completion is a
demonstration of the closed sensorimotor loop, not evidence of learning.

To claim learned character selection would require multiple seeds, independent
initial states, held-out trials, frozen-weight (`--frozen`) and
shuffled-feedback (`--shuffle-feedback`) controls, and a prespecified
statistical analysis comparing attempts-per-character or accuracy across
conditions. Avoid selecting a lucky seed or tuning parameters based on which
run finished fastest.

**No learning, strategy improvement, biological replication, or
independent choice by the network has been demonstrated by this repository's
tests.** See [validation](validation.md) for the narrower checks actually
performed and [calibration](calibration.md) for how sensory/decoder parameters
were chosen.
