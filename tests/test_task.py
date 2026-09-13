import random

import pytest

from flytype.task import Task, restore_random_state

ALPHABET = "abcdefghijklmnopqrstuvwxyz @"


def make_task(seed=1, target="ab c"):
    return Task(target, ALPHABET, random.Random(seed))


def test_rejects_target_characters_outside_alphabet():
    with pytest.raises(ValueError):
        Task("hello!", ALPHABET, random.Random(0))


def test_distractor_never_equals_correct_character():
    task = make_task()
    for _ in range(200):
        c = task._draw()
        assert c["left"] != c["right"]
        correct = c["correct_char"]
        distractor = c["right"] if c["left"] == correct else c["left"]
        assert distractor != correct


def test_seeded_candidate_placement_is_deterministic():
    a = make_task(seed=123, target="ab c")
    b = make_task(seed=123, target="ab c")
    while not a.done:
        ca = a.candidates()
        cb = b.candidates()
        assert ca == cb
        a.score(ca["correct_side"])
        b.score(cb["correct_side"])


def test_target_advances_only_on_correct_committed_selection():
    task = make_task(target="ab")
    c = task.candidates()
    wrong = "RIGHT" if c["correct_side"] == "LEFT" else "LEFT"
    outcome = task.score(wrong)
    assert outcome == "incorrect"
    assert task.typed == ""
    assert task.index == 0


def test_incorrect_choice_does_not_alter_accepted_text():
    task = make_task(target="ab")
    before = task.typed
    c = task.candidates()
    wrong = "RIGHT" if c["correct_side"] == "LEFT" else "LEFT"
    task.score(wrong)
    assert task.typed == before


def test_correct_choice_appends_character_and_resets_attempts():
    task = make_task(target="ab")
    c = task.candidates()
    task.score("RIGHT" if c["correct_side"] == "LEFT" else "LEFT")  # one incorrect
    assert task.attempts == 1
    c = task.candidates()
    task.score(c["correct_side"])
    assert task.typed == "a"
    assert task.attempts == 0


def test_hold_does_not_score_an_attempt_or_change_candidates():
    task = make_task(target="ab")
    c1 = task.candidates()
    outcome = task.score("HOLD")
    assert outcome == "hold"
    assert task.attempts == 0
    assert task.index == 0
    c2 = task.candidates()
    assert c1 == c2


def test_space_renders_as_space_but_persists_as_literal_space():
    from flytype.display import label

    task = make_task(target=" a")
    c = task.candidates()
    assert " " in (c["left"], c["right"])
    assert label(" ") == "SPACE"
    task.score(c["correct_side"])
    assert task.typed == " "


def test_completion_requires_exact_target_equality():
    task = make_task(target="ab")
    assert not task.done
    for _ in range(2):
        c = task.candidates()
        task.score(c["correct_side"])
    assert task.done
    assert task.typed == "ab" == task.target


def test_state_round_trip_restores_index_rng_and_pending():
    task = make_task(seed=99, target="abc")
    c = task.candidates()
    task.score("RIGHT" if c["correct_side"] == "LEFT" else "LEFT")
    saved = task.state()

    restored = make_task(seed=1, target="abc")  # different seed on purpose
    restored.restore(saved)
    assert restored.index == task.index
    assert restored.attempts == task.attempts
    assert restored.pending == task.pending
    assert restored.candidates() == task.candidates()


def test_restore_rejects_mismatched_target():
    task = make_task(target="abc")
    saved = task.state()
    other = make_task(target="xyz")
    with pytest.raises(ValueError):
        other.restore(saved)


def test_restore_random_state_round_trip():
    rng = random.Random(5)
    rng.random()
    state = list(rng.getstate())
    restored = random.Random(0)
    restored.setstate(restore_random_state(state))
    assert restored.getstate() == rng.getstate()


# ---------------- binary-tree speller (full alphabet) ----------------

def make_tree(seed=1, target="ab c"):
    from flytype.task import TreeTask
    return TreeTask(target, ALPHABET, random.Random(seed))


def test_tree_commits_only_at_a_leaf_after_full_descent():
    t = make_tree(target="a")
    assert t.depth_required == 5  # 28 symbols -> ceil(log2 28)
    steps = 0
    while not t.done:
        v = t.candidates()
        t.score(v["correct_side"])
        steps += 1
    assert steps == 5
    assert t.typed == "a"


def test_tree_narrows_candidates_each_correct_choice():
    t = make_tree(target="a")
    sizes = []
    for _ in range(5):
        v = t.candidates()
        sizes.append(len(v["left"]) + len(v["right"]))
        t.score(v["correct_side"])
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] == 28 and sizes[-1] == 2


def test_tree_wrong_turn_retries_same_node_and_types_nothing():
    t = make_tree(target="a")
    for _ in range(2):
        v = t.candidates()
        t.score(v["correct_side"])
    assert t.depth == 2
    deep = len(t.remaining)
    v = t.candidates()
    wrong = "RIGHT" if v["correct_side"] == "LEFT" else "LEFT"
    assert t.score(wrong) == "incorrect"
    assert t.typed == ""
    assert t.attempts == 1
    assert t.depth == 2                 # same level: no progress surrendered
    assert len(t.remaining) == deep     # same candidate set, split redrawn


def test_tree_never_commits_a_wrong_character():
    t = make_tree(seed=7, target="xyz")
    rng = random.Random(3)
    for _ in range(4000):
        if t.done:
            break
        v = t.candidates()
        pick = v["correct_side"] if rng.random() < 0.7 else (
            "RIGHT" if v["correct_side"] == "LEFT" else "LEFT")
        t.score(pick)
        assert t.target.startswith(t.typed)  # typed text is always a valid prefix
    assert t.target.startswith(t.typed)


def test_tree_hold_does_not_advance_or_change_groups():
    t = make_tree(target="a")
    first = t.candidates()
    assert t.score("HOLD") == "hold"
    assert t.candidates() == first
    assert t.attempts == 0 and t.depth == 0


def test_tree_state_round_trip():
    t = make_tree(seed=5, target="abc")
    for _ in range(3):
        v = t.candidates()
        t.score(v["correct_side"])
    saved = t.state()
    other = make_tree(seed=99, target="abc")
    other.restore(saved)
    assert other.remaining == t.remaining
    assert other.depth == t.depth
    assert other.candidates() == t.candidates()
