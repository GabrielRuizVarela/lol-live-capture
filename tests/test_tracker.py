"""Tests for the pure-Python draft tracker (no cv2 / numpy)."""
from __future__ import annotations

from dataclasses import dataclass

from lol_live_capture.tracker import (
    SEQUENCE,
    SLOT_KEYS,
    DraftTracker,
    SlotRead,
)


@dataclass(frozen=True)
class Mention:
    t: float
    champion_id: str
    cue: str
    score: float = 1.0
    text: str = ""


def read(state, champion=None, *, margin=0.9, score=0.95, cand=()):
    return SlotRead(
        state=state,
        champion_id=champion,
        name=None,
        score=score,
        margin=margin,
        brightness=0.5,
        candidates=tuple(cand),
    )


def empty():
    return read("empty")


def frame(**overrides):
    """A full 20-slot frame; every slot empty unless overridden."""
    reads = {slot: empty() for slot in SLOT_KEYS}
    reads.update(overrides)
    return reads


def feed(tracker, reads, *, frames, start_t=0.0, step=1.0):
    """Feed the same reads for ``frames`` draft frames; returns the last t fed."""
    t = start_t
    for _ in range(frames):
        tracker.observe(t, True, reads)
        t += step
    return t


def actions(tracker):
    return tracker.state()["actions"]


def walk_full_draft(tracker, champions, *, active_frames=3, lock_frames=3, tail_frames=5):
    """Feed a whole draft: each slot glows as the turn, then locks."""
    t = 0.0
    for sequence in range(1, 21):
        slot = next(s for s, seq in SEQUENCE.items() if seq == sequence)
        reads = frame(**{s: read("locked", c) for s, c in champions.items()
                         if SEQUENCE[s] < sequence})
        reads[slot] = read("active", champions[slot])
        t = feed(tracker, reads, frames=active_frames, start_t=t)
        reads = frame(**{s: read("locked", c) for s, c in champions.items()
                         if SEQUENCE[s] <= sequence})
        t = feed(tracker, reads, frames=lock_frames, start_t=t)
    # End of draft: no slot is active anymore; slot 20 commits after
    # no_active_frames of this (plus whatever the last lock_frames already gave).
    reads = frame(**{s: read("locked", c) for s, c in champions.items()})
    t = feed(tracker, reads, frames=tail_frames, start_t=t)
    return t


def seq_slot(sequence):
    return next(s for s, seq in SEQUENCE.items() if seq == sequence)


def champions_map():
    return {slot: str(i + 1) for i, slot in enumerate(SLOT_KEYS)}


def play_draft(tracker, champions, *, skip=(), hover_reads=None, frames=3, tail=3,
               active_frames=3, lock_frames=3):
    """Play a full draft; ``skip`` slots never commit but still read ``hover``.

    ``hover_reads`` maps a skipped slot to the champion it reads while played
    (defaults to that slot's own champion), so the turn evidence can be steered.
    """
    hover_reads = hover_reads or {}
    skipped = set(skip)
    t = 0.0
    for sequence in range(1, 21):
        slot = seq_slot(sequence)
        pre = {s: read("locked", c) for s, c in champions.items()
               if SEQUENCE[s] < sequence and s not in skipped}
        pre.update({s: read("hover", hover_reads.get(s, champions[s])) for s in skipped})
        if slot not in skipped:
            pre[slot] = read("active", champions[slot])
        t = feed(tracker, frame(**pre), frames=active_frames, start_t=t)
        post = {s: read("locked", c) for s, c in champions.items()
                if SEQUENCE[s] <= sequence and s not in skipped}
        post.update({s: read("hover", hover_reads.get(s, champions[s])) for s in skipped})
        t = feed(tracker, frame(**post), frames=lock_frames, start_t=t)
    settled = {s: read("locked", c) for s, c in champions.items() if s not in skipped}
    settled.update({s: read("hover", hover_reads.get(s, champions[s])) for s in skipped})
    t = feed(tracker, frame(**settled), frames=tail, start_t=t)
    return t


def final_layout(champions, side, *, permute=True):
    """The side's five picks in a settled (role-order) layout; order permuted."""
    slots = [f"{side}_pick_{i}" for i in range(1, 6)]
    values = [champions[s] for s in slots]
    if permute:
        values = values[1:] + values[:1]
    return {slot: read("locked", value) for slot, value in zip(slots, values)}


# -- end-of-draft reconciliation ----------------------------------------


def test_final_set_resolves_one_missing_pick():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=3, no_active_frames=3)
    champions = champions_map()
    t = play_draft(tracker, champions, skip={"blue_pick_4"})
    feed(tracker, frame(**final_layout(champions, "blue")), frames=4, start_t=t)
    by_sequence = {a["sequence"]: a for a in actions(tracker)}
    assert 18 in by_sequence, "the one missing blue pick should be filled"
    assert by_sequence[18]["champion_id"] == champions["blue_pick_4"]
    assert by_sequence[18]["via"] == "final_set"


def test_two_missing_picks_resolved_by_turn_evidence():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=3, no_active_frames=3)
    champions = champions_map()
    skip = {"blue_pick_4", "blue_pick_5"}
    t = play_draft(tracker, champions, skip=skip)
    feed(tracker, frame(**final_layout(champions, "blue")), frames=4, start_t=t)
    by_sequence = {a["sequence"]: a for a in actions(tracker)}
    assert by_sequence[18]["champion_id"] == champions["blue_pick_4"]
    assert by_sequence[19]["champion_id"] == champions["blue_pick_5"]
    assert by_sequence[18]["via"] == "final_set"
    assert by_sequence[19]["via"] == "final_set"


def test_ambiguous_missing_picks_left_missing():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=3, no_active_frames=3)
    champions = champions_map()
    skip = {"blue_pick_4", "blue_pick_5"}
    # Both skipped slots read blue_pick_1: neither is in the leftover set, so no
    # unique and consistent assignment exists.
    hover = {slot: champions["blue_pick_1"] for slot in skip}
    t = play_draft(tracker, champions, skip=skip, hover_reads=hover)
    feed(tracker, frame(**final_layout(champions, "blue")), frames=4, start_t=t)
    sequences = {a["sequence"] for a in actions(tracker)}
    assert 18 not in sequences and 19 not in sequences


def test_reorder_detected_when_two_committed_picks_change():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6, no_active_frames=3)
    champions = champions_map()
    walk_full_draft(tracker, champions, tail_frames=0)
    assert tracker._reorder_seen is False
    swapped = dict(champions)
    swapped["blue_pick_1"], swapped["blue_pick_2"] = (
        champions["blue_pick_2"],
        champions["blue_pick_1"],
    )
    reads = frame(**{s: read("locked", c) for s, c in swapped.items()})
    feed(tracker, reads, frames=4, start_t=1000.0)
    assert tracker._reorder_seen is True


def test_single_noisy_pick_read_is_not_a_reorder():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6, no_active_frames=3)
    champions = champions_map()
    walk_full_draft(tracker, champions, tail_frames=0)
    noisy = dict(champions)
    noisy["blue_pick_1"] = "999"
    reads = frame(**{s: read("locked", c) for s, c in noisy.items()})
    feed(tracker, reads, frames=4, start_t=1000.0)
    assert tracker._reorder_seen is False


def test_positional_pick_commits_stop_after_draft_over():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6, no_active_frames=3)
    champions = champions_map()
    t = play_draft(tracker, champions, skip={"red_pick_4"})
    assert tracker._draft_over is True
    # A stable locked read in the still-missing slot must not commit after the
    # draft is over (no final set can form from this single readable slot).
    feed(tracker, frame(red_pick_4=read("locked", "999")), frames=8, start_t=t)
    assert 17 not in {a["sequence"] for a in actions(tracker)}


# -- commit on advance --------------------------------------------------


def test_stable_alone_does_not_commit():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6)
    feed(tracker, frame(blue_ban_1=read("locked", "62")), frames=5)
    # Nothing higher has moved and it has not been locked for L frames yet.
    assert actions(tracker) == []


def test_commit_on_advance_of_higher_slot():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6)
    feed(tracker, frame(blue_ban_1=read("locked", "62")), frames=3)
    assert actions(tracker) == []
    # blue_ban_2 (sequence 3) becomes non-empty for K frames: blue_ban_1 commits.
    t = feed(tracker, frame(
        blue_ban_1=read("locked", "62"),
        blue_ban_2=read("active"),
    ), frames=3, start_t=3.0)
    state = actions(tracker)
    assert len(state) == 1
    assert state[0]["sequence"] == 1
    assert state[0]["champion_id"] == "62"
    # With turn_frames=2, lock happens at second frame of second feed (t=4.0)
    assert state[0]["locked_at"] == 4.0
    # blue_ban_2 itself is not committed (no higher slot has moved).
    assert [a["sequence"] for a in state] == [1]


def test_active_counts_as_nonempty_for_advance():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6)
    feed(tracker, frame(blue_ban_1=read("locked", "62")), frames=3)
    feed(tracker, frame(
        blue_ban_1=read("locked", "62"),
        blue_ban_2=read("active"),
    ), frames=3, start_t=3.0)
    assert [a["champion_id"] for a in actions(tracker)] == ["62"]


def test_active_ban_slot_never_reads_a_champion():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6)
    feed(tracker, frame(blue_ban_1=read("active"), blue_ban_2=read("active")), frames=6)
    assert actions(tracker) == []


def test_latest_stable_read_supersedes_a_browsed_champion():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6)
    # Feed 1: 3 frames establishing initial stable read (Anivia via locked frames).
    for i in range(3):
        reads = {slot: SlotRead('empty', None, None, 0.0, 0.0, 0.0) for slot in SLOT_KEYS}
        reads['blue_pick_1'] = SlotRead('locked', 'Anivia', None, 0.9, 0.9, 0.5)
        tracker.observe(float(i), True, reads)
    # Feed 2: 3 frames updating stable read to Vi.
    for i in range(3, 6):
        reads = {slot: SlotRead('empty', None, None, 0.0, 0.0, 0.0) for slot in SLOT_KEYS}
        reads['blue_pick_1'] = SlotRead('locked', 'Vi', None, 0.9, 0.9, 0.5)
        tracker.observe(float(i), True, reads)
    # Feed 3: red_pick_1 active (seq 8) advances the turn, committing blue_pick_1
    # with its latest stable read (Vi, not Anivia from Feed 1).
    for i in range(6, 9):
        reads = {slot: SlotRead('empty', None, None, 0.0, 0.0, 0.0) for slot in SLOT_KEYS}
        reads['blue_pick_1'] = SlotRead('locked', 'Vi', None, 0.9, 0.9, 0.5)
        reads['red_pick_1'] = SlotRead('active', 'Tristana', None, 0.5, 0.5, 0.5)  # seq 8
        tracker.observe(float(i), True, reads)
    state = tracker.state()
    actions = state['actions']
    assert len(actions) == 1
    assert actions[0]["sequence"] == 7
    # The stable read should be Vi (3 locked frames), not Anivia (Feed 1).
    assert actions[0]["champion_id"] == "Vi"


def test_streak_broken_by_different_champion():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6)
    tracker.observe(0.0, True, frame(blue_ban_1=read("locked", "1")))
    tracker.observe(1.0, True, frame(blue_ban_1=read("locked", "2")))
    tracker.observe(2.0, True, frame(blue_ban_1=read("locked", "1")))
    tracker.observe(3.0, True, frame(blue_ban_1=read("locked", "1")))
    feed(tracker, frame(blue_ban_1=read("locked", "1"), blue_ban_2=read("active")), frames=3, start_t=4.0)
    assert actions(tracker)[0]["champion_id"] == "1"


def test_streak_broken_by_empty_read():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6)
    tracker.observe(0.0, True, frame(blue_ban_1=read("locked", "1")))
    tracker.observe(1.0, True, frame())
    tracker.observe(2.0, True, frame(blue_ban_1=read("locked", "1")))
    tracker.observe(3.0, True, frame(blue_ban_1=read("locked", "1")))
    # Only two consecutive locked frames after the break: not stable yet.
    tracker.observe(4.0, True, frame(blue_ban_1=read("locked", "1"), blue_ban_2=read("active")))
    assert actions(tracker) == []
    tracker.observe(5.0, True, frame(blue_ban_1=read("locked", "1"), blue_ban_2=read("active")))
    tracker.observe(6.0, True, frame(blue_ban_1=read("locked", "1"), blue_ban_2=read("active")))
    assert actions(tracker)[0]["champion_id"] == "1"


def test_non_draft_frames_do_not_break_streak():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6)
    tracker.observe(0.0, True, frame(blue_ban_1=read("locked", "1")))
    tracker.observe(1.0, False, {})
    tracker.observe(2.0, True, frame(blue_ban_1=read("locked", "1")))
    tracker.observe(3.0, False, {})
    tracker.observe(4.0, True, frame(blue_ban_1=read("locked", "1")))
    feed(tracker, frame(blue_ban_1=read("locked", "1"), blue_ban_2=read("active")), frames=3, start_t=5.0)
    assert actions(tracker)[0]["champion_id"] == "1"


# -- last pending action ------------------------------------------------


def test_last_action_commits_after_no_active_frames():
    tracker = DraftTracker(stable_frames=3, no_active_frames=4)
    # The turn is on red_pick_5 (sequence 20) while it is hovered/empty...
    feed(tracker, frame(red_pick_5=read("active")), frames=3)
    assert actions(tracker) == []
    # ...then it locks and the glow goes away.
    t = feed(tracker, frame(red_pick_5=read("locked", "Ambessa")), frames=3, start_t=3.0)
    assert actions(tracker) == []
    # After no_active_frames consecutive frames with no active slot it commits.
    tracker.observe(t, True, frame(red_pick_5=read("locked", "Ambessa")))
    state = actions(tracker)
    assert len(state) == 1
    assert state[0]["sequence"] == 20
    assert state[0]["champion_id"] == "Ambessa"
    assert state[0]["locked_at"] == t


def test_last_action_not_committed_while_hovering():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6)
    feed(tracker, frame(red_pick_5=read("hover", "Ambessa")), frames=8)
    assert actions(tracker) == []


def test_last_action_not_committed_while_still_active():
    tracker = DraftTracker(stable_frames=3, no_active_frames=4)
    # Slot 20 stays the turn slot (glowing): never final while it is active.
    feed(tracker, frame(red_pick_5=read("active", "Ambessa")), frames=12)
    assert actions(tracker) == []


def test_reorder_cannot_change_committed_slot_20():
    tracker = DraftTracker(stable_frames=3, no_active_frames=4)
    champions = {slot: str(i + 1) for i, slot in enumerate(SLOT_KEYS)}
    # No tail: after slot 20's lock phase no_active_run is 3, below the
    # threshold of 4, so slot 20 is not committed yet.
    walk_full_draft(tracker, champions, lock_frames=3, tail_frames=0)
    assert all(a["sequence"] != 20 for a in actions(tracker))

    # The end-of-draft role reorder changes portraits before slot 20 commits.
    reordered = dict(champions)
    reordered["red_pick_5"] = "999"
    reordered["blue_pick_1"] = "998"
    feed(tracker, frame(**{s: read("locked", c) for s, c in reordered.items()}),
         frames=6, start_t=1000.0)
    state = actions(tracker)
    assert len(state) == 20
    # Slot 20 was committed with the pre-reorder read captured while it was
    # the turn, not with the reordered art.
    assert next(a for a in state if a["sequence"] == 20)["champion_id"] == champions["red_pick_5"]


def test_end_of_draft_without_stable_read_does_not_commit_wrong():
    # If slot 20 glows until the reorder and never reads locked, the tracker
    # must not commit the reordered art (a missing action beats a wrong one).
    tracker = DraftTracker(stable_frames=3, no_active_frames=4)
    champions = {slot: str(i + 1) for i, slot in enumerate(SLOT_KEYS)}
    t = 0.0
    for sequence in range(1, 21):
        slot = next(s for s, seq in SEQUENCE.items() if seq == sequence)
        reads = frame(**{s: read("locked", c) for s, c in champions.items()
                         if SEQUENCE[s] < sequence})
        reads[slot] = read("active", champions[slot])
        t = feed(tracker, reads, frames=3, start_t=t)
        if sequence < 20:
            reads = frame(**{s: read("locked", c) for s, c in champions.items()
                             if SEQUENCE[s] <= sequence})
            t = feed(tracker, reads, frames=3, start_t=t)
    # Slot 20 still glows (no locked read ever); the reorder hits now —
    # a real role reorder moves other portraits too, which marks the change.
    reordered = dict(champions)
    reordered["red_pick_5"] = champions["blue_pick_1"]
    reordered["blue_pick_1"] = champions["red_pick_5"]
    feed(tracker, frame(**{s: read("locked", c) for s, c in reordered.items()}),
         frames=8, start_t=t)
    state = actions(tracker)
    committed = {a["sequence"]: a["champion_id"] for a in state}
    assert 20 not in committed  # missing beats wrong
    assert committed.get(7) == champions["blue_pick_1"]  # committed actions stand


def test_end_of_draft_uses_read_before_reorder():
    # Slot 20 locks while glowing, then the reorder hits before the
    # no-active threshold: the committed read is the pre-reorder one.
    tracker = DraftTracker(stable_frames=3, no_active_frames=4)
    champions = {slot: str(i + 1) for i, slot in enumerate(SLOT_KEYS)}
    walk_full_draft(tracker, champions, lock_frames=3, tail_frames=0)
    assert all(a["sequence"] != 20 for a in actions(tracker))
    reordered = dict(champions)
    reordered["red_pick_5"] = "999"
    for _ in range(6):
        tracker.observe(100.0, True, frame(**{s: read("locked", c) for s, c in reordered.items()}))
    state = actions(tracker)
    committed_20 = next(a for a in state if a["sequence"] == 20)
    assert committed_20["champion_id"] == champions["red_pick_5"]


# -- full drafts and joins ----------------------------------------------


def test_join_mid_draft_commits_filled_slots():
    slots = {
        "blue_pick_1": "7",
        "red_pick_1": "8",
        "red_pick_2": "9",
        "blue_pick_2": "10",
        "blue_ban_1": "1",
        "red_ban_1": "2",
        "blue_ban_2": "3",
        "red_ban_2": "4",
    }
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6)
    # The stream joins while blue_pick_3 (sequence 11) is the turn.
    reads = frame(**{s: read("locked", c) for s, c in slots.items()})
    reads["blue_pick_3"] = read("active")
    feed(tracker, reads, frames=6)
    state = actions(tracker)
    sequences = [a["sequence"] for a in state]
    assert sequences == sorted(sequences)
    assert sequences == sorted(SEQUENCE[s] for s in slots)
    assert tracker.state()["status"] == "drafting"


def test_full_twenty_action_draft():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6)
    champions = {slot: str(i + 1) for i, slot in enumerate(SLOT_KEYS)}
    walk_full_draft(tracker, champions)
    state = tracker.state()
    assert len(state["actions"]) == 20
    assert [a["sequence"] for a in state["actions"]] == list(range(1, 21))
    assert state["status"] == "draft_complete"
    assert state["game_index"] == 1
    for action in state["actions"]:
        slot = next(s for s, seq in SEQUENCE.items() if seq == action["sequence"])
        assert action["champion_id"] == champions[slot]


def test_prehover_in_later_slot_does_not_move_the_turn():
    tracker = DraftTracker(stable_frames=3)
    # Blue is picking 1; red pre-hovers Akali in red_pick_5.
    feed(tracker, frame(
        blue_pick_1=read("locked", "103"),
        red_pick_5=read("hover", "84"),
    ), frames=3)
    assert actions(tracker) == []
    # Blue pick 1 locks; red_pick_1 becomes the turn: only then does seq 7 commit.
    reads = frame(
        blue_pick_1=read("locked", "103"),
        red_pick_5=read("hover", "84"),
        red_pick_1=read("active", "18"),
    )
    feed(tracker, reads, frames=3, start_t=3.0)
    state = actions(tracker)
    assert [a["sequence"] for a in state] == [7]
    assert state[0]["champion_id"] == "103"


def test_unknown_neither_counts_nor_breaks_streak():
    tracker = DraftTracker(stable_frames=3)
    feed(tracker, frame(blue_pick_1=read("locked", "103")), frames=2)
    tracker.observe(2.0, True, frame(
        blue_pick_1=read("unknown"),
        red_pick_1=read("active"),
    ))
    # The unknown frame did not break the streak nor advance the turn (only
    # one frame of red_pick_1 so far).
    assert actions(tracker) == []
    feed(tracker, frame(
        blue_pick_1=read("locked", "103"),
        red_pick_1=read("active"),
    ), frames=2, start_t=3.0)
    state = actions(tracker)
    assert [a["sequence"] for a in state] == [7]
    assert state[0]["champion_id"] == "103"


def test_pending_slot_commits_when_locked_arrives_late():
    # A slot whose turn is already over but which had not yet reached K locked
    # frames stays pending; it commits as soon as the streak completes.
    tracker = DraftTracker(stable_frames=3)
    feed(tracker, frame(blue_ban_1=read("locked", "62")), frames=2)
    reads = frame(blue_ban_1=read("locked", "62"), red_ban_1=read("active"))
    tracker.observe(2.0, True, reads)  # 3rd locked frame -> stable; 1st active frame
    assert actions(tracker) == []  # turn needs 2 consecutive active frames
    tracker.observe(3.0, True, reads)  # 2nd active frame -> turn 2 commits slot 1
    state = actions(tracker)
    assert [a["sequence"] for a in state] == [1]
    assert state[0]["champion_id"] == "62"


def test_committed_actions_never_change_after_reorder():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6)
    champions = {slot: str(i + 1) for i, slot in enumerate(SLOT_KEYS)}
    feed(tracker, frame(**{s: read("locked", c) for s, c in champions.items()}), frames=6)
    before = actions(tracker)
    reordered = dict(champions)
    reordered["blue_pick_1"] = "999"
    reordered["red_pick_5"] = "998"
    feed(tracker, frame(**{s: read("locked", c) for s, c in reordered.items()}), frames=5, start_t=10.0)
    assert actions(tracker) == before


# -- warnings, hover, new draft, misc -----------------------------------


def test_duplicate_champion_warning_keeps_first():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6)
    feed(tracker, frame(
        blue_ban_1=read("locked", "5"),
        blue_ban_2=read("active"),
    ), frames=3)
    feed(tracker, frame(
        blue_ban_1=read("locked", "5"),
        blue_ban_2=read("locked", "5"),
        blue_ban_3=read("active"),
    ), frames=3, start_t=3.0)
    state = tracker.state()
    assert len(state["actions"]) == 1
    assert state["actions"][0]["sequence"] == 1
    assert state["warnings"], "duplicate champion should warn"
    assert "5" in state["warnings"][0]


def test_duplicate_simultaneous_keeps_lowest_sequence():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6)
    feed(tracker, frame(
        blue_ban_1=read("locked", "5"),
        red_ban_1=read("locked", "5"),
        blue_ban_2=read("active"),
    ), frames=3)
    state = tracker.state()
    assert len(state["actions"]) == 1
    assert state["actions"][0]["sequence"] == 1
    assert state["warnings"]


def test_hover_lowest_sequence():
    tracker = DraftTracker()
    tracker.observe(
        0.0,
        True,
        frame(
            blue_pick_2=read("hover", "99", margin=0.5),
            blue_pick_1=read("hover", "42", margin=0.7),
        ),
    )
    hover = tracker.state()["hover"]
    assert hover is not None
    assert hover["sequence"] == 7
    assert hover["side"] == "blue"
    assert hover["action_type"] == "pick"
    assert hover["champion_id"] == "42"
    assert hover["confidence"] == 0.7


def test_hover_null_when_none_and_excludes_locked():
    tracker = DraftTracker(stable_frames=1)
    assert tracker.state()["hover"] is None

    tracker.observe(0.0, True, frame(blue_ban_1=read("hover", "42")))
    assert tracker.state()["hover"] is not None
    # Once committed (stable_frames=1) the slot can no longer hover.
    tracker.observe(1.0, True, frame(blue_ban_1=read("locked", "42")))
    assert tracker.state()["hover"] is None

    tracker.observe(2.0, True, frame(red_ban_1=read("hover", "7")))
    assert tracker.state()["hover"] is not None
    tracker.observe(3.0, False, {})
    assert tracker.state()["hover"] is None


def test_hover_without_champion():
    tracker = DraftTracker()
    tracker.observe(0.0, True, frame(blue_ban_1=read("hover", None)))
    hover = tracker.state()["hover"]
    assert hover is not None
    assert hover["champion_id"] is None


def test_status_before_any_draft_frame():
    tracker = DraftTracker()
    tracker.observe(0.0, False, {})
    assert tracker.state()["status"] == "waiting_for_draft"


def test_new_draft_rule_a_all_empty_after_complete():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6)
    champions = {slot: str(i + 1) for i, slot in enumerate(SLOT_KEYS)}
    walk_full_draft(tracker, champions)
    assert tracker.state()["status"] == "draft_complete"
    assert tracker.state()["game_index"] == 1

    for i in range(2):
        tracker.observe(5.0 + i, True, frame())
        assert tracker.state()["game_index"] == 1
    tracker.observe(7.0, True, frame())

    state = tracker.state()
    assert state["game_index"] == 2
    assert state["actions"] == []
    assert state["warnings"] == []


def test_new_draft_rule_b_after_idle_gap():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6, idle_reset_s=60.0)
    champions = {slot: str(i + 1) for i, slot in enumerate(SLOT_KEYS)}
    walk_full_draft(tracker, champions)
    assert tracker.state()["status"] == "draft_complete"

    tracker.observe(200.0, True, frame())
    state = tracker.state()
    assert state["game_index"] == 2
    assert state["actions"] == []
    assert state["status"] == "drafting"


def test_rule_b_not_triggered_without_gap():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6, idle_reset_s=60.0)
    feed(tracker, frame(blue_ban_1=read("locked", "1"), blue_ban_2=read("active")), frames=3)
    tracker.observe(1.0, True, frame())
    assert tracker.state()["game_index"] == 1
    assert len(tracker.state()["actions"]) == 1


def test_min_margin_filter():
    tracker = DraftTracker(stable_frames=3, final_stable_frames=6, min_margin=0.5)
    feed(tracker, frame(
        blue_ban_1=read("locked", "1", margin=0.3),
        blue_ban_2=read("active"),
    ), frames=5)
    assert tracker.state()["actions"] == []

    feed(tracker, frame(
        blue_ban_1=read("locked", "1", margin=0.9),
        blue_ban_2=read("active"),
    ), frames=3, start_t=5.0)
    state = tracker.state()["actions"]
    assert len(state) == 1
    assert state[0]["champion_id"] == "1"


def test_tricodes_keep_latest_non_none():
    tracker = DraftTracker()
    tracker.observe(0.0, True, frame(), tricodes=("KC", None))
    tracker.observe(1.0, True, frame(), tricodes=("KC2", "MVK"))
    tracker.observe(2.0, True, frame(), tricodes=(None, "MVK2"))

    teams = tracker.state()["teams"]
    assert teams["blue"]["tricode"] == "KC2"
    assert teams["red"]["tricode"] == "MVK2"


def test_reset_clears_everything():
    tracker = DraftTracker(stable_frames=1, final_stable_frames=1)
    reads = frame(blue_ban_1=read("locked", "1"), red_ban_1=read("active"))
    tracker.observe(0.0, True, reads, tricodes=("KC", "RED"))
    tracker.observe(0.5, True, reads)
    assert tracker.state()["game_index"] == 1
    assert tracker.state()["actions"]

    tracker.reset()
    state = tracker.state()
    assert state["actions"] == []
    assert state["warnings"] == []
    assert state["hover"] is None
    assert state["game_index"] == 0
    assert state["status"] == "idle"
    assert state["teams"] == {
        "blue": {"tricode": None},
        "red": {"tricode": None},
    }
    assert state["stream_t"] is None

    tracker.observe(0.0, False, {})
    assert tracker.state()["game_index"] == 0
    tracker.observe(1.0, True, frame())
    assert tracker.state()["game_index"] == 1


def test_state_shape_and_now_iso():
    tracker = DraftTracker(source="https://x", layout="worlds")
    state = tracker.state(now_iso="2026-10-18T00:00:00+00:00")
    assert state["source"] == "https://x"
    assert state["layout"] == "worlds"
    assert state["updated_at"] == "2026-10-18T00:00:00+00:00"
    assert state["frame_age_ms"] is None
    assert set(state) == {
        "status",
        "source",
        "layout",
        "updated_at",
        "stream_t",
        "frame_age_ms",
        "game_index",
        "teams",
        "actions",
        "hover",
        "suggestions",
        "warnings",
        "error",
    }


def test_observe_change_flag():
    tracker = DraftTracker(stable_frames=3)
    assert tracker.observe(0.0, True, frame()) is True
    assert tracker.observe(0.0, True, frame()) is False
    assert tracker.observe(1.0, True, frame(blue_ban_1=read("hover", "1"))) is True


def test_set_status_error_is_preserved():
    tracker = DraftTracker()
    tracker.set_status("error", "boom")
    state = tracker.state()
    assert state["status"] == "error"
    assert state["error"] == "boom"
    tracker.observe(0.0, True, frame())
    assert tracker.state()["status"] == "error"


# -- audio fusion -------------------------------------------------------


def commit_with_audio(tracker, slot, value, advance_slot, *, frames=3, wait=8):
    """Lock ``slot`` then advance the turn, leaving time for the audio window."""
    t = feed(tracker, frame(**{slot: value}), frames=frames)
    reads = frame(**{slot: value, advance_slot: read("active")})
    t = feed(tracker, reads, frames=frames, start_t=t)
    return feed(tracker, reads, frames=wait, start_t=t)


def test_fusion_a_audio_support_commits_image_champion():
    tracker = DraftTracker(stable_frames=3, turn_frames=2, audio_wait_s=5)
    tracker.observe_mentions([Mention(1.0, "62", "ban")])
    commit_with_audio(tracker, "blue_ban_1", read("locked", "62"), "blue_ban_2")
    action = actions(tracker)[0]
    assert action["champion_id"] == "62"
    assert action["via"] == "audio+image"
    assert action["confidence"] >= 0.9


def test_fusion_b_lookalike_named_by_audio_wins():
    tracker = DraftTracker(stable_frames=3, turn_frames=2, audio_wait_s=5,
                           lookalike_max_margin=3.0)
    tracker.observe_mentions([Mention(1.0, "22", "ban")])
    value = read("locked", "11", margin=1.0, score=10.0,
                 cand=(("11", 10.0), ("22", 6.0)))
    commit_with_audio(tracker, "blue_ban_1", value, "blue_ban_2")
    action = actions(tracker)[0]
    assert action["champion_id"] == "22"
    assert action["via"] == "audio+image"
    assert action["confidence"] == 0.8


def test_fusion_b_not_fired_when_image_margin_is_clear():
    tracker = DraftTracker(stable_frames=3, turn_frames=2, audio_wait_s=5,
                           lookalike_max_margin=3.0)
    tracker.observe_mentions([Mention(1.0, "22", "ban")])
    value = read("locked", "11", margin=20.0, score=30.0,
                 cand=(("11", 30.0), ("22", 12.0)))
    commit_with_audio(tracker, "blue_ban_1", value, "blue_ban_2")
    action = actions(tracker)[0]
    assert action["champion_id"] == "11"
    assert action["via"] == "image"
    assert action["confidence"] == 0.7


def test_fusion_c_no_audio_keeps_image_champion():
    tracker = DraftTracker(stable_frames=3, turn_frames=2, audio_wait_s=5)
    tracker.observe_mentions([])  # audio on, but nothing said
    commit_with_audio(tracker, "blue_ban_1", read("locked", "11"), "blue_ban_2")
    action = actions(tracker)[0]
    assert action["champion_id"] == "11"
    assert action["via"] == "image"
    assert action["confidence"] == 0.7


def test_fusion_missing_slot_audio_commit():
    tracker = DraftTracker(stable_frames=3, turn_frames=2, audio_wait_s=5)
    tracker.observe_mentions([Mention(1.0, "22", "ban")])
    value = read("hover", None, score=4.0, margin=4.0, cand=(("22", 3.0),))
    t = feed(tracker, frame(blue_ban_1=value), frames=3)
    reads = frame(blue_ban_1=value, blue_ban_2=read("active"))
    t = feed(tracker, reads, frames=3, start_t=t)
    feed(tracker, reads, frames=8, start_t=t)
    action = actions(tracker)[0]
    assert action["champion_id"] == "22"
    assert action["via"] == "audio+image"
    assert action["confidence"] == 0.6


def test_fusion_missing_slot_records_suggestion_when_not_candidate():
    tracker = DraftTracker(stable_frames=3, turn_frames=2, audio_wait_s=5)
    tracker.observe_mentions([Mention(1.0, "22", "ban")])
    value = read("hover", None, score=4.0, margin=4.0, cand=(("99", 3.0),))
    t = feed(tracker, frame(blue_ban_1=value), frames=3)
    reads = frame(blue_ban_1=value, blue_ban_2=read("active"))
    t = feed(tracker, reads, frames=3, start_t=t)
    feed(tracker, reads, frames=8, start_t=t)
    assert actions(tracker) == []
    assert {"sequence": 1, "suggestion": "22"} in tracker.state()["suggestions"]


def test_fusion_upgrade_raises_confidence_of_committed_action():
    tracker = DraftTracker(stable_frames=3, turn_frames=2, audio_wait_s=5)
    tracker.observe_mentions([])
    commit_with_audio(tracker, "blue_ban_1", read("locked", "11"), "blue_ban_2")
    action = actions(tracker)[0]
    assert action["via"] == "image"
    assert action["confidence"] == 0.7
    tracker.observe_mentions([Mention(action["locked_at"] + 2.0, "11", "ban")])
    action = actions(tracker)[0]
    assert action["champion_id"] == "11"
    assert action["via"] == "audio+image"
    assert action["confidence"] >= 0.95


def test_fusion_upgrade_ignores_other_champion_and_late_audio():
    tracker = DraftTracker(stable_frames=3, turn_frames=2, audio_wait_s=5)
    tracker.observe_mentions([])
    commit_with_audio(tracker, "blue_ban_1", read("locked", "11"), "blue_ban_2")
    locked_at = actions(tracker)[0]["locked_at"]
    tracker.observe_mentions([Mention(locked_at + 1.0, "99", "ban")])
    tracker.observe_mentions([Mention(locked_at + 30.0, "11", "ban")])
    action = actions(tracker)[0]
    assert action["via"] == "image"
    assert action["confidence"] == 0.7


def test_fusion_a_audio_upgrades_but_never_changes_champion():
    tracker = DraftTracker(stable_frames=3, turn_frames=2, audio_wait_s=5)
    tracker.observe_mentions([])
    commit_with_audio(tracker, "blue_ban_1", read("locked", "11"), "blue_ban_2")
    tracker.observe_mentions([Mention(actions(tracker)[0]["locked_at"] + 3.0, "11", "ban")])
    assert actions(tracker)[0]["champion_id"] == "11"


def test_audio_support_cues_and_windows():
    tracker = DraftTracker()
    tracker.observe_mentions([
        Mention(5.0, "62", "neutral", 1.0),
        Mention(20.0, "62", "pick", 0.8),
    ])
    # neutral counts half for both action types
    assert tracker.audio_support("62", 0.0, 10.0, "ban") == 0.5
    assert tracker.audio_support("62", 0.0, 10.0, "pick") == 0.5
    # a pick cue counts for picks, not bans, and only inside the window
    assert tracker.audio_support("62", 0.0, 25.0, "pick") == 0.8
    assert tracker.audio_support("62", 0.0, 25.0, "ban") == 0.5
    assert tracker.audio_support("62", 10.0, 15.0, "ban") == 0.0


def test_speculative_mentions_are_ignored():
    tracker = DraftTracker()
    tracker.observe_mentions([Mention(0.0, "62", "speculative", 1.0)])
    assert tracker.audio_support("62", -10.0, 10.0, "ban") == 0.0
    assert tracker.audio_support("62", -10.0, 10.0, "pick") == 0.0
    assert tracker._mentions == []
