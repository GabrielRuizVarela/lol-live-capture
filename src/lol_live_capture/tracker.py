"""Pure-Python draft tracker: noisy per-frame slot reads to draft actions.

See ``docs/contract.md`` (sections ``tracker.py`` and
``CaptureState``).  This module has no third-party dependencies: standard
library only.  ``SlotRead`` lives here (not in ``recognize.py``) so the
recognizer can import it without pulling cv2 into the tracker's tests.

Locking rule v3 ("the turn is the source of truth")
---------------------------------------------------
The recognizer reports at most one slot as ``active`` per draft frame: the slot
whose turn it is, taken from the overlay's own turn indicator.  A pick that is
pre-hovered before its turn is never ``active``, so it cannot move the turn.

For every slot we keep its **latest stable locked read**: the same
``champion_id`` seen in ``stable_frames`` (``K``, default 3) *consecutive draft
frames* with ``state == "locked"``.  A frame that is not a draft frame, or a
slot reported ``unknown`` (a stat-card expansion is on screen), neither counts
nor breaks a streak; a different champion or an ``empty``/``active`` read does
break it.

The **current turn** is the sequence of the ``active`` slot once it has been
seen for ``turn_frames`` (default 2) consecutive draft frames.  When the current
turn moves from sequence ``a`` to a later sequence ``b``, every slot with
sequence ``< b`` is final:

- a slot with a stable locked read is committed immediately;
- a slot with no stable locked read yet stays pending and is committed as soon
  as ``K`` consecutive ``locked`` reads of one champion arrive (its turn is
  over, so its art can only be the final champion).

Joining mid-draft: the first time the turn is seen at sequence ``b``, every slot
with sequence ``< b`` that has a stable locked read is committed.

Draft over / end-of-draft reconciliation
---------------------------------------
The draft is **over** as soon as either:

- the turn reached sequence 20 and no slot has been ``active`` for
  ``no_active_frames`` (default 4) consecutive draft frames; or
- the at-rest pick layout shows a side's portraits in a different order than the
  committed picks (the end-of-draft **role reorder**);

whichever comes first.  Once the draft is over, positional pick commits stop for
good: after the reorder a pick slot no longer holds the champion of its own
sequence.

Reconciliation then runs on every following draft frame:

- slot 20 is committed with its latest stable read captured **before** the
  reorder (while it was the turn);
- a missing **ban** is committed with its latest stable locked read (bans never
  reorder, so this is still positional);
- each side's **final pick set** is read from ``stable_frames`` consecutive
  at-rest frames where all five of that side's portraits read the same five
  distinct champions (``locked`` or an at-rest ``hover``).  Because it is a set,
  the role reorder does not change it.
  - exactly one pick of the side missing: it is the one champion of the final set
    not yet committed, committed only when the turn evidence in that slot does
    not contradict it;
  - two or more picks missing: the leftover champions are assigned from the
    champion read most often in each missing slot during its own turn (including
    hover reads), and committed only when that assignment is unique and
    consistent;
  - a committed pick that is **not** in the side's final set is left as is
    (committed is final), a warning is added, and the side's final-set commits
    are skipped so the leftover cannot be mis-assigned.

Every commit records its origin in the ``via`` field of the action:
``"turn"`` (positional commit on transition / slot 20), ``"final_set"`` or
``"final_ban"``.

Confidence
----------
The action/hover ``confidence`` is ``min(1.0, margin)`` where ``margin`` is the
recognizer's best-minus-second-best score for the slot.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone

SLOT_KEYS = (
    "blue_ban_1",
    "blue_ban_2",
    "blue_ban_3",
    "blue_ban_4",
    "blue_ban_5",
    "red_ban_1",
    "red_ban_2",
    "red_ban_3",
    "red_ban_4",
    "red_ban_5",
    "blue_pick_1",
    "blue_pick_2",
    "blue_pick_3",
    "blue_pick_4",
    "blue_pick_5",
    "red_pick_1",
    "red_pick_2",
    "red_pick_3",
    "red_pick_4",
    "red_pick_5",
)

# Standard 20-action pro draft order (sequence 1..20).
SEQUENCE: dict[str, int] = {
    "blue_ban_1": 1,
    "red_ban_1": 2,
    "blue_ban_2": 3,
    "red_ban_2": 4,
    "blue_ban_3": 5,
    "red_ban_3": 6,
    "blue_pick_1": 7,
    "red_pick_1": 8,
    "red_pick_2": 9,
    "blue_pick_2": 10,
    "blue_pick_3": 11,
    "red_pick_3": 12,
    "red_ban_4": 13,
    "blue_ban_4": 14,
    "red_ban_5": 15,
    "blue_ban_5": 16,
    "red_pick_4": 17,
    "blue_pick_4": 18,
    "blue_pick_5": 19,
    "red_pick_5": 20,
}

SEQUENCE_TO_SLOT = {sequence: slot for slot, sequence in SEQUENCE.items()}

TOTAL_ACTIONS = 20

# A slot reported with one of these states is a normal, readable slot.
READABLE_STATES = ("empty", "hover", "locked", "active")


@dataclass(frozen=True)
class SlotRead:
    """One recognizer read of one slot in one frame.

    ``state`` is ``"empty"``, ``"unknown"`` (the layout is not at rest, the read
    is meaningless), ``"active"`` (the slot whose turn it is), ``"hover"`` or
    ``"locked"``.  ``champion_id`` (Riot numeric key as a string) is ``None`` for
    ``empty``/``unknown`` and for a hover/active whose champion could not be
    matched.

    ``candidates`` holds the top three ``(champion_id, score)`` the recognizer
    matched for the slot, best first.  It is used by the audio fusion to find a
    look-alike the casters actually named; the tracker itself only needs
    ``champion_id``/``score``/``margin``.
    """

    state: str
    champion_id: str | None
    name: str | None = None
    score: float = 0.0
    margin: float = 0.0
    brightness: float = 0.0
    candidates: tuple[tuple[str, float], ...] = ()


@dataclass
class _Run:
    """A current run of the same champion, read ``locked``, in one slot."""

    champion_id: str
    name: str | None
    count: int
    state: str
    margin: float
    first_t: float
    score: float = 0.0


@dataclass
class _Stable:
    """The latest champion in one slot that reached ``stable_frames`` locked."""

    champion_id: str
    name: str | None
    state: str
    margin: float
    first_t: float
    score: float = 0.0


@dataclass
class _Pending:
    """A slot whose commit is deferred to let the audio evidence arrive.

    ``champion_id`` is the stable image read captured at the rule-v3 commit
    point (``None`` if the image never produced one).  The champion is frozen
    here: a later stable read (e.g. the end-of-draft role reorder) must not
    change a commit that has already been decided.
    """

    champion_id: str | None
    score: float
    margin: float
    deadline: float
    turn_start: float



def _side(slot: str) -> str:
    return "blue" if slot.startswith("blue_") else "red"


def _action_type(slot: str) -> str:
    return "ban" if "_ban_" in slot else "pick"


def _is_locked(read: SlotRead, min_margin: float) -> bool:
    return (
        read.state == "locked"
        and read.champion_id is not None
        and read.margin >= min_margin
    )


class DraftTracker:
    """Turns noisy per-frame slot reads into draft actions and hover."""

    def __init__(
        self,
        *,
        stable_frames: int = 3,
        final_stable_frames: int = 6,
        turn_frames: int = 2,
        no_active_frames: int = 4,
        min_margin: float = 0.0,
        idle_reset_s: float = 60.0,
        audio_wait_s: float = 15.0,
        audio_window_s: float = 20.0,
        lookalike_max_margin: float = 3.0,
        source: str | None = None,
        layout: str | None = None,
    ) -> None:
        self.stable_frames = stable_frames
        self.final_stable_frames = final_stable_frames
        self.turn_frames = turn_frames
        self.no_active_frames = no_active_frames
        self.min_margin = min_margin
        self.idle_reset_s = idle_reset_s
        self.audio_wait_s = audio_wait_s
        self.audio_window_s = audio_window_s
        self.lookalike_max_margin = lookalike_max_margin
        self._source = source
        self._layout = layout
        self.reset()

    # -- public API -----------------------------------------------------

    def observe(
        self,
        t: float,
        is_draft: bool,
        reads: Mapping[str, SlotRead],
        *,
        tricodes: tuple[str | None, str | None] | None = None,
    ) -> bool:
        """Feed one frame (``t`` = stream seconds).  True if the state changed."""
        before = self._fingerprint()
        self._stream_t = t
        if tricodes is not None:
            blue, red = tricodes
            if blue is not None:
                self._teams["blue"]["tricode"] = blue
            if red is not None:
                self._teams["red"]["tricode"] = red

        if is_draft:
            self._observe_draft(t, reads)
        else:
            # The overlay is gone: no hover, but running streaks survive.
            self._hover = None

        self._refresh_status()
        return before != self._fingerprint()

    def observe_mentions(self, mentions: Iterable[object]) -> None:
        """Feed caster champion mentions (from :mod:`spotter`) to the tracker.

        Each mention is duck-typed: ``t``, ``champion_id``, ``cue``, ``text`` and
        ``score``.  ``speculative`` mentions are ignored, as is any cue that is
        not ``ban``/``pick``/``neutral``.
        """
        self._audio_enabled = True
        for mention in mentions:
            cue = getattr(mention, "cue", None)
            champion_id = getattr(mention, "champion_id", None)
            t = getattr(mention, "t", None)
            if champion_id is None or t is None or cue not in ("ban", "pick", "neutral"):
                continue
            score = float(getattr(mention, "score", 0.0) or 0.0)
            self._mentions.append((float(t), str(champion_id), cue, score))
        self._upgrade_committed()

    def audio_support(self, champion_id: str, t0: float, t1: float, action_type: str) -> float:
        """Best mention strength for ``champion_id`` in ``[t0, t1]`` (0..1).

        A ``ban`` cue counts for bans, a ``pick`` cue for picks; a ``neutral``
        cue counts half for both.
        """
        value, _ = self._best_mention(champion_id, t0, t1, action_type)
        return value

    def _best_mention(
        self, champion_id: str, t0: float, t1: float, action_type: str
    ) -> tuple[float, str | None]:
        best = 0.0
        best_cue: str | None = None
        for mention_t, mention_id, cue, score in self._mentions:
            if mention_id != champion_id or mention_t < t0 or mention_t > t1:
                continue
            if cue == "neutral":
                value = 0.5 * score
            elif cue == action_type:
                value = score
            else:
                continue
            if value > best:
                best, best_cue = value, cue
        return best, best_cue

    def _upgrade_committed(self) -> None:
        """Raise a committed action's confidence when its audio arrives late."""
        for action in self._actions.values():
            if action["via"] == "audio+image":
                continue
            support, _ = self._best_mention(
                action["champion_id"],
                action["locked_at"],
                action["locked_at"] + self.audio_wait_s,
                action["action_type"],
            )
            if support > 0:
                action["via"] = "audio+image"
                action["confidence"] = max(action["confidence"], 0.95)

    def set_status(self, status: str, error: str | None = None) -> None:
        self._status = status
        self._error = error

    def reset(self) -> None:
        self._actions: dict[str, dict] = {}
        self._warnings: list[str] = []
        self._locked_champions: dict[str, str] = {}
        self._champion_slots: dict[str, str] = {}
        self._run: dict[str, _Run] = {}
        self._stable: dict[str, _Stable] = {}
        self._current_turn: int | None = None
        self._turn_candidate: int | None = None
        self._turn_run = 0
        self._no_active_run = 0
        self._final_stable: _Stable | None = None
        self._reorder_seen = False
        self._reorder_run: dict[str, tuple[str, int]] = {}
        self._draft_over = False
        # slot -> Counter of champions read while that slot was the turn.
        self._turn_evidence: dict[str, Counter] = {}
        # side -> (tuple of the five pick champions, run length, min margin).
        self._final_run: dict[str, tuple[tuple[str, ...], int, float]] = {}
        # side -> the first final pick set seen stable for K at-rest frames.
        self._final_pick_set: dict[str, tuple[str, ...]] = {}
        self._hover: dict | None = None
        self._all_empty_run = 0
        self._stream_t: float | None = None
        self._last_draft_t: float | None = None
        self._ever_draft = False
        self._game_index = 0
        self._status = "idle"
        self._error: str | None = None
        self._teams = {"blue": {"tricode": None}, "red": {"tricode": None}}
        # Audio fusion state.
        self._mentions: list[tuple[float, str, str, float]] = []
        self._audio_enabled = False
        self._start_new_draft_audio()

    def _start_new_draft_audio(self) -> None:
        self._pending: dict[str, _Pending] = {}
        self._missing_scheduled: set[str] = set()
        self._turn_start: dict[int, float] = {}
        self._suggestions: list[dict] = []
        self._slot_cand_frames: dict[str, Counter] = {}
        self._slot_cand_score: dict[str, dict[str, float]] = {}

    def state(self, *, now_iso: str | None = None) -> dict:
        return {
            "status": self._status,
            "source": self._source,
            "layout": self._layout,
            "updated_at": now_iso
            if now_iso is not None
            else datetime.now(timezone.utc).isoformat(),
            "stream_t": self._stream_t,
            "frame_age_ms": None,
            "game_index": self._game_index,
            "teams": {
                "blue": {"tricode": self._teams["blue"]["tricode"]},
                "red": {"tricode": self._teams["red"]["tricode"]},
            },
            "actions": [dict(action) for action in self._sorted_actions()],
            "hover": None if self._hover is None else dict(self._hover),
            "suggestions": [dict(row) for row in self._suggestions],
            "warnings": list(self._warnings),
            "error": self._error,
        }

    # -- internals ------------------------------------------------------

    def _observe_draft(self, t: float, reads: Mapping[str, SlotRead]) -> None:
        if self._game_index == 0:
            self._game_index = 1  # the first draft is game 1

        all_empty = all(
            slot in reads and reads[slot].state == "empty" for slot in SLOT_KEYS
        )
        self._all_empty_run = self._all_empty_run + 1 if all_empty else 0

        new_draft = False
        if (
            self._status == "draft_complete"
            and self._all_empty_run >= self.stable_frames
        ):
            # (a) the overlay reset to a fresh, fully empty board after a draft.
            new_draft = True
        elif (
            self._game_index >= 1
            and self._last_draft_t is not None
            and t - self._last_draft_t >= self.idle_reset_s
            and self._locked_read_count(reads) < len(self._actions)
        ):
            # (b) the draft screen was gone for a long time and came back with
            # fewer locked slots than we had actions -> a fresh draft.
            new_draft = True

        if new_draft:
            self._start_new_draft()

        self._last_draft_t = t
        self._ever_draft = True
        self._update_runs(t, reads)
        self._update_turn(t, reads)
        if self._draft_over:
            self._flush_pending(t, all_of_them=True)
            self._update_final_sets(reads)
            self._reconcile(t)
        else:
            self._record_evidence(reads)
            self._commit_ready(t)
        self._hover = self._compute_hover(reads)

    def _start_new_draft(self) -> None:
        self._actions.clear()
        self._warnings.clear()
        self._locked_champions.clear()
        self._champion_slots.clear()
        self._run.clear()
        self._stable.clear()
        self._current_turn = None
        self._turn_candidate = None
        self._turn_run = 0
        self._no_active_run = 0
        self._final_stable = None
        self._reorder_seen = False
        self._reorder_run.clear()
        self._draft_over = False
        self._turn_evidence.clear()
        self._final_run.clear()
        self._final_pick_set.clear()
        self._hover = None
        self._all_empty_run = 0
        self._game_index += 1
        self._start_new_draft_audio()

    def _update_runs(self, t: float, reads: Mapping[str, SlotRead]) -> None:
        for slot in SLOT_KEYS:
            read = reads.get(slot)
            if read is None or read.state == "unknown":
                # The layout is not at rest (or the frame carries no read):
                # neither counts nor breaks this slot's streak.
                continue
            if slot not in self._actions and read.candidates:
                frames = self._slot_cand_frames.setdefault(slot, Counter())
                scores = self._slot_cand_score.setdefault(slot, {})
                for champion_id, score in read.candidates:
                    frames[champion_id] += 1
                    if score > scores.get(champion_id, 0.0):
                        scores[champion_id] = float(score)
            if not _is_locked(read, self.min_margin):
                self._run.pop(slot, None)
                continue
            previous = self._run.get(slot)
            if previous is not None and previous.champion_id == read.champion_id:
                count = previous.count + 1
                first_t = previous.first_t
            else:
                count = 1
                first_t = t
            run = _Run(read.champion_id, read.name, count, read.state, read.margin, first_t, read.score)
            self._run[slot] = run
            if count >= self.stable_frames:
                self._stable[slot] = _Stable(
                    read.champion_id, read.name, read.state, read.margin, first_t, read.score
                )

    def _active_sequence(self, reads: Mapping[str, SlotRead]) -> int | None:
        active = [
            slot for slot in SLOT_KEYS if reads.get(slot) is not None
            and reads[slot].state == "active"
        ]
        if not active:
            return None
        return SEQUENCE[min(active, key=lambda key: SEQUENCE[key])]

    def _update_turn(self, t: float, reads: Mapping[str, SlotRead]) -> None:
        self._detect_reorder(reads)
        active = self._active_sequence(reads)
        if active is None:
            self._turn_candidate = None
            self._turn_run = 0
            self._no_active_run += 1
        else:
            self._no_active_run = 0
            if active == self._turn_candidate:
                self._turn_run += 1
            else:
                self._turn_candidate = active
                self._turn_run = 1
            if self._turn_run >= self.turn_frames:
                if self._current_turn is None or active > self._current_turn:
                    self._current_turn = active
                    self._turn_start[active] = t

        if self._current_turn == TOTAL_ACTIONS and not self._reorder_seen:
            slot = SEQUENCE_TO_SLOT[TOTAL_ACTIONS]
            stable = self._stable.get(slot)
            if stable is not None:
                self._final_stable = stable

        if not self._draft_over:
            if self._reorder_seen:
                self._draft_over = True
            elif (
                self._current_turn == TOTAL_ACTIONS
                and self._no_active_run >= self.no_active_frames
            ):
                self._draft_over = True

    def _record_evidence(self, reads: Mapping[str, SlotRead]) -> None:
        """Per pick slot, count every champion read before the draft is over.

        Before the role reorder a pick slot still holds the champion of its own
        sequence, so these are positional reads (locked or hover, including the
        turn slot).  Reads after the reorder would be meaningless and are not
        recorded.
        """
        for slot in SLOT_KEYS:
            if _action_type(slot) != "pick":
                continue
            read = reads.get(slot)
            if read is None or read.champion_id is None:
                continue
            self._turn_evidence.setdefault(slot, Counter())[read.champion_id] += 1

    def _detect_reorder(self, reads: Mapping[str, SlotRead]) -> None:
        """The end-of-draft role reorder changes committed pick portraits.

        A single noisy read is not a reorder: the changed champion must be seen in
        the same committed slot for ``stable_frames`` consecutive draft frames.
        """
        if self._reorder_seen:
            return
        changed: list[str] = []
        for slot in SLOT_KEYS:
            action = self._actions.get(slot)
            if action is None or _action_type(slot) != "pick":
                continue
            read = reads.get(slot)
            if (
                read is None
                or read.champion_id is None
                or read.champion_id == action["champion_id"]
            ):
                self._reorder_run.pop(slot, None)
                continue
            previous = self._reorder_run.get(slot)
            count = previous[1] + 1 if previous is not None and previous[0] == read.champion_id else 1
            self._reorder_run[slot] = (read.champion_id, count)
            if count >= self.stable_frames:
                changed.append(slot)
        # A real role reorder rearranges several portraits at once; a single
        # committed slot reading a different champion is usually just a noisy
        # read (or the correction of a wrong commit), not a reorder.
        if len(changed) >= 2:
            self._reorder_seen = True
            return
        for slot in changed:
            champion = reads[slot].champion_id
            for other, action in self._actions.items():
                if other != slot and action["champion_id"] == champion:
                    self._reorder_seen = True
                    return

    def _commit_ready(self, t: float) -> None:
        if self._current_turn is None:
            return
        # 1. Slots already waiting for the audio: commit those whose wait is over
        #    or whose next turn's commit point has arrived (never commit late).
        self._flush_pending(t)
        # 2. Slots whose turn has just passed become candidates for commit.
        for slot in SLOT_KEYS:
            if slot in self._actions or slot in self._pending:
                continue
            if SEQUENCE[slot] >= self._current_turn:
                continue
            self._start_pending(slot, t)

    def _start_pending(self, slot: str, t: float) -> None:
        stable = self._stable.get(slot)
        if stable is not None:
            if not self._audio_enabled:
                # Image-only: keep the rule-v3 timing exactly (commit now).
                self._lock_image(slot, stable.champion_id, stable.margin, stable.score, t)
                return
            sequence = SEQUENCE[slot]
            self._pending[slot] = _Pending(
                champion_id=stable.champion_id,
                score=stable.score,
                margin=stable.margin,
                deadline=t + self.audio_wait_s,
                turn_start=self._turn_start.get(sequence, t),
            )
            return
        # No stable image read: schedule the audio-only rescue once, but keep the
        # slot eligible for the rule-v3 late-stable commit (handled below when a
        # stable read finally arrives).
        if not self._audio_enabled or slot in self._missing_scheduled:
            return
        self._missing_scheduled.add(slot)
        sequence = SEQUENCE[slot]
        self._pending[slot] = _Pending(
            champion_id=None,
            score=0.0,
            margin=0.0,
            deadline=t + self.audio_wait_s,
            turn_start=self._turn_start.get(sequence, t),
        )

    def _flush_pending(self, t: float, *, all_of_them: bool = False) -> None:
        for slot in list(self._pending):
            pending = self._pending[slot]
            if all_of_them or t >= pending.deadline:
                del self._pending[slot]
                self._finalize_pending(slot, pending, t)

    def _finalize_pending(self, slot: str, pending: _Pending, t: float) -> None:
        if slot in self._actions:
            return
        action_type = _action_type(slot)
        w0 = pending.turn_start - self.audio_window_s
        w1 = t
        champion = pending.champion_id
        score = pending.score
        margin = pending.margin
        if not self._reorder_seen and not self._draft_over:
            # The image never had a stable read at the commit point, but a late
            # locked read may have arrived while we waited for the audio.  Re-read
            # only before the role reorder: after it the art is another player's.
            stable = self._stable.get(slot)
            if stable is not None:
                champion, score, margin = stable.champion_id, stable.score, stable.margin
        if champion is not None:
            support, _ = self._best_mention(champion, w0, w1, action_type)
            if support > 0:
                self._lock(slot, champion, margin, t, via="audio+image", confidence=0.9)
                return
            lookalike = self._lookalike(slot, champion, score, margin, action_type, w0, w1)
            if lookalike is not None:
                self._lock(slot, lookalike, 0.8, t, via="audio+image", confidence=0.8)
                return
            self._lock_image(slot, champion, margin, score, t)
            return
        self._finalize_missing(slot, action_type, w0, w1, t)

    def _lock_image(self, slot: str, champion: str, margin: float, score: float, t: float) -> None:
        self._lock(slot, champion, margin, t, via="image", confidence=0.7)

    def _lookalike(
        self, slot: str, champion: str, champion_score: float, champion_margin: float,
        action_type: str, w0: float, w1: float,
    ) -> str | None:
        """A top-3 candidate the casters named, that the image also scored decently.

        Only fires when the image itself is unsure (a look-alike): a clear
        open-set margin on ``champion`` is trusted over a fuzzy name.
        """
        if champion_margin > self.lookalike_max_margin:
            return None
        scores = self._slot_cand_score.get(slot, {})
        best: tuple[str, float] | None = None
        for candidate, score in sorted(scores.items(), key=lambda item: -item[1]):
            if candidate == champion or candidate in self._champion_slots:
                continue
            if score < 0.5 * champion_score:
                continue
            support, cue = self._best_mention(candidate, w0, w1, action_type)
            if support > 0 and cue == action_type:
                if best is None or support > best[1]:
                    best = (candidate, support)
        return None if best is None else best[0]

    def _finalize_missing(
        self, slot: str, action_type: str, w0: float, w1: float, t: float,
    ) -> None:
        strong: dict[str, float] = {}
        for mention_t, champion_id, cue, score in self._mentions:
            if mention_t < w0 or mention_t > w1:
                continue
            if cue != action_type or champion_id in self._champion_slots:
                continue
            if score > strong.get(champion_id, 0.0):
                strong[champion_id] = score
        if len(strong) != 1:
            return
        champion = next(iter(strong))
        frames = self._slot_cand_frames.get(slot, Counter()).get(champion, 0)
        if frames >= 2:
            self._lock(slot, champion, 0.6, t, via="audio+image", confidence=0.6)
        elif not self._has_suggestion(SEQUENCE[slot], champion):
            self._suggestions.append({"sequence": SEQUENCE[slot], "suggestion": champion})

    def _has_suggestion(self, sequence: int, champion: str) -> bool:
        return any(
            row["sequence"] == sequence and row["suggestion"] == champion
            for row in self._suggestions
        )

    def _update_final_sets(self, reads: Mapping[str, SlotRead]) -> None:
        """Track each side's settled pick set (order-independent), K frames."""
        for side in ("blue", "red"):
            champions: list[str] = []
            margin = 1.0
            settled = True
            for index in range(1, 6):
                read = reads.get(f"{side}_pick_{index}")
                if (
                    read is None
                    or read.state not in ("locked", "hover")
                    or read.champion_id is None
                ):
                    settled = False
                    break
                champions.append(read.champion_id)
                margin = min(margin, read.margin)
            key = tuple(champions)
            if settled and len(set(key)) == 5:
                previous = self._final_run.get(side)
                count = previous[1] + 1 if previous is not None and previous[0] == key else 1
                self._final_run[side] = (key, count, margin)
                # A longer settle window than K: right after a lock the portraits
                # are still animating (and the reorder may not have reached this
                # side yet), so a short run can latch a stale layout.
                if count >= self.final_stable_frames and side not in self._final_pick_set:
                    self._final_pick_set[side] = key
            else:
                self._final_run.pop(side, None)

    def _reconcile(self, t: float) -> None:
        """After the draft is over: no positional picks, resolve what we can."""
        slot20 = SEQUENCE_TO_SLOT[TOTAL_ACTIONS]
        if slot20 not in self._actions and self._final_stable is not None:
            # Only the pre-reorder read is safe: after the role reorder the art
            # in slot 20 belongs to another player's champion.
            self._lock(
                slot20,
                self._final_stable.champion_id,
                self._final_stable.margin,
                t,
                via="turn",
            )
        for slot in SLOT_KEYS:
            if _action_type(slot) != "ban" or slot in self._actions:
                continue
            stable = self._stable.get(slot)
            if stable is not None:
                self._lock(slot, stable.champion_id, stable.margin, t, via="final_ban")
        self._apply_final_sets(t)

    def _apply_final_sets(self, t: float) -> None:
        for side in ("blue", "red"):
            final = self._final_pick_set.get(side)
            if final is None:
                continue
            final_set = set(final)
            committed = {
                slot: action
                for slot, action in self._actions.items()
                if _side(slot) == side and _action_type(slot) == "pick"
            }
            committed_ids = {action["champion_id"] for action in committed.values()}
            outside = committed_ids - final_set
            if outside:
                for champion in sorted(outside):
                    self._warn(
                        f"committed {side} pick {champion} is not in the final"
                        f" {side} pick set"
                    )
                # committed is final: the leftover cannot be trusted, skip.
                continue
            missing = [
                slot
                for slot in SLOT_KEYS
                if _side(slot) == side
                and _action_type(slot) == "pick"
                and slot not in self._actions
            ]
            leftovers = final_set - committed_ids
            if not missing:
                continue
            if len(missing) == 1 and len(leftovers) == 1:
                self._commit_single_final(missing[0], next(iter(leftovers)), t)
            elif len(leftovers) == len(missing) and len(missing) >= 2:
                assignment = self._unique_assignment(missing, leftovers)
                if assignment is not None:
                    for slot, champion in assignment.items():
                        self._lock(slot, champion, self._margin_for(slot), t, via="final_set")

    def _commit_single_final(self, slot: str, champion: str, t: float) -> None:
        evidence = self._turn_evidence.get(slot)
        if evidence and evidence.get(champion, 0) == 0:
            # The turn evidence points at another champion: the leftover belongs
            # to a sequence already (wrongly) committed; leave this slot missing.
            return
        self._lock(slot, champion, self._margin_for(slot), t, via="final_set")

    def _unique_assignment(self, missing: list[str], leftovers: set[str]) -> dict[str, str] | None:
        assigned: dict[str, str] = {}
        remaining = set(leftovers)
        while True:
            progressed = False
            for slot in missing:
                if slot in assigned:
                    continue
                evidence = self._turn_evidence.get(slot)
                if not evidence:
                    continue
                votes = [(evidence.get(champion, 0), champion) for champion in remaining]
                positive = [(count, champion) for count, champion in votes if count > 0]
                if not positive:
                    continue
                best = max(count for count, _ in positive)
                top = [champion for count, champion in positive if count == best]
                if len(top) == 1:
                    assigned[slot] = top[0]
                    remaining.discard(top[0])
                    progressed = True
                    break
            if progressed:
                continue
            unassigned = [slot for slot in missing if slot not in assigned]
            if len(unassigned) == 1 and len(remaining) == 1:
                assigned[unassigned[0]] = next(iter(remaining))
                remaining.clear()
                progressed = True
            if not progressed:
                break
        if len(assigned) == len(missing):
            return assigned
        return None

    def _margin_for(self, slot: str) -> float:
        stable = self._stable.get(slot)
        if stable is not None:
            return stable.margin
        run = self._run.get(slot)
        return run.margin if run is not None else 0.9

    def _warn(self, message: str) -> None:
        if message not in self._warnings:
            self._warnings.append(message)

    def _lock(
        self, slot: str, champion: str, margin: float, t: float, *,
        via: str = "turn", confidence: float | None = None,
    ) -> None:
        self._locked_champions[slot] = champion
        first_slot = self._champion_slots.get(champion)
        if first_slot is not None:
            self._warn(
                f"champion {champion} locked twice: kept {first_slot},"
                f" ignored {slot}"
            )
            return
        self._champion_slots[champion] = slot
        self._actions[slot] = {
            "sequence": SEQUENCE[slot],
            "side": _side(slot),
            "action_type": _action_type(slot),
            "champion_id": champion,
            "confidence": min(1.0, margin) if confidence is None else confidence,
            "locked_at": t,
            "via": via,
        }

    def _compute_hover(self, reads: Mapping[str, SlotRead]) -> dict | None:
        best: str | None = None
        for slot in SLOT_KEYS:
            if slot in self._locked_champions:
                continue
            read = reads.get(slot)
            if read is None or read.state != "hover":
                continue
            if best is None or SEQUENCE[slot] < SEQUENCE[best]:
                best = slot
        if best is not None:
            read = reads[best]
            return {
                "sequence": SEQUENCE[best],
                "side": _side(best),
                "action_type": _action_type(best),
                "champion_id": read.champion_id,
                "confidence": min(1.0, read.margin),
            }
        # The turn slot holding a hover is the most meaningful hover.
        for slot in SLOT_KEYS:
            read = reads.get(slot)
            if read is not None and read.state == "active" and read.champion_id is not None:
                return {
                    "sequence": SEQUENCE[slot],
                    "side": _side(slot),
                    "action_type": _action_type(slot),
                    "champion_id": read.champion_id,
                    "confidence": min(1.0, read.margin),
                }
        return None

    @staticmethod
    def _locked_read_count(reads: Mapping[str, SlotRead]) -> int:
        return sum(
            1 for slot in SLOT_KEYS if slot in reads and reads[slot].state == "locked"
        )

    def _refresh_status(self) -> None:
        if self._status == "error":
            return
        if len(self._actions) >= TOTAL_ACTIONS:
            self._status = "draft_complete"
        elif self._ever_draft:
            self._status = "drafting"
        else:
            self._status = "waiting_for_draft"

    def _sorted_actions(self) -> list[dict]:
        return sorted(self._actions.values(), key=lambda action: action["sequence"])

    def _fingerprint(self) -> tuple:
        return (
            self._status,
            self._stream_t,
            self._game_index,
            self._teams["blue"]["tricode"],
            self._teams["red"]["tricode"],
            tuple(
                (
                    action["sequence"],
                    action["champion_id"],
                    action["confidence"],
                    action["locked_at"],
                )
                for action in self._sorted_actions()
            ),
            None
            if self._hover is None
            else (
                self._hover["sequence"],
                self._hover["side"],
                self._hover["action_type"],
                self._hover["champion_id"],
                self._hover["confidence"],
            ),
            tuple(self._warnings),
            tuple((row["sequence"], row["suggestion"]) for row in self._suggestions),
            self._error,
        )
