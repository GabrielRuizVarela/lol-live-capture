"""Recognition on a real Worlds 2026 champion-select frame (docs/contract.md).

Needs the broadcast still in ``tests/fixtures/private/`` (not in the repository: it is Riot's footage) and the
champion art from ``lol-live-capture setup``; skipped without them.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from conftest import ART_DIR
from lol_live_capture import assets, layout, recognize

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "private" / "s2520.jpg"

if not FIXTURE.is_file() or not assets.is_ready(ART_DIR):
    pytest.skip("needs tests/fixtures/private/s2520.jpg and `lol-live-capture setup`", allow_module_level=True)

BLUE_PICKS = {1: "Ashe", 2: "Seraphine", 3: "Galio", 4: "LeeSin"}
RED_PICKS = {1: "Rumble", 2: "Ezreal", 3: "Pyke", 4: "Qiyana"}
BLUE_BANS = {1: "Olaf", 2: "Nocturne", 3: "Cassiopeia", 4: "MonkeyKing", 5: "Yone"}
RED_BANS = {1: "Pantheon", 2: "Karma", 3: "Orianna", 4: "Ryze", 5: "Jax"}


@pytest.fixture(scope="module")
def bank() -> recognize.ReferenceBank:
    return recognize.ReferenceBank.load(ART_DIR)


@pytest.fixture(scope="module")
def reads(bank: recognize.ReferenceBank) -> dict[str, recognize.SlotRead]:
    frame = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
    assert frame is not None, f"fixture not readable: {FIXTURE}"
    assert frame.shape == (720, 1280, 3)
    return recognize.read_frame(frame, bank)


def test_layout_id() -> None:
    assert layout.LAYOUT_ID == "worlds-2026-1280x720-v1"


def test_draft_screen_detected(bank: recognize.ReferenceBank) -> None:
    frame = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
    ok, score = layout.is_draft_screen(frame)
    assert ok is True
    assert score > 0.5


def test_blue_picks(reads: dict[str, recognize.SlotRead]) -> None:
    for index, name in BLUE_PICKS.items():
        read = reads[f"blue_pick_{index}"]
        assert read.state == "locked", f"blue_pick_{index} state={read.state}"
        assert read.name == name, f"blue_pick_{index} name={read.name}"
        assert read.champion_id is not None
    fifth = reads["blue_pick_5"]
    assert fifth.state == "empty", f"blue_pick_5 state={fifth.state}"
    assert fifth.name is None


def test_red_picks(reads: dict[str, recognize.SlotRead]) -> None:
    for index, name in RED_PICKS.items():
        read = reads[f"red_pick_{index}"]
        assert read.state == "locked", f"red_pick_{index} state={read.state}"
        assert read.name == name, f"red_pick_{index} name={read.name}"
    fifth = reads["red_pick_5"]
    assert fifth.state == "empty", f"red_pick_5 state={fifth.state}"
    assert fifth.name is None


def test_bans(reads: dict[str, recognize.SlotRead]) -> None:
    for side, expected in (("blue", BLUE_BANS), ("red", RED_BANS)):
        for index, name in expected.items():
            read = reads[f"{side}_ban_{index}"]
            assert read.state == "locked", f"{side}_ban_{index} state={read.state}"
            assert read.name == name, f"{side}_ban_{index} name={read.name}"


def test_champion_ids_are_numeric(reads: dict[str, recognize.SlotRead]) -> None:
    for read in reads.values():
        if read.champion_id is not None:
            assert read.champion_id.isdigit()


def test_read_frame_is_fast(bank: recognize.ReferenceBank) -> None:
    frame = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
    recognize.read_frame(frame, bank)  # warm caches
    start = time.perf_counter()
    recognize.read_frame(frame, bank)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5, f"read_frame took {elapsed:.3f}s"


def test_black_frame_is_empty(bank: recognize.ReferenceBank) -> None:
    black = np.zeros((720, 1280, 3), dtype=np.uint8)
    ok, _ = layout.is_draft_screen(black)
    assert ok is False
    reads = recognize.read_frame(black, bank)
    assert len(reads) == 20
    assert all(read.state == "empty" for read in reads.values())
    assert all(read.champion_id is None for read in reads.values())


def test_dimmed_hover_is_not_locked(bank: recognize.ReferenceBank) -> None:
    """A dimmed hovered pick must never be reported as locked (tracker invariant)."""
    frame = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
    box = layout.pick_art_box("blue", 1)
    x0, y0, x1, y1 = box
    locked = frame[y0:y1, x0:x1]
    dimmed = cv2.convertScaleAbs(locked, alpha=0.6, beta=0)
    dimmed = cv2.GaussianBlur(dimmed, (5, 5), 0)
    frame[y0:y1, x0:x1] = cv2.resize(dimmed, (x1 - x0, y1 - y0))
    read = recognize.read_frame(frame, bank)["blue_pick_1"]
    assert read.state == "hover", f"dimmed hover read as {read.state}"
