"""Where the Worlds 2026 champion-select overlay puts things on a 1280x720 frame.

Boxes are pixels (x0, y0, x1, y1) calibrated from Play-In broadcast frames.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

LAYOUT_ID = "worlds-2026-1280x720-v1"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

BAN_ROW_Y = (476, 516)
BAN_SLOT_WIDTH = 48
BAN_ART_INSET = 4  # icon art inside a ban slot: x+4 .. x+44
BAN_ART_WIDTH = 40

PICK_ROW_Y = (527, 720)
PICK_ART_Y = (527, 690)  # player name + role icon live below this
PICK_SLOT_WIDTH = 106
PICK_ART_INSET = 4
PICK_ART_WIDTH = 98

BLUE_PICK_X = (0, 106, 212, 319, 426)
RED_PICK_X = (747, 853, 959, 1066, 1172)

CENTER_PANEL_BOX = (540, 540, 740, 715)
BLUE_TRICODE_BOX = (550, 660, 625, 705)
RED_TRICODE_BOX = (655, 660, 735, 705)
PHASE_BOX = (540, 555, 740, 600)

ACTING_LINE_Y = (518, 526)  # long team-colour bar under the ban row
ACTING_LINE_X = (540, 740)  # sample the centre of the bar (unaffected by side art)

# At rest the four inner portrait separators of a side sit at these x positions
# and show a thin, dark vertical line over the pick band.
PICK_BOUNDARIES = {"blue": (106, 212, 319, 426), "red": (853, 959, 1066, 1172)}
PICK_SEPARATOR_Y = (530, 716)

_SEPARATOR_MAX = 18.0  # a separator column is very dark (mean value below this)
_SEPARATOR_RATIO = 0.5  # ... and at least twice as dark as the columns around it
_LINE_MARGIN = 25.0  # |B - R| on the acting bar needed to name a side

_TEMPLATE_PATH = Path(__file__).resolve().parent / "assets" / "draft_center.png"
_PATCH_WINDOW = (698, 715, 548, 735)  # "PATCH 26.20" strip of the centre panel, frame pixels
_DRAFT_MIN_SCORE = 0.5

_template_cache: dict[Path, np.ndarray] = {}


def slot_boxes() -> dict[str, tuple[int, int, int, int]]:
    """Pixel box of every slot on screen, keyed like ``blue_ban_1`` .. ``red_pick_5``."""
    boxes: dict[str, tuple[int, int, int, int]] = {}
    y0, y1 = BAN_ROW_Y
    for k in range(5):
        boxes[f"blue_ban_{k + 1}"] = (k * BAN_SLOT_WIDTH, y0, (k + 1) * BAN_SLOT_WIDTH, y1)
    for k in range(5):
        boxes[f"red_ban_{k + 1}"] = (FRAME_WIDTH - (k + 1) * BAN_SLOT_WIDTH, y0, FRAME_WIDTH - k * BAN_SLOT_WIDTH, y1)
    py0, py1 = PICK_ROW_Y
    for k, x in enumerate(BLUE_PICK_X):
        boxes[f"blue_pick_{k + 1}"] = (x, py0, x + PICK_SLOT_WIDTH, py1)
    for k, x in enumerate(RED_PICK_X):
        boxes[f"red_pick_{k + 1}"] = (x, py0, x + PICK_SLOT_WIDTH, py1)
    return boxes


def ban_art_box(side: str, index: int) -> tuple[int, int, int, int]:
    x0, _, _, _ = slot_boxes()[f"{side}_ban_{index}"]
    y0, y1 = BAN_ROW_Y
    return (x0 + BAN_ART_INSET, y0, x0 + BAN_ART_INSET + BAN_ART_WIDTH, y1)


def pick_art_box(side: str, index: int) -> tuple[int, int, int, int]:
    x0, _, x1, _ = slot_boxes()[f"{side}_pick_{index}"]
    y0, y1 = PICK_ART_Y
    return (x0 + PICK_ART_INSET, y0, x0 + PICK_ART_INSET + PICK_ART_WIDTH, y1)


def acting_side(frame: np.ndarray) -> str | None:
    """Which side is acting right now, from the long team-colour bar (or ``None``)."""
    y0, y1 = ACTING_LINE_Y
    x0, x1 = ACTING_LINE_X
    bar = frame[y0:y1, x0:x1].astype(np.float32)
    blue = float(bar[:, :, 0].mean())
    red = float(bar[:, :, 2].mean())
    if blue - red > _LINE_MARGIN:
        return "blue"
    if red - blue > _LINE_MARGIN:
        return "red"
    return None


def pick_at_rest(frame: np.ndarray, side: str) -> bool:
    """True when a side's pick portraits are all at their fixed, unexpanded width.

    Right after a lock the broadcast widens one portrait and squeezes the rest,
    so the fixed slot boxes cut through the wrong art.  At rest every expected
    inner boundary shows a thin dark separator; if one is missing the side is
    not at rest and its pick slots must not be read.
    """
    y0, y1 = PICK_SEPARATOR_Y
    band = frame[y0:y1, :]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.float32)
    profile = gray.mean(axis=0)
    width = profile.shape[0]
    for x in PICK_BOUNDARIES[side]:
        lo, hi = max(0, x - 4), min(width, x + 5)
        column = float(profile[lo:hi].min())
        around = np.concatenate(
            [profile[max(0, x - 14) : max(0, x - 4)], profile[min(width, x + 5) : x + 15]]
        )
        surrounding = float(around.mean()) if around.size else 0.0
        if not (column < _SEPARATOR_MAX and column < _SEPARATOR_RATIO * surrounding):
            return False
    return True


def tricode_crops(frame: np.ndarray) -> dict[str, np.ndarray]:
    """Crops of the two tricode areas (left team then right team); no OCR."""
    crops = {}
    for side, box in (("blue", BLUE_TRICODE_BOX), ("red", RED_TRICODE_BOX)):
        x0, y0, x1, y1 = box
        crops[side] = np.ascontiguousarray(frame[y0:y1, x0:x1])
    return crops


def _draft_template() -> np.ndarray:
    template = _template_cache.get(_TEMPLATE_PATH)
    if template is not None:
        return template
    panel = cv2.imread(str(_TEMPLATE_PATH), cv2.IMREAD_COLOR)
    if panel is None:
        raise FileNotFoundError(f"draft template missing: {_TEMPLATE_PATH}")
    y0, y1, x0, x1 = _PATCH_WINDOW
    panel_box = CENTER_PANEL_BOX
    template = np.ascontiguousarray(panel[y0 - panel_box[1] : y1 - panel_box[1], x0 - panel_box[0] : x1 - panel_box[0]])
    _template_cache[_TEMPLATE_PATH] = template
    return template


def is_draft_screen(frame: np.ndarray) -> tuple[bool, float]:
    """Cheap check that the champion-select overlay is visible (template of the patch line)."""
    template = _draft_template()
    y0, y1, x0, x1 = _PATCH_WINDOW
    if frame.shape[0] < y1 or frame.shape[1] < x1:
        return False, 0.0
    window = np.ascontiguousarray(frame[y0:y1, x0:x1])
    score = float(cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)[0, 0])
    return score >= _DRAFT_MIN_SCORE, score
