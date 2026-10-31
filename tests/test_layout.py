"""Layout checks: at-rest detection and the acting-side bar on real frames."""
from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from lol_live_capture import layout

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "private"  # broadcast stills, not in the repository

if not FIXTURES.is_dir() or not any(FIXTURES.glob("holdout-*-band.jpg")):
    pytest.skip("needs the broadcast stills in tests/fixtures/private/", allow_module_level=True)
BAND_Y0 = 470  # the fixture bands are cropped at this frame row


def band_frame(name: str) -> np.ndarray:
    """Rebuild a full-size frame from a cropped ban-row + pick-band fixture."""
    band = cv2.imread(str(FIXTURES / f"holdout-{name}-band.jpg"), cv2.IMREAD_COLOR)
    assert band is not None, name
    frame = np.zeros((layout.FRAME_HEIGHT, layout.FRAME_WIDTH, 3), dtype=np.uint8)
    frame[BAND_Y0 : BAND_Y0 + band.shape[0], : band.shape[1]] = band
    return frame


def test_at_rest_blue_expanded_stat_card() -> None:
    # 2876: blue just locked THANATOS; red portraits are squeezed (not at rest).
    frame = band_frame("2876")
    assert layout.pick_at_rest(frame, "blue") is True
    assert layout.pick_at_rest(frame, "red") is False


def test_at_rest_blue_squeezed_red_at_rest() -> None:
    # 2906: blue picks 1-2 are widened stat cards; red is at rest.
    frame = band_frame("2906")
    assert layout.pick_at_rest(frame, "blue") is False
    assert layout.pick_at_rest(frame, "red") is True


def test_at_rest_both_sides() -> None:
    for name in ("3072", "3085"):
        frame = band_frame(name)
        assert layout.pick_at_rest(frame, "blue") is True, name
        assert layout.pick_at_rest(frame, "red") is True, name


def test_acting_side_from_bar() -> None:
    assert layout.acting_side(band_frame("2876")) == "blue"
    assert layout.acting_side(band_frame("2906")) == "red"
    assert layout.acting_side(band_frame("3072")) == "blue"
    assert layout.acting_side(band_frame("3085")) == "blue"


def test_acting_side_none_on_blank() -> None:
    blank = np.zeros((layout.FRAME_HEIGHT, layout.FRAME_WIDTH, 3), dtype=np.uint8)
    assert layout.acting_side(blank) is None
