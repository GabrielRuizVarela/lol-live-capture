"""Which champion is in each draft slot (SIFT matcher over Data Dragon art)."""
from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from . import champions as champions_mod
from . import layout, paths
from .tracker import SlotRead

log = logging.getLogger(__name__)

PICK_REF_SIZE = (212, 386)
PICK_CROP_SIZE = (196, 326)
PICK_TINT_SIZE = (98, 163)
BAN_REF_SIZE = (40, 40)
BAN_CROP_SIZE = (80, 80)

PICK_REF_TOPN = 120
BAN_REF_TOPN = 100
MATCH_RATIO = 0.75

PICK_MIN_VOTES = 5.0
BAN_MIN_VOTES = 2.0
BAN_TIE_MARGIN = 1.0
BAN_FLAT_MODAL = 0.45  # empty ban slots are a flat team colour; icons are varied

LOCK_SHARPNESS = 13.5  # abs Laplacian mean: hovered (dimmed/blurred) art is flatter
TINT_HOVER_DELTA = 0.30  # hover adds a blue wash: crop B/((R+G)/2) exceeds the reference
HOVER_MIN_VALUE = 60.0  # empty pick slots are very dark (mean V < ~50)
HOVER_MIN_SATURATION = 60.0

BAN_ACTIVE_MIN_VALUE = 200.0  # the acting (empty, lighter team-colour + slash) ban slot is bright; locked art is darker

# An acting pick slot glows in team colour: its bright border is strongly hued.
# Measured on the Play-In broadcast: active slots score ~90-110, locked art < 45.
EDGE_TINT_MIN = 60.0

BANK_VERSION = 2

_SIFT = cv2.SIFT_create()
_MATCHER = cv2.BFMatcher()


def _sift_descriptors(gray: np.ndarray, topn: int | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    keypoints, descriptors = _SIFT.detectAndCompute(gray, None)
    if descriptors is None or len(descriptors) == 0:
        return None, None
    if topn is not None and len(descriptors) > topn:
        keep = np.argsort([-kp.response for kp in keypoints])[:topn]
        descriptors = descriptors[keep]
    return keypoints, descriptors


def _stacked_votes(descriptors: np.ndarray | None, bank_desc: np.ndarray, bank_labels: np.ndarray, count: int) -> np.ndarray:
    votes = np.zeros(count, dtype=np.float32)
    if descriptors is None or bank_desc.shape[0] == 0:
        return votes
    for pair in _MATCHER.knnMatch(descriptors, bank_desc, k=2):
        if len(pair) < 2:
            continue
        best, second = pair
        if best.distance < MATCH_RATIO * second.distance:
            votes[bank_labels[best.trainIdx]] += 1.0
    return votes


def _top_two(votes: np.ndarray) -> tuple[int, float, float]:
    if votes.size == 0:
        return -1, 0.0, 0.0
    order = np.argpartition(votes, -2)[-2:]
    order = order[np.argsort(votes[order])[::-1]]
    best, second = int(order[0]), int(order[1])
    return best, float(votes[best]), float(votes[best] - votes[second])


def _top_candidates(
    votes: np.ndarray, names: list[str], bank: "ReferenceBank", count: int = 3
) -> tuple[tuple[str, float], ...]:
    """The ``count`` best-matching champions as ``(riot_id, score)``, best first."""
    if votes.size == 0:
        return ()
    order = np.argsort(votes)[::-1][:count]
    out: list[tuple[str, float]] = []
    for index in order:
        index = int(index)
        score = float(votes[index])
        if score <= 0:
            break
        champion_id = _champion_id(bank, names[index])
        if champion_id is not None:
            out.append((champion_id, score))
    return tuple(out)


def _crop_gray(frame: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1]
    scaled = cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)


def _crop_color(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    return frame[y0:y1, x0:x1]


def _blue_tint(frame: np.ndarray) -> float:
    """How blue a crop is: B / mean(R, G). Hovered picks gain a blue wash."""
    crop = frame.astype(np.float32)
    blue = float(crop[:, :, 0].mean())
    red = float(crop[:, :, 2].mean())
    green = float(crop[:, :, 1].mean())
    return blue / ((red + green) / 2.0 + 1e-6)


def _crop_features(frame: np.ndarray, box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    crop = _crop_color(frame, box)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    laplacian = float(np.abs(cv2.Laplacian(gray, cv2.CV_32F)).mean())
    value = float(hsv[:, :, 2].mean())
    saturation = float(hsv[:, :, 1].mean())
    return value, saturation, laplacian, _blue_tint(crop)


class ReferenceBank:
    """SIFT descriptors and 40x40 squares for every champion, built once per patch."""

    def __init__(
        self,
        *,
        pick_names: list[str],
        pick_desc: np.ndarray,
        pick_labels: np.ndarray,
        ban_names: list[str],
        ban_desc: np.ndarray,
        ban_labels: np.ndarray,
        squares: dict[str, np.ndarray],
        champion_ids: dict[str, str],
        signature: str,
        pick_tints: dict[str, float] | None = None,
    ) -> None:
        self.pick_names = pick_names
        self.pick_desc = pick_desc
        self.pick_labels = pick_labels
        self.ban_names = ban_names
        self.ban_desc = ban_desc
        self.ban_labels = ban_labels
        self.squares = squares
        self.champion_ids = champion_ids
        self.signature = signature
        self.pick_tints = pick_tints or {}

    @classmethod
    def load(cls, champions_dir: Path | str | None = None, cache_dir: Path | str | None = None) -> "ReferenceBank":
        """Build (or load from ``cache_dir``) the bank from the art ``lol-live-capture setup`` downloaded."""
        champions = Path(champions_dir) if champions_dir is not None else paths.champions_dir()
        loading_dir = champions / "loading"
        squares_dir = champions
        if not loading_dir.is_dir():
            raise FileNotFoundError(f"champion art not found: {loading_dir} (run `lol-live-capture setup`)")
        champion_ids, ddragon_sig = _load_champion_ids(champions)

        pick_files = sorted(loading_dir.glob("*.jpg"))
        ban_files = sorted(p for p in squares_dir.glob("*.png") if p.stem != "manifest")
        signature = _bank_signature(ddragon_sig, pick_files, ban_files)

        if cache_dir is not None:
            cached = cls._load_cache(Path(cache_dir), signature)
            if cached is not None:
                log.info("reference bank loaded from cache (%s)", signature)
                return cached

        pick_names: list[str] = []
        pick_parts: list[np.ndarray] = []
        pick_labels: list[int] = []
        pick_tints: dict[str, float] = {}
        ban_names: list[str] = []
        ban_parts: list[np.ndarray] = []
        ban_labels: list[int] = []
        squares: dict[str, np.ndarray] = {}

        for index, path in enumerate(pick_files):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                log.warning("skipping unreadable loading art: %s", path.name)
                continue
            resized = cv2.resize(image, PICK_REF_SIZE, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            _, descriptors = _sift_descriptors(gray, PICK_REF_TOPN)
            if descriptors is None:
                log.warning("no SIFT features in loading art: %s", path.name)
                continue
            pick_names.append(path.stem)
            pick_parts.append(descriptors)
            pick_labels.extend([len(pick_names) - 1] * len(descriptors))
            tint_ref = cv2.resize(image, PICK_TINT_SIZE, interpolation=cv2.INTER_AREA)
            pick_tints[path.stem] = _blue_tint(tint_ref)

        for path in ban_files:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                log.warning("skipping unreadable square icon: %s", path.name)
                continue
            square = cv2.resize(image, BAN_REF_SIZE, interpolation=cv2.INTER_AREA)
            squares[path.stem] = square
            gray = cv2.cvtColor(cv2.resize(image, BAN_CROP_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            _, descriptors = _sift_descriptors(gray, BAN_REF_TOPN)
            if descriptors is None:
                log.warning("no SIFT features in square icon: %s", path.name)
                continue
            ban_names.append(path.stem)
            ban_parts.append(descriptors)
            ban_labels.extend([len(ban_names) - 1] * len(descriptors))

        bank = cls(
            pick_names=pick_names,
            pick_desc=np.vstack(pick_parts).astype(np.float32),
            pick_labels=np.asarray(pick_labels, dtype=np.int32),
            ban_names=ban_names,
            ban_desc=np.vstack(ban_parts).astype(np.float32),
            ban_labels=np.asarray(ban_labels, dtype=np.int32),
            squares=squares,
            champion_ids=champion_ids,
            signature=signature,
            pick_tints=pick_tints,
        )
        if cache_dir is not None:
            bank.save_cache(Path(cache_dir))
        log.info(
            "reference bank built: %d picks (%d descriptors), %d bans (%d descriptors)",
            len(pick_names),
            bank.pick_desc.shape[0],
            len(ban_names),
            bank.ban_desc.shape[0],
        )
        return bank

    def save_cache(self, cache_dir: Path) -> None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        target = cache_dir / f"refs-v{BANK_VERSION}-{self.signature}.npz"
        square_names = sorted(self.squares)
        squares = np.stack([self.squares[n] for n in square_names]) if square_names else np.zeros((0, 40, 40, 3), np.uint8)
        id_names = sorted(self.champion_ids)
        tint_names = sorted(self.pick_tints)
        payload = {
            "pick_names": np.asarray(self.pick_names),
            "pick_desc": self.pick_desc,
            "pick_labels": self.pick_labels,
            "tint_names": np.asarray(tint_names),
            "tint_values": np.asarray([self.pick_tints[n] for n in tint_names], dtype=np.float32),
            "ban_names": np.asarray(self.ban_names),
            "ban_desc": self.ban_desc,
            "ban_labels": self.ban_labels,
            "square_names": np.asarray(square_names),
            "squares": squares,
            "id_names": np.asarray(id_names),
            "ids": np.asarray([self.champion_ids[n] for n in id_names]),
            "signature": np.asarray(self.signature),
        }
        tmp = target.with_suffix(".npz.tmp")
        with open(tmp, "wb") as handle:
            np.savez_compressed(handle, **payload)
        tmp.replace(target)

    @classmethod
    def _load_cache(cls, cache_dir: Path, signature: str) -> "ReferenceBank | None":
        if not cache_dir.is_dir():
            return None
        target = cache_dir / f"refs-v{BANK_VERSION}-{signature}.npz"
        if not target.is_file():
            return None
        try:
            with np.load(target, allow_pickle=False) as data:
                square_names = [str(n) for n in data["square_names"]]
                squares = {n: data["squares"][i] for i, n in enumerate(square_names)}
                id_names = [str(n) for n in data["id_names"]]
                champion_ids = {n: str(v) for n, v in zip(id_names, data["ids"])}
                tint_names = [str(n) for n in data["tint_names"]]
                pick_tints = {n: float(v) for n, v in zip(tint_names, data["tint_values"])}
                return cls(
                    pick_names=[str(n) for n in data["pick_names"]],
                    pick_desc=data["pick_desc"],
                    pick_labels=data["pick_labels"],
                    ban_names=[str(n) for n in data["ban_names"]],
                    ban_desc=data["ban_desc"],
                    ban_labels=data["ban_labels"],
                    squares=squares,
                    champion_ids=champion_ids,
                    signature=str(data["signature"]),
                    pick_tints=pick_tints,
                )
        except Exception as exc:  # noqa: BLE001 - a broken cache is rebuilt
            log.warning("ignoring broken reference cache %s: %s", target, exc)
            return None


def _load_champion_ids(champions_dir: Path) -> tuple[dict[str, str], str]:
    """Data Dragon id -> Riot key, plus a signature of the table (the downloaded one wins over the bundled)."""
    downloaded = champions_dir / champions_mod.FILE_NAME
    path = downloaded if downloaded.is_file() else champions_mod.BUNDLED
    return champions_mod.id_to_key(path), f"{path.name}:{path.stat().st_size}"


def _bank_signature(ddragon_sig: str, pick_files: Iterable[Path], ban_files: Iterable[Path]) -> str:
    digest = hashlib.sha1()
    digest.update(ddragon_sig.encode())
    for path in pick_files:
        digest.update(path.name.encode())
    for path in ban_files:
        digest.update(path.name.encode())
    return digest.hexdigest()[:16]


def read_slot(
    crop_gray: np.ndarray,
    kind: str,
    bank: ReferenceBank,
    *,
    features: tuple[float, float, float, float] | None = None,
) -> SlotRead:
    """Match one slot crop (already resized to the matcher's working size)."""
    if kind == "pick":
        votes = _stacked_votes(
            _sift_descriptors(crop_gray, None)[1], bank.pick_desc, bank.pick_labels, len(bank.pick_names)
        )
        name_index, score, margin = _top_two(votes)
        candidates = _top_candidates(votes, bank.pick_names, bank)
        brightness = features[0] if features else 0.0
        if score >= PICK_MIN_VOTES:
            name = bank.pick_names[name_index]
            hover = False
            if features is not None:
                reference_tint = bank.pick_tints.get(name)
                tint_delta = features[3] - reference_tint if reference_tint is not None else 0.0
                hover = features[2] < LOCK_SHARPNESS or tint_delta > TINT_HOVER_DELTA
            state = "hover" if hover else "locked"
            return SlotRead(state, _champion_id(bank, name), name, score, margin, brightness, candidates)
        if features is not None and brightness >= HOVER_MIN_VALUE and features[1] >= HOVER_MIN_SATURATION:
            return SlotRead("hover", None, None, score, margin, brightness, candidates)
        return SlotRead("empty", None, None, score, margin, brightness, candidates)

    votes = _stacked_votes(
        _sift_descriptors(crop_gray, None)[1], bank.ban_desc, bank.ban_labels, len(bank.ban_names)
    )
    name_index, score, margin = _top_two(votes)
    candidates = _top_candidates(votes, bank.ban_names, bank)
    brightness = features[0] if features else 0.0
    if score >= BAN_MIN_VOTES and margin > BAN_MIN_VOTES:
        name = bank.ban_names[name_index]
        if margin <= BAN_TIE_MARGIN:
            name = _ban_template_tiebreak(crop_gray, bank, name_index, votes)
        return SlotRead("locked", _champion_id(bank, name), name, score, margin, brightness, candidates)
    return SlotRead("empty", None, None, score, margin, brightness, candidates)


def _champion_id(bank: ReferenceBank, name: str) -> str | None:
    champion_id = bank.champion_ids.get(name)
    if champion_id is None:
        log.warning("no riot id for champion %s", name)
    return champion_id


def _ban_template_tiebreak(crop_gray: np.ndarray, bank: ReferenceBank, best_index: int, votes: np.ndarray) -> str:
    candidates = [best_index]
    for index in np.argsort(votes)[::-1]:
        if votes[index] <= 0:
            break
        if int(index) != best_index and votes[best_index] - votes[index] <= BAN_TIE_MARGIN:
            candidates.append(int(index))
    query = cv2.resize(crop_gray, BAN_REF_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32)
    best_name = bank.ban_names[best_index]
    best_score = -2.0
    for index in candidates:
        name = bank.ban_names[index]
        square = bank.squares.get(name)
        if square is None:
            continue
        reference = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY).astype(np.float32)
        score = float(cv2.matchTemplate(query, reference, cv2.TM_CCOEFF_NORMED)[0, 0])
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


def _is_flat_ban(crop: np.ndarray) -> bool:
    """An empty ban slot is a flat team colour (+ slash); its quantised colours repeat."""
    quantised = (crop // 16).reshape(-1, 3)
    _, counts = np.unique(quantised, axis=0, return_counts=True)
    return float(counts.max()) / quantised.shape[0] >= BAN_FLAT_MODAL


def _read_ban_slot(frame: np.ndarray, side: str, index: int, bank: ReferenceBank) -> SlotRead:
    box = layout.ban_art_box(side, index)
    x0, y0, x1, y1 = box
    crop = frame[y0:y1, x0:x1]
    features = _crop_features(frame, box)
    if _is_flat_ban(crop):
        # Empty ban slots are a flat team colour (+ slash). The lighter one is
        # the acting slot (being chosen right now); the darker ones are spent.
        if features[0] >= BAN_ACTIVE_MIN_VALUE:
            return SlotRead("active", None, None, 0.0, 0.0, features[0])
        return SlotRead("empty", None, None, 0.0, 0.0, features[0])
    gray = _crop_gray(frame, box, BAN_CROP_SIZE)
    return read_slot(gray, "ban", bank, features=features)


def _read_pick_slot(frame: np.ndarray, side: str, index: int, bank: ReferenceBank) -> SlotRead:
    box = layout.pick_art_box(side, index)
    gray = _crop_gray(frame, box, PICK_CROP_SIZE)
    features = _crop_features(frame, box)
    return read_slot(gray, "pick", bank, features=features)


def _edge_team_tint(frame: np.ndarray, side: str, index: int) -> float:
    """Strongest team hue on the border of one pick slot (the acting glow)."""
    x0, y0, x1, y1 = layout.pick_art_box(side, index)
    sx0, sx1 = x0 - layout.PICK_ART_INSET, x1 + layout.PICK_ART_INSET
    left = frame[y0 + 3 : y1, sx0 : sx0 + 4]
    right = frame[y0 + 3 : y1, sx1 - 4 : sx1]
    top = frame[y0 : y0 + 4, sx0:sx1]
    bottom = frame[y1 - 4 : y1, sx0:sx1]
    scores = []
    for edge in (left, right, top, bottom):
        crop = edge.astype(np.float32)
        blue = float(crop[:, :, 0].mean())
        green = float(crop[:, :, 1].mean())
        red = float(crop[:, :, 2].mean())
        if side == "blue":
            scores.append(blue - (red + green) / 2.0)
        else:
            scores.append(red - (blue + green) / 2.0)
    return max(scores)


def _is_ban(slot: str) -> bool:
    return "_ban_" in slot


def _detect_active(frame: np.ndarray, reads: dict[str, SlotRead], rest: dict[str, bool]) -> str | None:
    """The one slot whose turn it is, from the overlay's own cues."""
    side = layout.acting_side(frame)
    ban_active = [slot for slot in reads if _is_ban(slot) and reads[slot].state == "active"]
    if ban_active:
        on_side = [slot for slot in ban_active if side and slot.startswith(f"{side}_")]
        return (on_side or ban_active)[0]
    if side is None or not rest.get(side, False):
        return None
    best_index, best_score = None, 0.0
    second_best = 0.0
    for index in range(1, 6):
        score = _edge_team_tint(frame, side, index)
        if score > best_score:
            second_best = best_score
            best_index, best_score = index, score
        elif score > second_best:
            second_best = score
    margin = best_score - second_best
    if best_index is not None and best_score >= EDGE_TINT_MIN and margin >= 15.0:
        return f"{side}_pick_{best_index}"
    return None


_SLOT_KEYS: tuple[str, ...] = (  # noqa: E741
    "blue_ban_1", "blue_ban_2", "blue_ban_3", "blue_ban_4", "blue_ban_5",
    "red_ban_1", "red_ban_2", "red_ban_3", "red_ban_4", "red_ban_5",
    "blue_pick_1", "blue_pick_2", "blue_pick_3", "blue_pick_4", "blue_pick_5",
    "red_pick_1", "red_pick_2", "red_pick_3", "red_pick_4", "red_pick_5",
)


def read_frame(frame: np.ndarray, bank: ReferenceBank) -> dict[str, SlotRead]:
    """Read all 20 draft slots of one frame (slots in parallel).

    A side whose portraits are not at rest (a stat-card expansion is on screen)
    has all its pick slots reported ``unknown``: the fixed boxes cut through the
    wrong art.  At most one slot is reported ``active`` (the current turn).
    """
    rest = {side: layout.pick_at_rest(frame, side) for side in ("blue", "red")}
    is_draft, _ = layout.is_draft_screen(frame)
    if not is_draft:
        # The draft screen is not visible: all slots are empty/unreadable.
        return {slot: SlotRead("empty", None, None, 0.0, 0.0) for slot in _SLOT_KEYS}

    tasks: list[tuple[str, str, str, int]] = []
    for side in ("blue", "red"):
        for index in range(1, 6):
            tasks.append((f"{side}_ban_{index}", "ban", side, index))
    for side in ("blue", "red"):
        for index in range(1, 6):
            tasks.append((f"{side}_pick_{index}", "pick", side, index))

    def run(task: tuple[str, str, str, int]) -> SlotRead:
        _, kind, side, index = task
        if kind == "ban":
            return _read_ban_slot(frame, side, index, bank)
        if not rest[side]:
            return SlotRead("unknown", None, None, 0.0, 0.0)
        return _read_pick_slot(frame, side, index, bank)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run, tasks))
    reads = {key: result for (key, _, _, _), result in zip(tasks, results)}

    active = _detect_active(frame, reads, rest)
    if active is not None:
        base = reads[active]
        reads[active] = SlotRead(
            "active", base.champion_id, base.name, base.score, base.margin, base.brightness,
            base.candidates,
        )
    return reads
