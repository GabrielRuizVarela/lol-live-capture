"""Rule-based champion mentions from transcript text.

The :class:`Spotter` knows the champion names from Data Dragon plus the
aliases and Whisper misspellings that actually show up in caster speech.  It
matches words fuzzily and attaches a *cue* (``ban`` / ``pick`` / ``speculative``
/ ``neutral``) taken from the words around the match.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from . import champions as champions_mod


# Champion id -> extra tokens casters use (or Whisper hears).  Anything here is
# also added to the hotword bias.
ALIASES: dict[str, tuple[str, ...]] = {
    "62": ("wukong", "monkey king", "monkey"),
    "4": ("tf", "twisted fate", "twistedfate"),
    "59": ("j4", "jarvan", "jarvan iv", "jarvaniv"),
    "5": ("xin", "xin zhao", "xinzhao"),
    "36": ("mundo", "dr mundo", "drmundo"),
    "96": ("kog", "kog maw", "kogmaw", "kogmow"),
    "897": ("ksante", "kassante", "k sante", "kaysante", "kesante"),
    "136": ("asol", "aurelion", "aurelion sol", "aurelionsol"),
    "7": ("lb", "leblanc", "le blanc"),
    "21": ("mf", "miss fortune", "missfortune"),
    "41": ("gp", "gangplank"),
    "145": ("kaisa", "kai sa", "kaisa", "kaysa", "kysa"),
    "200": ("belveth", "bel veth", "bel"),
    "31": ("cho", "cho gath", "chogath", "chogat"),
    "161": ("velkoz", "vel koz", "vel"),
    "421": ("reksai", "rek sai", "rek"),
    "121": ("khazix", "kha zix", "kha"),
    "20": ("nunu", "nunu and willump", "nunu willump", "willump"),
    "888": ("renata", "renata glasc", "renataglasc"),
    "804": ("yunara",),
    "111": ("naut", "nautilus"),
    "223": ("tahm", "tahm kench", "tahmkench"),
    "517": ("sylas",),
    "555": ("pyke",),
    "166": ("akshan",),
    "523": ("aphelios",),
    "147": ("seraphine", "sera"),
    "875": ("sett",),
    "246": ("qiyana",),
    "234": ("viego",),
    "711": ("vex",),
    "902": ("milio",),
    "233": ("briar",),
    "895": ("nilah",),
    "910": ("hwei",),
    "901": ("smolder",),
    "950": ("naafiri",),
    "117": ("lulu",),
}

_SPECULATIVE = (
    "could",
    "couldn't",
    "couldnt",
    "maybe",
    "might",
    "would",
    "if",
    "possibly",
    "perhaps",
    "guess",
    "guessing",
    "think",
    "thinking",
    "consider",
    "considering",
    "what about",
    "maybe",
    "potentially",
)
_BAN = (
    "ban",
    "bans",
    "banned",
    "banning",
    "ban's",
    "takes away",
    "take away",
    "taken away",
    "won't see",
    "wont see",
    "off the board",
    "remove",
    "removed",
    "removing",
)
_PICK = (
    "lock",
    "locks",
    "locked",
    "locking",
    "lock in",
    "locked in",
    "first pick",
    "picks",
    "picked",
    "picking",
    "pick",
    "go",
    "goes",
    "going",
    "coming out",
    "comes out",
    "comes",
    "coming",
    "take",
    "takes",
    "grab",
    "grabs",
    "grab a",
    "go with",
    "with the",
)

_WORD_RE = re.compile(r"[a-z0-9]+")

# Draft-commentary words ("lock", "picks", "ban"): never fuzzy-matched to a champion, or a new champion with a
# similar name (Locke) would be "heard" every time a caster says "lock in". An exact alias still matches.
_CUE_WORDS = frozenset(w for group in (_SPECULATIVE, _BAN, _PICK) for phrase in group for w in phrase.split())


def _norm(text: str) -> str:
    return "".join(_WORD_RE.findall(text.lower()))


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


class _AliasTable:
    def __init__(self, path: Path | None = None) -> None:
        self.id_to_name: dict[str, str] = {}
        self.alias_to_id: dict[str, str] = {}
        champions = self._load_champions(path)
        for cid, name in champions.items():
            self.id_to_name[cid] = name
            self.alias_to_id[_norm(name)] = cid
        for cid, aliases in ALIASES.items():
            for alias in aliases:
                key = _norm(alias)
                self.alias_to_id.setdefault(key, cid)
        # longest aliases first so "jarvan iv" beats "jarvan"
        self._sorted = sorted(self.alias_to_id, key=len, reverse=True)

    @staticmethod
    def _load_champions(path: Path | None) -> dict[str, str]:
        try:
            return champions_mod.key_to_name(path)
        except FileNotFoundError:
            return {}

    def lookup(self, token: str) -> tuple[str, float] | None:
        key = _norm(token)
        if not key:
            return None
        cid = self.alias_to_id.get(key)
        if cid is not None:
            return cid, 1.0
        if len(key) < 4 or key in _CUE_WORDS:
            return None
        best: tuple[str, float] | None = None
        for alias in self._sorted:
            if abs(len(alias) - len(key)) > 2 or len(alias) < 4:
                continue
            ratio = SequenceMatcher(None, key, alias).ratio()
            if ratio >= 0.86 and (best is None or ratio > best[1]):
                best = (self.alias_to_id[alias], ratio)
        return best

    def hotwords(self) -> list[str]:
        words: set[str] = set()
        for name in self.id_to_name.values():
            words.add(name)
        for aliases in ALIASES.values():
            for alias in aliases:
                if len(alias) > 1:
                    words.add(alias)
        return sorted(words)


_TABLE: _AliasTable | None = None


def alias_table() -> _AliasTable:
    global _TABLE
    if _TABLE is None:
        _TABLE = _AliasTable()
    return _TABLE


def champion_hotwords() -> list[str]:
    return alias_table().hotwords()


def lookup(token: str) -> tuple[str, float] | None:
    return alias_table().lookup(token)


@dataclass(frozen=True)
class Mention:
    t: float
    champion_id: str
    cue: str
    text: str
    score: float


def _cue_around(tokens: list[str], start: int, end: int) -> str:
    window = " ".join(tokens[max(0, start - 8) : end + 5])
    for phrase in _SPECULATIVE:
        if phrase in window:
            return "speculative"
    for phrase in _BAN:
        if phrase in window:
            return "ban"
    for phrase in _PICK:
        if phrase in window:
            return "pick"
    return "neutral"


class Spotter:
    def __init__(self, table: _AliasTable | None = None) -> None:
        self.table = table or alias_table()

    def spot(self, t: float, text: str) -> list[Mention]:
        tokens = _tokens(text)
        out: list[Mention] = []
        i = 0
        n = len(tokens)
        while i < n:
            found: tuple[int, str, float] | None = None
            for size in (3, 2, 1):
                if i + size > n:
                    continue
                phrase = " ".join(tokens[i : i + size])
                hit = self.table.lookup(phrase)
                if hit is not None:
                    found = (size, hit[0], hit[1])
                    break
            if found is None:
                i += 1
                continue
            size, cid, score = found
            cue = _cue_around(tokens, i, i + size)
            out.append(Mention(t=t, champion_id=cid, cue=cue, text=text, score=score))
            i += size
        return out

    def mentions(self, segments: Iterable[object]) -> list[Mention]:
        out: list[Mention] = []
        for seg in segments:
            t = getattr(seg, "t0", None)
            text = getattr(seg, "text", None)
            if t is None or text is None:
                t, _, text = seg  # type: ignore[misc]
            out.extend(self.spot(float(t), str(text)))
        return out


def spot_segments(segments: Iterable[object]) -> list[Mention]:
    return Spotter().mentions(segments)
