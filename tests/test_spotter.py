"""Tests for the rule-based champion spotter."""
from __future__ import annotations

import pytest

from lol_live_capture import spotter
from lol_live_capture.spotter import Spotter, champion_hotwords

s = Spotter()


def _ids(text: str) -> list[str]:
    return [m.champion_id for m in s.spot(0.0, text)]


@pytest.mark.parametrize(
    "text,cid",
    [
        ("they ban Wukong", "62"),
        ("Monkey King is gone", "62"),
        ("Twisted Fate first pick", "4"),
        ("lock in TF", "4"),
        ("J4 comes out", "59"),
        ("Jarvan IV is picked", "59"),
        ("Xin Zhao banned", "5"),
        ("Dr Mundo", "36"),
        ("Kog'Maw", "96"),
        ("K'Sante", "897"),
        ("Ksante", "897"),
        ("Kassante", "897"),
        ("Aurelion Sol", "136"),
        ("ASol", "136"),
        ("LeBlanc", "7"),
        ("LB", "7"),
        ("Miss Fortune", "21"),
        ("MF", "21"),
        ("Gangplank", "41"),
        ("GP", "41"),
        ("Kai'Sa", "145"),
        ("Kaisa", "145"),
        ("Bel'Veth", "200"),
        ("Cho'Gath", "31"),
        ("Vel'Koz", "161"),
        ("Rek'Sai", "421"),
        ("Kha'Zix", "121"),
        ("Nunu and Willump", "20"),
        ("Renata Glasc", "888"),
        ("Yunara", "804"),
    ],
)
def test_aliases(text: str, cid: str) -> None:
    assert cid in _ids(text)


def test_misspellings_fuzzy() -> None:
    assert "897" in _ids("Cassante")
    assert "136" in _ids("Ausol")
    assert "5" in _ids("Xinzhao")
    assert "145" in _ids("Kaysa")


def test_cues() -> None:
    assert s.spot(1.0, "they ban Wukong")[0].cue == "ban"
    assert s.spot(1.0, "KC locks in K'Sante")[0].cue == "pick"
    assert s.spot(1.0, "maybe LeBlanc could come out")[0].cue == "speculative"
    assert s.spot(1.0, "J4 and Xin look strong")[0].cue == "neutral"


def test_speculative_wins_over_ban() -> None:
    mentions = s.spot(1.0, "if they ban Wukong that would be rough")
    assert mentions[0].cue == "speculative"


def test_multiple_mentions_and_text_preserved() -> None:
    text = "Kalista then Renata Glasc are picked up"
    mentions = s.spot(12.5, text)
    assert [m.champion_id for m in mentions] == ["429", "888"]
    assert all(m.t == 12.5 and m.text == text for m in mentions)


def test_mentions_from_segments() -> None:
    class Seg:
        t0 = 3.0
        text = "they take Olaf"

    mentions = s.mentions([Seg()])
    assert mentions[0].champion_id == "2"
    assert mentions[0].t == 3.0


def test_hotwords_include_champions_and_aliases() -> None:
    words = champion_hotwords()
    assert "Wukong" in words and "monkey king" in words and "K'Sante" in words


def test_cue_words_never_fuzzy_match_a_champion():
    """"lock" is 0.89 similar to a champion called Locke; the cue word must not be heard as that champion."""
    table = spotter._AliasTable.__new__(spotter._AliasTable)
    table.id_to_name = {"805": "Locke"}
    table.alias_to_id = {"locke": "805"}
    table._sorted = ["locke"]
    assert table.lookup("lock") is None
    assert table.lookup("locke") == ("805", 1.0)
