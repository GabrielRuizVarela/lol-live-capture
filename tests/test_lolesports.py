"""Tests for the lolesports confirmation client (fake fetch, real ddragon map)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_live_capture import lolesports as lo


FIXTURES = Path(__file__).parent / "fixtures" / "lolesports"

WINDOW_URL = (
    "https://feed.lolesports.com/livestats/v1/window/115660588583171791"
    "?startingTime=2026-10-16T18:11:40.000Z"
)
WINDOW_BEFORE_URL = (
    "https://feed.lolesports.com/livestats/v1/window/115660588583171791"
    "?startingTime=2026-10-16T18:10:40.000Z"
)
WINDOW_G2_URL = (
    "https://feed.lolesports.com/livestats/v1/window/115660588583171792"
    "?startingTime=2026-10-16T18:11:40.000Z"
)
GET_LIVE_URL = "https://esports-api.lolesports.com/persisted/gw/getLive?hl=en-US"
SCHEDULE_URL = (
    "https://esports-api.lolesports.com/persisted/gw/getSchedule"
    f"?hl=en-US&leagueId={lo.WORLDS_LEAGUE_ID}"
)
EVENT_URL = (
    "https://esports-api.lolesports.com/persisted/gw/getEventDetails"
    "?hl=en-US&id=115660588583171790"
)


@pytest.fixture()
def ddragon():
    """The bundled Data Dragon table (no ``setup`` download needed)."""


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def fetch_route(mapping):
    """Fake fetch: exact-URL dict lookup, {} maps to the 204-empty body."""

    def fetch(url, headers):
        if url not in mapping:
            raise AssertionError(f"unexpected fetch {url}")
        return mapping[url]

    return fetch


def test_live_games_parses_codes_and_game(ddragon):
    fetch = fetch_route({GET_LIVE_URL: _load("getLive_worlds.json")})
    games = lo.live_games(fetch, league_id=lo.WORLDS_LEAGUE_ID)
    assert [g.match_id for g in games] == ["115660588583171790"]
    assert games[0].game_id == "115660588583171792"
    assert games[0].game_number == 2
    assert {games[0].blue_code, games[0].red_code} == {"KC", "MVK"}
    assert games[0].blue_name == "Karmine Corp"


def test_champions_maps_ddragon_ids(ddragon):
    fetch = fetch_route({WINDOW_URL: _load("window_kcmvk_g1.json")})
    champs = lo.game_champions(
        "115660588583171791",
        now="2026-10-16T18:13:10+00:00",
        fetch=fetch,
    )
    assert champs is not None
    assert champs.game_id == "115660588583171791"
    assert champs.blue == ["126", "254", "134", "119", "111"]
    assert champs.red == ["799", "57", "897", "18", "117"]
    assert champs.bans is None


def test_confirm_all_confirmed(ddragon):
    fetch = fetch_route({WINDOW_URL: _load("window_kcmvk_g1.json")})
    champs = lo.game_champions("115660588583171791",
                               now="2026-10-16T18:13:10+00:00", fetch=fetch)
    blue = ["126", "254", "134", "119", "111"]
    red = ["799", "57", "897", "18", "117"]
    actions = [
        {"sequence": 5, "side": "blue", "action_type": "pick",
         "champion_id": str(champ)}
        for champ in blue
    ] + [
        {"sequence": 10, "side": "red", "action_type": "pick",
         "champion_id": str(champ)}
        for champ in red
    ] + [
        {"sequence": 1, "side": "blue", "action_type": "ban",
         "champion_id": "212"},
    ]
    result = lo.confirm(actions, champs)
    assert result["status"] == "confirmed"
    assert result["picks"]["blue"] == sorted(blue)
    assert result["picks"]["red"] == sorted(red)
    assert result["mismatches"] == []
    assert result["bans"] is None  # the feed has no bans today
    assert result["game_id"] == "115660588583171791"


def test_confirm_mismatch(ddragon):
    fetch = fetch_route({WINDOW_URL: _load("window_kcmvk_g1.json")})
    champs = lo.game_champions("115660588583171791",
                               now="2026-10-16T18:13:10+00:00", fetch=fetch)
    blue = ["126", "254", "134", "119", "111"]
    red = ["799", "57", "897", "18", "117"]
    actions = [
        {"sequence": n, "side": "blue", "action_type": "pick", "champion_id": c}
        for n, c in zip(range(5, 10), blue)
    ] + [
        {"sequence": n, "side": "red", "action_type": "pick", "champion_id": c}
        for n, c in zip(range(10, 15), red)
    ]
    actions[7]["champion_id"] = "1"  # red pick wrong: feed red has no "1"
    result = lo.confirm(actions, champs)
    assert result["status"] == "mismatch"
    assert result["mismatches"] == [
        {"sequence": 12, "captured": "1", "feed": ["897"]}
    ]
    assert result["missing"]["blue"] == []
    assert result["missing"]["red"] == ["897"]


def test_confirm_partial_lists_missing(ddragon):
    fetch = fetch_route({WINDOW_URL: _load("window_kcmvk_g1.json")})
    champs = lo.game_champions("115660588583171791",
                               now="2026-10-16T18:13:10+00:00", fetch=fetch)
    actions = [
        {"sequence": 5, "side": "blue", "action_type": "pick", "champion_id": "126"},
        {"sequence": 7, "side": "blue", "action_type": "pick", "champion_id": "134"},
    ]
    result = lo.confirm(actions, champs)
    assert result["status"] == "partial"
    assert result["missing"]["blue"] == ["111", "119", "254"]
    assert result["missing"]["red"] == ["117", "18", "57", "799", "897"]
    assert result["mismatches"] == []


def test_find_game_swapped_sides(ddragon):
    empty_schedule = {"data": {"schedule": {"events": []}}}
    fetch = fetch_route({GET_LIVE_URL: _load("getLive_worlds.json"),
                         SCHEDULE_URL: empty_schedule})
    # capture saw MVK as blue, but the live event had KC blue: still found
    assert lo.find_game("MVK", "KC", 2, fetch=fetch) == "115660588583171792"
    assert lo.find_game("KC", "MVK", 3, fetch=fetch) is None


def test_find_game_by_team_names_and_swapped_sides(ddragon):
    empty_schedule = {"data": {"schedule": {"events": []}}}
    fetch = fetch_route({GET_LIVE_URL: _load("getLive_worlds.json"),
                         SCHEDULE_URL: empty_schedule})
    assert lo.find_game("Karmine Corp", "MVK Esports", 2, fetch=fetch) == "115660588583171792"
    assert lo.find_game("MVK Esports", "Karmine Corp", 2, fetch=fetch) == "115660588583171792"


def test_find_game_matches_names_accent_and_case_insensitively(ddragon):
    empty_schedule = {"data": {"schedule": {"events": []}}}
    fetch = fetch_route({GET_LIVE_URL: _load("getLive_worlds.json"),
                         SCHEDULE_URL: empty_schedule})
    assert lo.find_game("karminé corp", "MVK ESPORTS", 2, fetch=fetch) == "115660588583171792"


def test_find_game_uses_names_when_tricodes_are_missing(ddragon):
    fetch = fetch_route({GET_LIVE_URL: _load("getLive_worlds.json")})
    assert lo.find_game(None, None, 2, blue_name="Karmine Corp",
                        red_name="MVK Esports", fetch=fetch) == "115660588583171792"


def test_find_game_uses_event_details_fallback(ddragon):
    today = __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
    event_fixture = _load("event_details_kcmvk.json")
    schedule = {
        "data": {"schedule": {"events": [
            {
                "startTime": today.isoformat(timespec="seconds"),
                "match": {
                    "id": "115660588583171790",
                    "teams": [{"code": "KC"}, {"code": "MVK"}],
                },
            }
        ]}}
    }
    fetch = fetch_route({
        GET_LIVE_URL: _load("getLive_worlds.json"),
        SCHEDULE_URL: schedule,
        EVENT_URL: event_fixture,
    })
    game = lo.find_game("KC", "MVK", 1, fetch=fetch)
    assert game == "115660588583171791"  # number 1, sides per details


def test_attach_confirmation_unavailable_on_network_error(ddragon):
    def fetch(url, headers):
        raise __import__("urllib.error", fromlist=["URLError"]).URLError("conn refused")

    state = {
        "teams": {"blue": {"tricode": "KC"}, "red": {"tricode": "MVK"}},
        "game_index": 1,
        "actions": [],
    }
    result = lo.attach_confirmation(state, fetch=fetch)
    assert result["lolesports"]["status"] == "unavailable"
    assert "conn refused" in result["lolesports"]["message"]


def test_attach_confirmation_uses_names_when_tricodes_missing(ddragon):
    def fetch(url, headers):
        if url == GET_LIVE_URL:
            return _load("getLive_worlds.json")
        if url == WINDOW_G2_URL:
            return _load("window_kcmvk_g2.json")
        raise AssertionError(f"unexpected fetch {url}")

    state = {
        "teams": {"blue": {"name": "Karmine Corp"}, "red": {"name": "MVK Esports"}},
        "game_index": 2,
        "actions": [
            {"sequence": n, "side": "blue", "action_type": "pick", "champion_id": c}
            for n, c in zip(range(5, 10), ["3", "64", "103", "22", "147"])
        ] + [
            {"sequence": n, "side": "red", "action_type": "pick", "champion_id": c}
            for n, c in zip(range(10, 15), ["4", "246", "68", "81", "555"])
        ],
    }
    result = lo.attach_confirmation(
        state, fetch=fetch, now="2026-10-16T18:13:10+00:00")
    assert result["lolesports"]["status"] == "confirmed"
    assert result["lolesports"]["game_id"] == "115660588583171792"


def test_attach_confirmation_confirmed(ddragon):
    def fetch(url, headers):
        if url == GET_LIVE_URL:
            return _load("getLive_worlds.json")
        if url == WINDOW_G2_URL:
            return _load("window_kcmvk_g2.json")
        raise AssertionError(f"unexpected fetch {url}")

    state = {
        "teams": {"blue": {"tricode": "KC"}, "red": {"tricode": "MVK"}},
        "game_index": 2,
        "actions": [
            {"sequence": n, "side": "blue", "action_type": "pick",
             "champion_id": c}
            for n, c in zip(range(5, 10), ["3", "64", "103", "22", "147"])
        ] + [
            {"sequence": n, "side": "red", "action_type": "pick",
             "champion_id": c}
            for n, c in zip(range(10, 15), ["4", "246", "68", "81", "555"])
        ],
    }
    result = lo.attach_confirmation(
        state, fetch=fetch, now="2026-10-16T18:13:10+00:00")
    entry = result["lolesports"]
    assert entry["status"] == "confirmed"
    assert entry["game_id"] == "115660588583171792"
    assert entry["checked_at"]
