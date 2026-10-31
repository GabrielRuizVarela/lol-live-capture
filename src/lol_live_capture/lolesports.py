"""lolesports confirmation: match the captured draft against the official feed.

Uses the public, unofficial lolesports endpoints that lolesports.com itself
uses (see docs/contract.md, "lolesports confirmation": the
x-api-key is a public key shipped for their web client, override with env
LOLESPORTS_API_KEY) plus the key-less livestats feed.

Standard library only (urllib, json, gzip).  All network access goes through
an injectable ``fetch(url, headers) -> dict`` (default: urllib with 5 s
timeouts and at most one request per second; errors surface as
``status: "unavailable"``).
"""
from __future__ import annotations

import gzip
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from . import champions as champions_mod

__all__ = [
    "LiveGame",
    "GameChampions",
    "live_games",
    "get_event",
    "event_games",
    "game_champions",
    "find_game",
    "confirm",
    "attach_confirmation",
    "ddragon_champion_map",
]

# Public key for esports-api.lolesports.com. It ships for the lolesports.com
# web client (their build injects it as ESPORTS_API_KEY); the older widely
# copied `0TvQnueqKa5RJZkHzGYj9iAFGBmuwnch` no longer works. Verified with
# getLeagues/getSchedule/getEventDetails on 2026-10-19.
DEFAULT_API_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"

API_BASE = "https://esports-api.lolesports.com/persisted/gw"
FEED_BASE = "https://feed.lolesports.com/livestats/v1"
REQUEST_TIMEOUT = 5.0
REQUEST_INTERVAL = 1.0  # at most one request per second
WORLDS_LEAGUE_ID = "98767975604431411"

_last_request = 0.0


def api_headers() -> dict:
    return {"x-api-key": os.environ.get("LOLESPORTS_API_KEY") or DEFAULT_API_KEY}


def _throttle() -> None:
    global _last_request
    wait = _last_request + REQUEST_INTERVAL - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def _default_fetch(url: str, headers: Mapping[str, str]) -> dict:
    _throttle()
    request = urllib.request.Request(url, headers=dict(headers))
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def _request(fetch: Callable[[str, Mapping[str, str]], dict] | None,
             url: str, headers: Mapping[str, str]) -> dict:
    real_fetch = fetch or _default_fetch
    return real_fetch(url, dict(headers))


@dataclass
class LiveGame:
    match_id: str
    game_id: str
    game_number: int | None
    blue_code: str | None
    red_code: str | None
    blue_name: str | None
    red_name: str | None
    league: str | None = None


@dataclass
class GameChampions:
    """A game's window feed: picks per side in role order, bans if ever given."""

    game_id: str
    blue: list[str] = field(default_factory=list)
    red: list[str] = field(default_factory=list)
    bans: dict[str, list[str]] | None = field(default=None)


def live_games(fetch=None, *, league_id: str | None = None) -> list[LiveGame]:
    """In-progress (or up next) League games via ``getLive``."""
    url = f"{API_BASE}/getLive?hl=en-US"
    payload = _request(fetch, url, api_headers())
    events = payload.get("data", {}).get("schedule", {}).get("events") or []
    games: list[LiveGame] = []
    for event in events:
        if league_id and ((event.get("league") or {}).get("id")) != league_id:
            continue
        games.extend(_match_games(event.get("match") or {},
                                  league=(event.get("league") or {}).get("name")))
    return games


def get_event(match_id: str, fetch=None) -> dict:
    """Event details for a lolesports match id: {id, match: {...}, league}.

    Note the id comes from ``event["id"]`` (match id), not from inside
    ``match`` (which has none).
    """
    url = f"{API_BASE}/getEventDetails?hl=en-US&id={match_id}"
    payload = _request(fetch, url, api_headers())
    return payload.get("data", {}).get("event") or {}


def event_games(match: Mapping[str, Any]) -> list[dict]:
    """One entry per series game: number, id, side codes, state."""
    teams_by_id = {str(t.get("id")): t for t in match.get("teams") or []}
    games = []
    for game in match.get("games") or []:
        sides = {g.get("side"): str(g.get("id"))
                 for g in game.get("teams") or [] if g.get("side") in ("blue", "red")}
        blue = teams_by_id.get(sides.get("blue"))
        red = teams_by_id.get(sides.get("red"))
        games.append(
            {
                "number": game.get("number"),
                "id": str(game.get("id")),
                "state": game.get("state"),
                "blue_code": blue and blue.get("code"),
                "red_code": red and red.get("code"),
                "blue_name": blue and blue.get("name"),
                "red_name": red and red.get("name"),
            }
        )
    return games


def _match_games(match: Mapping[str, Any], *, league: str | None = None) -> list[LiveGame]:
    games = [g for g in match.get("games") or [] if g.get("gameState") in ("inProgress", "unstarted")]
    if not games:
        return []
    game = games[0]
    sides = {g.get("side"): g.get("id")
             for g in game.get("teams") or [] if g.get("side") in ("blue", "red")}
    by_id = {str(t.get("id")): t for t in match.get("teams") or []}
    blue = by_id.get(str(sides.get("blue")))
    red = by_id.get(str(sides.get("red")))
    return [
        LiveGame(
            match_id=str(match.get("id")),
            game_id=str(game.get("id")),
            game_number=game.get("number"),
            blue_code=blue and blue.get("code"),
            red_code=red and red.get("code"),
            blue_name=blue and blue.get("name"),
            red_name=red and red.get("name"),
            league=league,
        )
    ]


def ddragon_champion_map() -> dict[str, str]:
    """Data Dragon champion id -> Riot numeric key (as a string).

    E.g. MonkeyKing -> "62", KSante -> "897" (Wukong is not a Data Dragon id).
    """
    cache = getattr(ddragon_champion_map, "_cache", None)
    if cache is None:
        cache = champions_mod.id_to_key()
        ddragon_champion_map._cache = cache  # type: ignore[attr-defined]
    return cache


def _map_ids(champions: list, mapping: Mapping[str, str]) -> list[str]:
    return [str(mapping.get(str(c), str(c))) for c in champions]


def game_champions(game_id: str, *, now: datetime | None = None, fetch=None) -> GameChampions | None:
    """A game's champions from the livestats window, as Riot numeric keys.

    ``startingTime`` must be round-to-10s ISO in the past, and 204 (empty)
    means no frames past that point; a few past offsets are tried before
    giving up.  Feed bans: none today, so ``bans`` stays None.
    """
    base = now or datetime.now(timezone.utc)
    if isinstance(base, str):
        base = datetime.fromisoformat(base)
    if base.tzinfo is None:
        base = base.replace(tzinfo=timezone.utc)
    try:
        mapping = ddragon_champion_map()
    except FileNotFoundError:
        raise
    for offset in (90, 180, 600):
        at = (base - timedelta(seconds=offset)).replace(microsecond=0)
        at -= timedelta(seconds=at.second % 10)
        url = f"{FEED_BASE}/window/{game_id}?startingTime={at.strftime('%Y-%m-%dT%H:%M:%S')}.000Z"
        try:
            payload = _request(fetch, url, {})
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
            continue
        if not payload:
            continue
        metadata = payload.get("gameMetadata") or {}
        blue_meta = ((metadata.get("blueTeamMetadata") or {}).get("participantMetadata")) or []
        red_meta = ((metadata.get("redTeamMetadata") or {}).get("participantMetadata")) or []
        if not blue_meta and not red_meta:
            continue
        return GameChampions(
            game_id=str(payload.get("esportsGameId") or game_id),
            blue=_map_ids([p.get("championId") for p in blue_meta], mapping),
            red=_map_ids([p.get("championId") for p in red_meta], mapping),
        )
    return None


def _normalise(value: Any) -> str:
    """Lowercase, accent-free alphanumerics: ``"Karmine Corp"`` -> ``"karminecorp"``."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def _initials(value: Any) -> str:
    if value is None:
        return ""
    words = re.findall(r"[A-Za-z0-9]+", unicodedata.normalize("NFKD", str(value)))
    return "".join(_normalise(word)[:1] for word in words if _normalise(word))


def _team_matches(identifier: Any, team: Mapping[str, Any]) -> bool:
    """A tricode or name matches a lolesports team's code or name.

    Names compare case- and accent-insensitively; a name may equal the
    lolesports name, start with it (or be its start), or have initials equal
    to the team's code (``"Karmine Corp"`` -> ``"KC"``).
    """
    ident = _normalise(identifier)
    if not ident:
        return False
    name = _normalise(team.get("name"))
    code = _normalise(team.get("code"))
    if ident == code or ident == name:
        return True
    if name and (name.startswith(ident) or ident.startswith(name)):
        return True
    if code and _initials(identifier) == code:
        return True
    return _initials(team.get("name")) == ident


def _any_identifier_matches(identifiers: list[str], team: Mapping[str, Any]) -> bool:
    return any(_team_matches(identifier, team) for identifier in identifiers)


def _pair_matches(
    blue_ids: list[str],
    red_ids: list[str],
    blue_team: Mapping[str, Any],
    red_team: Mapping[str, Any],
) -> bool:
    return (
        _any_identifier_matches(blue_ids, blue_team) and _any_identifier_matches(red_ids, red_team)
    ) or (
        _any_identifier_matches(blue_ids, red_team) and _any_identifier_matches(red_ids, blue_team)
    )


def _event_matches(match: Mapping[str, Any], blue_ids: list[str], red_ids: list[str]) -> bool:
    teams = [team for team in match.get("teams") or [] if isinstance(team, Mapping)]
    for index, blue_team in enumerate(teams):
        for other, red_team in enumerate(teams):
            if index == other:
                continue
            if _pair_matches(blue_ids, red_ids, blue_team, red_team):
                return True
    return False


def find_game(
    blue_code: str | None = None,
    red_code: str | None = None,
    game_index: int | None = None,
    *,
    blue_name: str | None = None,
    red_name: str | None = None,
    fetch=None,
) -> str | None:
    """Match a session's teams (either side order) to a live game.

    Each side may be given as a broadcast tricode or an app team name; both
    are matched against ``match.teams[].code`` and ``.name`` (case- and
    accent-insensitive; a name may equal, start with, or give the initials of
    the lolesports team).  ``game_index`` is CaptureState.game_index (1-based
    series game number).  Falls back to the Worlds schedule for today (getLive
    is empty between games) via getEventDetails, then to the most recent
    completed game.
    """
    blue_ids = [str(value) for value in (blue_code, blue_name) if value]
    red_ids = [str(value) for value in (red_code, red_name) if value]
    if not blue_ids or not red_ids:
        return None
    for game in live_games(fetch=fetch, league_id=WORLDS_LEAGUE_ID):
        blue_team = {"code": game.blue_code, "name": game.blue_name}
        red_team = {"code": game.red_code, "name": game.red_name}
        if not _pair_matches(blue_ids, red_ids, blue_team, red_team):
            continue
        if game_index is not None and game.game_number not in (None, game_index):
            continue
        return game.game_id
    if game_index is None:
        return None
    url = f"{API_BASE}/getSchedule?hl=en-US&leagueId={WORLDS_LEAGUE_ID}"
    payload = _request(fetch, url, api_headers())
    events = payload.get("data", {}).get("schedule", {}).get("events") or []
    today = datetime.now(timezone.utc).date()
    fallback: str | None = None
    for event in events:
        match = event.get("match") or {}
        if not _event_matches(match, blue_ids, red_ids):
            continue
        try:
            start = datetime.fromisoformat(event["startTime"].replace("Z", "+00:00"))
        except (KeyError, ValueError):
            continue
        if start.date() != today:
            continue
        details = get_event(str(match.get("id")), fetch=fetch)
        inner_match = details.get("match") or {}
        for game in event_games(inner_match):
            blue_team = {"code": game["blue_code"], "name": game.get("blue_name")}
            red_team = {"code": game["red_code"], "name": game.get("red_name")}
            if not _pair_matches(blue_ids, red_ids, blue_team, red_team):
                continue
            if game["number"] is None or game["number"] == game_index:
                if game["state"] in ("completed", "inProgress"):
                    return game["id"]
                fallback = fallback or game["id"]
    return fallback


def _bucket(actions, action_type: str) -> dict[str, list[Mapping[str, Any]]]:
    buckets: dict[str, list[Mapping[str, Any]]] = {"blue": [], "red": []}
    for action in actions:
        side = (action.get("side") or "").lower()
        if side in buckets and action.get("action_type") == action_type \
                and action.get("champion_id"):
            buckets[side].append(action)
    return buckets


def confirm(capture_actions, champions: GameChampions | None, *, now=None) -> dict:
    """Compare captured pick actions (sets, per side) with the feed window.

    The feed is in role order, so picks are compared as a set per side.
    A captured side with fewer than five picks (or no picks at all) is
    "partial"; ``missing`` then lists the feed champions still unaccounted
    for.  Feed bans are compared per side as sets only if the feed has some
    (today it never does, so `bans` stays null).

    Returns ``{status: confirmed|mismatch|partial, picks, missing, bans,
    mismatches: [{sequence, captured, feed}], game_id, checked_at}``.
    """
    result: dict[str, Any] = {
        "status": "partial",
        "picks": {},
        "missing": {},
        "bans": None,
        "mismatches": [],
        "game_id": None,
        "checked_at": _iso(now),
    }
    if champions is None:
        result["status"] = "unavailable"
        result["message"] = "no window data for this game (yet)"
        return result
    result["game_id"] = champions.game_id
    buckets = _bucket(capture_actions, "pick")
    statuses = []
    for side in ("blue", "red"):
        feed_ids = set(champions.blue if side == "blue" else champions.red)
        actions = buckets[side]
        captured = [str(a.get("champion_id")) for a in actions]
        missing = sorted(feed_ids - set(captured))
        result["picks"][side] = sorted(feed_ids)
        result["missing"][side] = missing
        if set(captured) <= feed_ids and set(captured) == feed_ids:
            statuses.append("confirmed" if captured else "partial")
        elif not set(captured) <= feed_ids:
            statuses.append("mismatch")
            for action in actions:
                if str(action.get("champion_id")) not in feed_ids:
                    result["mismatches"].append({
                        "sequence": action.get("sequence"),
                        "captured": str(action.get("champion_id")),
                        "feed": missing,  # feed champions not seen otherwise
                    })
        else:
            statuses.append("partial")
    result["status"] = statuses[0] if len(set(statuses)) == 1 else (
        "mismatch" if "mismatch" in statuses else "partial"
    )
    if champions.bans:
        result["bans"] = confirm_bans(capture_actions, champions.bans)
    return result


def confirm_bans(capture_actions, feed_bans) -> dict[str, list[str]]:
    buckets = _bucket(capture_actions, "ban")
    out = {}
    for side in ("blue", "red"):
        feed_side = feed_bans.get(side, [])
        if not feed_side:  # incomplete feed ban list: nothing to confirm
            out[side] = []
            continue
        out[side] = sorted(set(str(a.get("champion_id")) for a in buckets.get(side, [])) & set(feed_side))
    return out


def _iso(now=None) -> str:
    stamp = now or datetime.now(timezone.utc)
    if isinstance(stamp, str):
        stamp = datetime.fromisoformat(stamp)
    if isinstance(stamp, (int, float)):
        stamp = datetime.fromtimestamp(stamp, tz=timezone.utc)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.isoformat(timespec="seconds")


_NETWORK_ERRORS = (urllib.error.HTTPError, urllib.error.URLError, OSError)


def attach_confirmation(state: Mapping[str, Any], *, fetch=None, now=None) -> dict:
    """Fill ``state["lolesports"]`` for a CaptureState dict (returns a copy).

    Finds the live game (sides may be swapped) from ``state["teams"]``
    tricodes and confirms the captured pick actions against the feed window.
    """
    out = dict(state)
    teams = out.get("teams") or {}
    blue_team = teams.get("blue") or {}
    red_team = teams.get("red") or {}
    blue_code = (blue_team.get("tricode") or "").upper() or None
    red_code = (red_team.get("tricode") or "").upper() or None
    blue_name = blue_team.get("name") or None
    red_name = red_team.get("name") or None
    if not (blue_code or blue_name) or not (red_code or red_name):
        out["lolesports"] = _unavailable("team names and tricodes missing from CaptureState")
        return out
    try:
        game_id = find_game(blue_code, red_code, out.get("game_index"),
                            blue_name=blue_name, red_name=red_name, fetch=fetch)
        champions = game_champions(game_id, now=now, fetch=fetch) if game_id else None
        result = confirm(out.get("actions") or [], champions, now=now)
        result["game_id"] = game_id
    except _NETWORK_ERRORS as exc:
        out["lolesports"] = _unavailable(f"lolesports feed error: {exc}")
        return out
    out["lolesports"] = result
    return out


def _unavailable(message: str) -> dict:
    return {
        "status": "unavailable",
        "message": message,
        "picks": {},
        "missing": {},
        "bans": None,
        "mismatches": [],
        "game_id": None,
        "checked_at": _iso(),
    }
