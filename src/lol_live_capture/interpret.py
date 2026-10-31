"""LLM client that turns caster speech into draft events.

Any OpenAI-compatible chat-completions endpoint works: set ``LIVE_LLM_API_KEY`` and, for a provider other than
OpenAI, ``LIVE_LLM_BASE_URL`` (the full ``.../chat/completions`` URL) and ``LIVE_LLM_MODEL``.
``LIVE_LLM_EXTRA_BODY`` (a JSON object) is merged into the request for provider-specific options.
The client is tiny on purpose: the
HTTP call is injected so tests never need the network or the API key.  With no
``LIVE_LLM_API_KEY`` the interpreter reports itself disabled and every
:meth:`Interpreter.interpret` returns ``None``; the caller then falls back to
the rule-based :mod:`spotter`.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable
from urllib import request as urllib_request

from . import spotter

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 6.0
MAX_TOKENS = 400

_VALID_ACTIONS = {"ban", "pick", "hover", "speculative"}

# http(url, payload, headers, timeout) -> response body (str/bytes) or raises.
HttpCallable = Callable[[str, dict[str, Any], dict[str, str], float], Any]


@dataclass(frozen=True)
class Event:
    champion_id: str | None
    team: str | None
    action: str
    confidence: float


def _build_prompt(
    transcript_tail: str, context: dict[str, Any] | None
) -> list[dict[str, str]]:
    context = context or {}
    teams = context.get("teams") or {}
    lines = [
        "You read League of Legends draft commentary and extract the champion "
        "the casters are naming, and whether it is a ban, pick or hover.",
        'Return ONLY JSON: {"events": [{"champion": str, '
        '"team": "blue"|"red"|null, "action": '
        '"ban"|"pick"|"hover"|"speculative", "confidence": 0..1}]}.',
        "Drop speculative chatter about champions that are not being acted on.",
        f"Teams: {json.dumps(teams)}",
        f"Acting side: {context.get('acting_side')}",
        f"Current turn: {context.get('turn')}",
        f"Champions already used: {context.get('used')}",
        f"Fearless locks: {context.get('fearless')}",
        "",
        "Transcript tail:",
        transcript_tail.strip(),
    ]
    return [
        {"role": "system", "content": "You extract structured draft events as JSON."},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _extra_body() -> dict[str, Any]:
    raw = os.environ.get("LIVE_LLM_EXTRA_BODY")
    if not raw:
        return {}
    try:
        extra = json.loads(raw)
    except ValueError:
        log.warning("LIVE_LLM_EXTRA_BODY is not valid JSON; ignored")
        return {}
    return extra if isinstance(extra, dict) else {}


class Interpreter:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        http: HttpCallable | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.api_key = (
            api_key if api_key is not None else os.environ.get("LIVE_LLM_API_KEY")
        )
        self.base_url = base_url or os.environ.get("LIVE_LLM_BASE_URL") or DEFAULT_BASE_URL
        self.model = model or os.environ.get("LIVE_LLM_MODEL") or DEFAULT_MODEL
        self.http = http
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def interpret(
        self,
        transcript_tail: str,
        context: dict[str, Any] | None = None,
    ) -> list[Event] | None:
        """Return the parsed events, or ``None`` on any failure or timeout."""
        if not self.enabled:
            return None
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "lol-live-capture/0.1",
        }
        payload = {
            "model": self.model,
            "messages": _build_prompt(transcript_tail, context),
            "response_format": {"type": "json_object"},
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
        }
        payload.update(_extra_body())
        call = self.http or self._default_http
        try:
            body = call(self.base_url, payload, headers, self.timeout)
        except Exception as exc:  # noqa: BLE001 - timeout/network/JSON all -> None
            log.warning("interpreter call failed: %s", exc)
            return None
        return self._parse(body)

    @staticmethod
    def _default_http(
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        timeout: float,
    ) -> str:
        data = json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(url, data=data, headers=headers, method="POST")
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8")

    def _parse(self, body: Any) -> list[Event] | None:
        if isinstance(body, bytes):
            body = body.decode("utf-8", "replace")
        if not isinstance(body, str):
            return None
        try:
            envelope = json.loads(body)
            content = envelope["choices"][0]["message"]["content"]
            if isinstance(content, dict):
                data = content
            else:
                data = json.loads(content)
        except (KeyError, IndexError, TypeError, ValueError):
            return None
        raw_events = data.get("events") if isinstance(data, dict) else None
        if not isinstance(raw_events, list):
            return None
        out: list[Event] = []
        for raw in raw_events:
            if not isinstance(raw, dict):
                continue
            cid = _champion_id(raw.get("champion"))
            if cid is None:
                continue
            action = str(raw.get("action", "neutral")).lower()
            if action not in _VALID_ACTIONS:
                action = "speculative"
            team = raw.get("team")
            team = team if team in ("blue", "red") else None
            try:
                confidence = float(raw.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            confidence = min(1.0, max(0.0, confidence))
            out.append(Event(cid, team, action, confidence))
        return out


def _champion_id(name: Any) -> str | None:
    if not isinstance(name, str) or not name.strip():
        return None
    hit = spotter.lookup(name)
    return hit[0] if hit else None
