"""Tests for the LLM transcript interpreter (fake HTTP)."""
from __future__ import annotations

import json

from lol_live_capture.interpret import Event, Interpreter


def _envelope(content: str) -> str:
    return json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": content}}]}
    )


class Recorder:
    def __init__(self, body, exc: Exception | None = None) -> None:
        self.body = body
        self.exc = exc
        self.calls: list[dict] = []

    def __call__(self, url, payload, headers, timeout):  # noqa: ANN001
        self.calls.append(
            {"url": url, "payload": payload, "headers": headers, "timeout": timeout}
        )
        if self.exc is not None:
            raise self.exc
        return self.body


def test_disabled_without_key() -> None:
    rec = Recorder(_envelope('{"events": []}'))
    interp = Interpreter(api_key="", http=rec)
    assert interp.enabled is False
    assert interp.interpret("they ban Wukong", {}) is None
    assert rec.calls == []


def test_good_json_maps_names_to_ids() -> None:
    content = json.dumps(
        {
            "events": [
                {
                    "champion": "Wukong",
                    "team": "blue",
                    "action": "ban",
                    "confidence": 0.9,
                },
                {
                    "champion": "K'Sante",
                    "team": "red",
                    "action": "pick",
                    "confidence": 0.7,
                },
            ]
        }
    )
    rec = Recorder(_envelope(content))
    interp = Interpreter(api_key="k", http=rec)
    events = interp.interpret("ban Wukong, pick K'Sante", {"teams": {"blue": "KC"}})
    assert events == [
        Event("62", "blue", "ban", 0.9),
        Event("897", "red", "pick", 0.7),
    ]
    assert rec.calls[0]["payload"]["max_tokens"] == 400
    assert rec.calls[0]["payload"]["response_format"] == {"type": "json_object"}


def test_unknown_champion_dropped() -> None:
    content = json.dumps(
        {
            "events": [
                {"champion": "Notachampion", "action": "pick"},
                {"champion": "Olaf", "action": "ban"},
            ]
        }
    )
    interp = Interpreter(api_key="k", http=Recorder(_envelope(content)))
    events = interp.interpret("x", {})
    assert [e.champion_id for e in events] == ["2"]


def test_team_and_action_normalised() -> None:
    content = json.dumps(
        {
            "events": [
                {
                    "champion": "Annie",
                    "team": "purple",
                    "action": "delete",
                    "confidence": 5,
                },
            ]
        }
    )
    interp = Interpreter(api_key="k", http=Recorder(_envelope(content)))
    (event,) = interp.interpret("x", {})
    assert event.team is None
    assert event.action == "speculative"
    assert event.confidence == 1.0


def test_junk_body_returns_none() -> None:
    interp = Interpreter(api_key="k", http=Recorder("definitely not json"))
    assert interp.interpret("x", {}) is None


def test_junk_content_returns_none() -> None:
    interp = Interpreter(api_key="k", http=Recorder(_envelope("<<not json>>")))
    assert interp.interpret("x", {}) is None


def test_timeout_returns_none() -> None:
    interp = Interpreter(api_key="k", http=Recorder(None, exc=TimeoutError("slow")))
    assert interp.interpret("x", {}) is None


def test_headers_and_timeout() -> None:
    rec = Recorder(_envelope('{"events": []}'))
    interp = Interpreter(api_key="k", http=rec, timeout=6.0)
    interp.interpret("x", {"session": "abc", "game_index": 2})
    headers = rec.calls[0]["headers"]
    assert headers["Authorization"] == "Bearer k"
    assert headers["User-Agent"] == "lol-live-capture/0.1"
    assert rec.calls[0]["timeout"] == 6.0


def test_endpoint_model_and_extra_body_come_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("LIVE_LLM_BASE_URL", "https://llm.example/v1/chat/completions")
    monkeypatch.setenv("LIVE_LLM_MODEL", "my-model")
    monkeypatch.setenv("LIVE_LLM_EXTRA_BODY", '{"thinking": {"type": "disabled"}}')
    rec = Recorder(_envelope('{"events": []}'))
    Interpreter(api_key="k", http=rec).interpret("x", {})
    call = rec.calls[0]
    assert call["url"] == "https://llm.example/v1/chat/completions"
    assert call["payload"]["model"] == "my-model"
    assert call["payload"]["thinking"] == {"type": "disabled"}
