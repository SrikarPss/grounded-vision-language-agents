"""Robust parser that extracts a structured Action from an LLM rationale.

The preferred format is a JSON block emitted after a ``Action:`` tag, e.g.::

    Thought: The login button is in the top-right corner.
    Action: {"type": "click", "target": "#login-button"}

If JSON parsing fails, we fall back to a permissive regex over common
natural-language patterns (``click <something>``, ``type "hello"``, etc.) so
that a single mis-escaped quote doesn't nuke an otherwise reasonable rollout.
Parsing failures are reported (not silently swallowed) so error analysis can
categorize them as ``action_parsing_failure``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .schemas import Action, ActionType

_ACTION_BLOCK_RE = re.compile(
    r"(?:Action|ACTION)\s*:\s*(?P<body>.+?)(?:\n\s*\n|\Z)", re.DOTALL
)

_NL_PATTERNS: list[tuple[re.Pattern[str], ActionType]] = [
    (re.compile(r"\bclick(?:\s+on)?\s+(?P<target>.+?)(?:[.\n]|$)", re.IGNORECASE), ActionType.CLICK),
    (re.compile(r"\btype\s+[\"'](?P<value>.+?)[\"']\s+(?:in|into)\s+(?P<target>.+?)(?:[.\n]|$)", re.IGNORECASE), ActionType.TYPE),
    (re.compile(r"\bselect\s+[\"'](?P<value>.+?)[\"']\s+(?:from|in)\s+(?P<target>.+?)(?:[.\n]|$)", re.IGNORECASE), ActionType.SELECT),
    (re.compile(r"\bscroll\s+(?P<value>up|down)(?:[.\n]|$)", re.IGNORECASE), ActionType.SCROLL),
    (re.compile(r"\b(?:final\s+)?answer\s*[:\-]\s*(?P<value>.+?)(?:[.\n]|$)", re.IGNORECASE), ActionType.ANSWER),
    (re.compile(r"\bstop\b", re.IGNORECASE), ActionType.STOP),
]


@dataclass
class ParseResult:
    action: Optional[Action]
    rationale: str
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.action is not None


def _split_rationale(text: str) -> tuple[str, str]:
    """Split a model response into (rationale, action_body).

    We look for the last ``Action:`` block so that models which re-state the
    prompt template don't confuse us.
    """
    matches = list(_ACTION_BLOCK_RE.finditer(text))
    if not matches:
        return text.strip(), ""
    m = matches[-1]
    rationale = text[: m.start()].strip()
    body = m.group("body").strip()
    return rationale, body


def _try_json(body: str) -> Optional[Action]:
    # Allow both raw JSON and JSON wrapped in a ```json ... ``` fence.
    body = body.strip()
    fence = re.match(r"```(?:json)?\s*(?P<inner>.+?)\s*```", body, re.DOTALL)
    if fence:
        body = fence.group("inner")
    # Take the first balanced {...} substring in case the model appended chatter.
    brace = re.search(r"\{.*\}", body, re.DOTALL)
    if not brace:
        return None
    try:
        payload = json.loads(brace.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "type" not in payload:
        return None
    try:
        return Action(**payload)
    except Exception:
        return None


def _try_nl(body: str) -> Optional[Action]:
    for pattern, atype in _NL_PATTERNS:
        m = pattern.search(body)
        if not m:
            continue
        groups = m.groupdict()
        target = groups.get("target")
        value = groups.get("value")
        # For SCROLL the direction is captured as `value`; leave target empty.
        return Action(
            type=atype,
            target=target.strip().strip("'\"") if target else None,
            value=value.strip() if value else None,
        )
    return None


def parse(text: str) -> ParseResult:
    """Parse a raw LLM response into an Action plus rationale."""
    if not text or not text.strip():
        return ParseResult(action=None, rationale="", error="empty response")

    rationale, body = _split_rationale(text)

    # Strategy 1: JSON block (preferred, unambiguous).
    action = _try_json(body) if body else None

    # Strategy 2: natural-language patterns.
    if action is None:
        search_space = body if body else text
        action = _try_nl(search_space)

    if action is None:
        return ParseResult(
            action=None,
            rationale=rationale or text.strip(),
            error="no parseable action found",
        )

    # Attach the rationale back onto the action for logging / error analysis.
    if not action.rationale:
        action.rationale = rationale
    return ParseResult(action=action, rationale=rationale)
