"""Tests for quality checks in compute_metrics.py."""

import pytest

from datasets.data_aug_pranav.compute_metrics import (
    QualityResult,
    check_conversation,
    check_leakage_llm,
)


def _make_convo(turns: list[dict], **extra) -> dict:
    """Helper: build a minimal conversation dict."""
    return {"turns": turns, **extra}


def _user(turn_num: int, utterance: str, **extra) -> dict:
    return {"turn_num": turn_num, "speaker": "user", "utterance": utterance, **extra}


def _agent(turn_num: int, utterance: str) -> dict:
    return {"turn_num": turn_num, "speaker": "agent", "utterance": utterance}


# ── Regex-based checks (check_conversation) ─────────────────────────

# 1. Clean conversation passes
def test_clean_conversation_passes():
    convo = _make_convo([
        _user(1, "Help me with my post"),
        _agent(2, "Sure, what do you need?"),
        _user(3, "Fix the intro"),
    ])
    result = check_conversation(convo, "same_flow")
    assert result.passed
    assert result.reasons == []


# 2. Filler in user turn -> rejected
def test_filler_in_user_turn():
    convo = _make_convo([
        _user(1, "I was wondering if you could help"),
        _agent(2, "Sure."),
        _user(3, "Fix it"),
    ])
    result = check_conversation(convo, "same_flow")
    assert not result.passed
    assert any("filler" in r for r in result.reasons)


# 3. Overack in agent turn -> rejected
def test_overack_in_agent_turn():
    convo = _make_convo([
        _user(1, "Help me"),
        _agent(2, "Absolutely! Let me help."),
        _user(3, "Fix it"),
    ])
    result = check_conversation(convo, "same_flow")
    assert not result.passed
    assert any("overack" in r for r in result.reasons)


# 4. Em-dash in any turn -> rejected
def test_emdash_rejected():
    convo = _make_convo([
        _user(1, "Help me"),
        _agent(2, "Here\u2014let me check"),
        _user(3, "Fix it"),
    ])
    result = check_conversation(convo, "same_flow")
    assert not result.passed
    assert any("em dash" in r for r in result.reasons)


# 5. Turn 3 with 15 words -> rejected
def test_turn3_too_long():
    convo = _make_convo([
        _user(1, "Help me"),
        _agent(2, "Sure."),
        _user(3, "I would like you to please go ahead and fix the introduction paragraph of my blog post now"),
    ])
    result = check_conversation(convo, "same_flow")
    assert not result.passed
    assert any("turn 3 too long" in r for r in result.reasons)


# 6. Turn 3 with 5 words -> passes
def test_turn3_short_passes():
    convo = _make_convo([
        _user(1, "Help me"),
        _agent(2, "Sure."),
        _user(3, "Fix the intro please"),
    ])
    result = check_conversation(convo, "same_flow")
    assert result.passed


# 7. ambiguous_second without connector -> rejected
def test_ambiguous_second_missing_connector():
    convo = _make_convo([
        _user(1, "Help me"),
        _agent(2, "Sure."),
        _user(3, "Fix it"),
    ])
    result = check_conversation(convo, "ambiguous_second")
    assert not result.passed
    assert any("multi-request connector" in r for r in result.reasons)


# 8. ambiguous_second with "and then" -> passes
def test_ambiguous_second_with_connector():
    convo = _make_convo([
        _user(1, "Help me"),
        _agent(2, "Sure."),
        _user(3, "Fix it and then publish"),
    ])
    result = check_conversation(convo, "ambiguous_second")
    assert result.passed


# 9. same_flow without connector -> passes (not required)
def test_same_flow_no_connector_ok():
    convo = _make_convo([
        _user(1, "Help me"),
        _agent(2, "Sure."),
        _user(3, "Fix it"),
    ])
    result = check_conversation(convo, "same_flow")
    assert result.passed


# 10. Multiple violations -> all reasons returned
def test_multiple_violations():
    convo = _make_convo([
        _user(1, "I was wondering about this"),
        _agent(2, "Absolutely! I\u2019ll help."),
        _user(3, "Fix it"),
    ])
    result = check_conversation(convo, "same_flow")
    assert not result.passed
    # filler + overack + unicode (right single quote)
    assert len(result.reasons) >= 3


# 11. QualityResult.passed is True when reasons is empty
def test_quality_result_passed_when_empty():
    qr = QualityResult(passed=True, reasons=[])
    assert qr.passed is True
    assert qr.reasons == []

    qr2 = QualityResult(passed=False, reasons=["something wrong"])
    assert qr2.passed is False


# ── LLM leakage judge (check_leakage_llm) ───────────────────────────
# These tests call the actual Anthropic API.

_SCENARIO = {"scenario": "User wants to retrieve and compare beginner baking posts"}
_ASSIGNED_FLOWS = {
    "turn1_flow": "retrieve",
    "turn1_intent": "Research",
    "turn1_tools": ["list_posts"],
    "turn3_flow": "retrieve",
    "turn3_intent": "Research",
    "turn3_tools": ["list_posts"],
}


@pytest.mark.asyncio
async def test_leakage_detected_agent_infers_goal():
    """Agent reveals knowledge from scenario ('compare wording') not stated by user."""
    convo = _make_convo([
        _user(1, "Can you pull my beginner baking posts from the last 6 months?"),
        _agent(2, "Yep, I'll list your recent beginner-friendly baking drafts and published posts so we can compare wording."),
        _user(3, "Just the ones tagged beginner or 101, and include anything that mentions laminated dough."),
    ], category="same_flow")
    result = await check_leakage_llm(convo, _SCENARIO, _ASSIGNED_FLOWS)
    assert not result.passed
    assert any("leakage" in r for r in result.reasons)


@pytest.mark.asyncio
async def test_clean_conversation_no_leakage():
    """Agent responds only to what user said — no leaked metadata."""
    convo = _make_convo([
        _user(1, "beginner baking posts from the last 6 months"),
        _agent(2, "Are beginner posts those that contain the word '101' or those that have the tag 'beginner'?"),
        _user(3, "beginner tag is good, also anything that mentions laminated dough"),
    ], category="same_flow")
    result = await check_leakage_llm(convo, _SCENARIO, _ASSIGNED_FLOWS)
    assert result.passed


@pytest.mark.asyncio
async def test_leakage_agent_anticipates_next_turn():
    """Agent anticipates what user will ask next based on scenario metadata."""
    convo = _make_convo([
        _user(1, "show me posts from last month"),
        _agent(2, "Here are your recent posts. I noticed some could use a tone revision if you'd like to update them."),
        _user(3, "yeah fix the tone on the first one"),
    ], category="switch_flow")
    result = await check_leakage_llm(
        convo,
        {"scenario": "User retrieves posts then revises tone on one"},
        {"turn1_flow": "browse", "turn1_intent": "Research",
         "turn3_flow": "revise_tone", "turn3_intent": "Revise"},
    )
    assert not result.passed
    assert any("leakage" in r for r in result.reasons)
