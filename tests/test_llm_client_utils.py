import json as _json

import pytest as _pytest

from src.shared.llm_client import extract_json_text, strip_markdown_fences


def test_strip_markdown_fences():
    assert strip_markdown_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert strip_markdown_fences('plain text') == 'plain text'


def test_extract_json_text_exact():
    assert extract_json_text('{"a": 1}') == '{"a": 1}'


def test_extract_json_text_from_fenced_block():
    assert extract_json_text('```json\n{"a": 1}\n```') == '{"a": 1}'


def test_extract_json_text_from_surrounding_text():
    text = 'Here is the result:\n```json\n{"virality_score": 85}\n```\nThanks!'
    assert extract_json_text(text) == '{"virality_score": 85}'



@_pytest.mark.parametrize("text,expected", [
    ('{"a": 1}', {"a": 1}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('Sure! {"a": {"b": [1, 2]}} hope it helps', {"a": {"b": [1, 2]}}),
    ('{"a": 1}\n\nNote: use {curly} braces in captions', {"a": 1}),  # trailing prose with braces
    ('Let me think {about this}. Final: {"score": 85, "hints": ["x"]}', {"score": 85, "hints": ["x"]}),
    ('Result: [{"x": 1}, {"x": 2}] done', [{"x": 1}, {"x": 2}]),
    ('{"outer": {"inner": 1}} and {"tiny": 2}', {"outer": {"inner": 1}}),  # the largest document wins
])
def test_extract_json_from_messy_model_answers(text, expected):
    """Real run: Groq/Gemini answers for the virality predictor did not parse with the
    old first-brace/last-brace heuristic, so every variant got a neutral score of 50."""
    assert _json.loads(extract_json_text(text)) == expected


def test_extract_json_returns_garbage_unchanged_so_the_caller_fails_honestly():
    assert extract_json_text("no json here, sorry") == "no json here, sorry"
    with _pytest.raises(ValueError):
        _json.loads(extract_json_text("no json here, sorry"))
