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