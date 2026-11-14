"""Per-call max_tokens sizing: real Groq log showed 413s because 8192 was reserved for every call."""
from src.shared import llm_client as lc
from src.shared.config import settings


def test_small_label_cap_beats_the_global_default():
    assert lc.completion_budget("groq", "openai/gpt-oss-120b", "short prompt", "virality-predictor") == 1500


def test_unknown_label_is_bounded_by_the_providers_tpm_budget():
    budget = lc.completion_budget("groq", "openai/gpt-oss-120b", "x" * 3000, None)
    assert budget < settings.llm_max_tokens
    assert budget + lc.estimate_tokens("x" * 3000) <= 8000


def test_provider_without_a_tpm_limit_keeps_the_label_cap():
    assert lc.completion_budget("gemini", "g", "x" * 30000, "prompt-forge") == 7000


def test_model_output_cap_applies_and_too_small_budgets_skip_the_model():
    # qwen on Groq allows 1000 output tokens/minute: usable for a small answer, not for a forge answer
    assert lc.completion_budget("groq", "qwen/qwen3.8-27b", "p", "virality-predictor") == 1000
    assert lc.completion_budget("groq", "x", "y" * 26000, "prompt-forge") == 0  # prompt alone eats the TPM budget


def test_budget_is_used_in_the_request(monkeypatch):
    seen = {}

    class Msg:
        content = '{"ok": true}'

    class Choice:
        message = Msg()
        finish_reason = "stop"

    class Resp:
        choices = [Choice()]

    class Client:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kw):
                    seen.update(kw)
                    return Resp()

    monkeypatch.setattr(lc, "make_client", lambda *_a, **_k: Client())
    monkeypatch.setattr(settings, "groq_api_key", "k")
    monkeypatch.setattr(settings, "llm_provider_order", "groq")
    lc.HEALTH.reset() if hasattr(lc.HEALTH, "reset") else None
    result = lc.LLMClient().run("hello", label="virality-predictor")
    assert result.provider == "groq"
    assert seen["max_tokens"] == 1500


def _fake_client(content, finish_reason, seen):
    class Msg:
        pass

    Msg.content = content

    class Choice:
        message = Msg()

    Choice.finish_reason = finish_reason

    class Resp:
        choices = [Choice()]

    class Client:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kw):
                    seen.append(kw)
                    return Resp()

    return Client()


def test_gpt_oss_requests_ask_for_low_reasoning_effort_other_models_do_not(monkeypatch):
    seen: list[dict] = []
    monkeypatch.setattr(lc, "make_client", lambda *_a, **_k: _fake_client('{"ok": 1}', "stop", seen))
    monkeypatch.setattr(settings, "groq_api_key", "k")
    monkeypatch.setattr(settings, "llm_provider_order", "groq")
    monkeypatch.setattr(settings, "groq_model", "openai/gpt-oss-120b")
    monkeypatch.setattr(settings, "groq_fallback_models", "other/model")
    lc.LLMClient().run("hello", label="virality-predictor")
    assert seen[0]["reasoning_effort"] == "low"
    seen.clear()
    monkeypatch.setattr(settings, "groq_model", "other/model")
    lc.LLMClient().run("hello", label="virality-predictor")
    assert "reasoning_effort" not in seen[0]
    seen.clear()
    monkeypatch.setattr(settings, "groq_model", "openai/gpt-oss-120b")
    monkeypatch.setattr(settings, "groq_reasoning_effort", "")
    lc.LLMClient().run("hello", label="virality-predictor")
    assert "reasoning_effort" not in seen[0]


def test_empty_answer_with_length_finish_is_diagnosed_as_exhausted_reasoning(monkeypatch, caplog):
    seen: list[dict] = []
    monkeypatch.setattr(lc, "make_client", lambda *_a, **_k: _fake_client("", "length", seen))
    monkeypatch.setattr(settings, "groq_api_key", "k")
    monkeypatch.setattr(settings, "llm_provider_order", "groq")
    with caplog.at_level("ERROR"):
        try:
            lc.LLMClient().run("hello", label="virality-predictor")
        except lc.AllProvidersFailedError:
            pass
    assert "used all 1500 tokens" in caplog.text


def test_prompt_forge_asks_for_fewer_variants_when_the_token_budget_is_small(monkeypatch):
    """Real run: 5 variants need ~5k answer tokens but Groq's 8000 TPM leaves ~3k after the large prompt."""
    from unittest.mock import patch

    from src.generation import prompt_forge as pf

    monkeypatch.setattr(settings, "groq_api_key", "k")
    monkeypatch.setattr(settings, "llm_provider_order", "groq")
    monkeypatch.setattr(settings, "groq_model", "openai/gpt-oss-120b")
    forge = pf.PromptForge()
    seen = {}

    def fake_run_json(prompt, label=None, **kw):
        seen["prompt"] = prompt
        return [{"prompt_text": f"brief {i}", "title": f"t{i}"} for i in range(5)]

    with patch.object(forge.llm, "run_json", fake_run_json):
        variants = forge.generate_variants({"name": "B"}, {}, {}, ["educational"], count=5)
    budget = forge.llm.primary_budget(seen["prompt"], "prompt-forge")
    assert budget is not None
    assert len(variants) <= max(pf.MIN_VARIANTS_PER_CALL, budget // pf.TOKENS_PER_VARIANT)
    assert "Generate 5 distinct" not in seen["prompt"]  # the prompt itself was rebuilt for the smaller count


def test_prompt_forge_keeps_the_requested_count_when_the_budget_is_unbounded(monkeypatch):
    from unittest.mock import patch

    from src.generation import prompt_forge as pf

    monkeypatch.setattr(settings, "gemini_api_key", "k")
    monkeypatch.setattr(settings, "groq_api_key", "")
    monkeypatch.setattr(settings, "llm_provider_order", "gemini")
    forge = pf.PromptForge()
    seen = {}

    def fake_run_json(prompt, label=None, **kw):
        seen["prompt"] = prompt
        return [{"prompt_text": f"brief {i}"} for i in range(5)]

    with patch.object(forge.llm, "run_json", fake_run_json):
        assert len(forge.generate_variants({"name": "B"}, {}, {}, ["educational"], count=5)) == 5
    assert "Generate 5 distinct" in seen["prompt"]
