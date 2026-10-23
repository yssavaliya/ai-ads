"""The free-tier fallback chain: limits, outages, bad auth, retired models and
unusable output must all fall through to the next free option."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.shared import llm_client as lc
from src.shared import llm_providers as lp
from src.shared.config import Settings, settings
from src.shared.llm_client import AllProvidersFailedError, LLMClient


class ApiError(Exception):
    def __init__(self, status, message="boom", headers=None):
        super().__init__(message)
        self.status_code = status
        self.response = SimpleNamespace(status_code=status, headers=headers or {})


class FakeNetwork:
    """Scripted stand-in for every provider. script[(host_fragment, model)] is a list of
    outcomes consumed in order: a string (answer), an Exception (raised) or None (empty)."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[tuple[str, str]] = []
        self.max_tokens: list[int | None] = []

    def factory(self, **kwargs):
        net = self
        base = kwargs["base_url"]

        class Completions:
            @staticmethod
            def create(model, messages, temperature, max_tokens=None):
                net.calls.append((base, model))
                net.max_tokens.append(max_tokens)
                for (frag, m), outcomes in net.script.items():
                    if frag in base and m == model:
                        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
                        if isinstance(outcome, Exception):
                            raise outcome
                        if isinstance(outcome, tuple):  # (content, finish_reason)
                            content, finish = outcome
                        else:
                            content, finish = outcome, "stop"
                        if content == "<<RAW-STRING-BODY>>":
                            return "<html>not the API</html>"  # what the SDK returns for a non-JSON reply
                        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)])
                raise ApiError(404, "model_not_found")

        return SimpleNamespace(chat=SimpleNamespace(completions=Completions))

    def hosts(self):
        return [c[0].split("//")[1].split(".")[0] for c in self.calls]


@pytest.fixture
def chain(monkeypatch):
    """Groq (2 models) -> Gemini (1 model) -> Cerebras, all configured."""
    monkeypatch.setattr(settings, "groq_api_key", "g")
    monkeypatch.setattr(settings, "groq_model", "groq-big")
    monkeypatch.setattr(settings, "groq_fallback_models", "groq-small")
    monkeypatch.setattr(settings, "gemini_api_key", "m")
    monkeypatch.setattr(settings, "gemini_model", "gem-flash")
    monkeypatch.setattr(settings, "gemini_fallback_models", "")
    monkeypatch.setattr(settings, "cerebras_api_key", "c")
    monkeypatch.setattr(settings, "cerebras_model", "cb-70b")
    monkeypatch.setattr(settings, "llm_provider_order", "groq,gemini,cerebras")
    monkeypatch.setattr(lc.time, "sleep", lambda s: None)

    def install(script):
        net = FakeNetwork(script)
        patcher = patch("src.shared.llm_client.make_client", side_effect=lambda provider, timeout: net.factory(base_url=provider.base_url))
        patcher.start()
        monkeypatch.setattr(lc, "LLMClient", lc.LLMClient)
        net._patcher = patcher
        return net

    yield install
    patch.stopall()


def test_rate_limited_primary_falls_through_to_sibling_model_then_provider(chain):
    net = chain({
        ("groq", "groq-big"): [ApiError(429, "rate limit reached")],
        ("groq", "groq-small"): [ApiError(429, "rate limit reached")],
        ("generativelanguage", "gem-flash"): ["from gemini"],
    })
    result = LLMClient().run("hi", label="t")
    assert result.stdout == "from gemini" and result.provider == "gemini"
    assert net.hosts() == ["api", "api", "generativelanguage"]  # both groq models tried first


def test_sibling_model_of_same_provider_is_used_before_leaving_it(chain):
    chain({
        ("groq", "groq-big"): [ApiError(429, "slow down")],
        ("groq", "groq-small"): ["small model answer"],
    })
    result = LLMClient().run("hi")
    assert (result.provider, result.model) == ("groq", "groq-small")


def test_rate_limited_models_are_skipped_on_the_next_call(chain):
    net = chain({
        ("groq", "groq-big"): [ApiError(429, "rate limit")],
        ("groq", "groq-small"): [ApiError(429, "rate limit")],
        ("generativelanguage", "gem-flash"): ["ok"],
    })
    client = LLMClient()
    client.run("one")
    net.calls.clear()
    client.run("two")
    assert net.hosts() == ["generativelanguage"]  # circuit breaker: no wasted groq calls


def test_bad_key_disables_the_whole_provider_not_just_one_model(chain):
    net = chain({
        ("groq", "groq-big"): [ApiError(401, "invalid api key")],
        ("groq", "groq-small"): ["never reached"],
        ("generativelanguage", "gem-flash"): ["gemini ok"],
    })
    result = LLMClient().run("hi")
    assert result.provider == "gemini"
    assert ("https://api.groq.com/openai/v1", "groq-small") not in net.calls


def test_retired_model_only_disables_that_model(chain):
    chain({
        ("groq", "groq-big"): [ApiError(404, "The model `groq-big` does not exist")],
        ("groq", "groq-small"): ["small ok"],
    })
    assert LLMClient().run("hi").model == "groq-small"


def test_empty_answers_fall_through(chain):
    chain({
        ("groq", "groq-big"): [""],
        ("groq", "groq-small"): ["   "],
        ("generativelanguage", "gem-flash"): ["real answer"],
    })
    assert LLMClient().run("hi").provider == "gemini"


def test_five_hundred_errors_move_on_after_bounded_retries(chain):
    net = chain({
        ("groq", "groq-big"): [ApiError(503, "unavailable")],
        ("groq", "groq-small"): [ApiError(503, "unavailable")],
        ("generativelanguage", "gem-flash"): [ApiError(503, "unavailable")],
        ("cerebras", "cb-70b"): ["cerebras saved the day"],
    })
    result = LLMClient().run("hi", max_retries=2)
    assert result.provider == "cerebras"
    assert net.calls.count(("https://api.groq.com/openai/v1", "groq-big")) == 2  # retried, then gave up


def test_everything_down_raises_with_the_chain_in_the_message(chain):
    chain({
        ("groq", "groq-big"): [ApiError(401)],
        ("generativelanguage", "gem-flash"): [ApiError(401)],
        ("cerebras", "cb-70b"): [ApiError(401)],
    })
    with pytest.raises(AllProvidersFailedError, match="groq.*gemini.*cerebras"):
        LLMClient().run("hi", label="unit")


def test_waits_once_when_everything_is_only_rate_limited(chain, monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(lp.HEALTH, "_clock", lambda: now["t"])
    sleeps: list[float] = []

    def fake_sleep(seconds):
        sleeps.append(seconds)
        now["t"] += seconds

    monkeypatch.setattr(lc.time, "sleep", fake_sleep)
    monkeypatch.setattr(settings, "llm_provider_order", "groq")
    chain({
        ("groq", "groq-big"): [ApiError(429, "rate limit", {"retry-after": "7"}), "after the wait"],
        ("groq", "groq-small"): [ApiError(429, "rate limit", {"retry-after": "9"}), "after the wait"],
    })
    assert LLMClient().run("hi").stdout == "after the wait"
    assert sleeps and 6 < sleeps[0] < 9  # waited for the soonest retry-after, not forever


def test_long_cooldowns_do_not_block_the_run(chain, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(lc.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(settings, "llm_provider_order", "groq")
    chain({
        ("groq", "groq-big"): [ApiError(429, "daily quota exhausted (tokens per day)")],
        ("groq", "groq-small"): [ApiError(429, "daily quota exhausted (tokens per day)")],
    })
    with pytest.raises(AllProvidersFailedError):
        LLMClient().run("hi")
    assert slept == []  # a 15-minute cooldown is not worth sleeping on


# ---------------------------------------------------------------- run_json
def test_invalid_json_from_one_provider_falls_through_to_the_next(chain):
    net = chain({
        ("groq", "groq-big"): ["Sure! Here is some prose, no JSON."],
        ("generativelanguage", "gem-flash"): ['```json\n{"ok": true}\n```'],
    })
    assert LLMClient().run_json("give json", label="t") == {"ok": True}
    assert net.hosts()[0] == "api" and "generativelanguage" in net.hosts()


def test_wrong_json_shape_also_falls_through(chain):
    chain({
        ("groq", "groq-big"): ["[1, 2, 3]"],
        ("generativelanguage", "gem-flash"): [json.dumps({"a": 1})],
    })
    assert LLMClient().run_json("x", expect=(dict,)) == {"a": 1}


def test_all_providers_returning_garbage_raises_the_json_error(chain):
    chain({
        ("groq", "groq-big"): ["nope"],
        ("generativelanguage", "gem-flash"): ["still nope"],
        ("cerebras", "cb-70b"): ["nothing"],
    })
    with pytest.raises(ValueError):
        LLMClient().run_json("x")


# ---------------------------------------------------------- catalog / config
def test_only_configured_providers_take_part_in_order(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider_order", "openrouter,groq,mistral")
    monkeypatch.setattr(settings, "groq_api_key", "g")
    monkeypatch.setattr(settings, "openrouter_api_key", "o")
    monkeypatch.setattr(settings, "mistral_api_key", None)
    assert [p.name for p in lp.configured_providers()] == ["openrouter", "groq"]
    openrouter = lp.get_provider("openrouter")
    assert openrouter.models[0].endswith(":free") and "HTTP-Referer" in openrouter.headers


def test_keyless_fallback_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(settings, "llm_keyless_fallback", True)
    assert lp.get_provider("pollinations") is not None
    monkeypatch.setattr(settings, "llm_keyless_fallback", False)
    assert lp.get_provider("pollinations") is None


def test_github_models_use_the_built_in_actions_token(monkeypatch):
    monkeypatch.setattr(settings, "github_token", "ghs_x")
    monkeypatch.setattr(settings, "github_models_enabled", True)
    provider = lp.get_provider("github")
    assert provider.base_url == "https://models.github.ai/inference" and provider.api_key == "ghs_x"
    monkeypatch.setattr(settings, "github_models_enabled", False)
    assert lp.get_provider("github") is None


def test_model_lists_merge_without_duplicates_or_blanks():
    assert lp.csv("a, b", "b,,c", None) == ["a", "b", "c"]


@pytest.mark.parametrize("status,msg,kind,scope", [
    (401, "bad key", "auth", "provider"),
    (403, "forbidden", "auth", "provider"),
    (404, "no such thing", "model", "model"),
    (429, "slow down", "rate_limit", "model"),
    (500, "oops", "transient", "model"),
    (503, "model is overloaded", "transient", "model"),
    (402, "You have no remaining credits", "quota", "provider"),
    (400, "context length", "bad_request", "none"),
    (None, "Connection error.", "transient", "provider"),
])
def test_error_classification(status, msg, kind, scope):
    exc = ApiError(status, msg) if status else Exception(msg)
    failure = lp.classify(exc)
    assert (failure.kind, failure.scope) == (kind, scope)


def test_daily_quota_cooldown_is_long_and_retry_after_is_honoured():
    assert lp.classify(ApiError(429, "tokens per day limit")).cooldown == 900.0
    assert lp.classify(ApiError(429, "rate limit", {"retry-after": "12"})).cooldown == 12.0
    assert lp.classify(ApiError(429, "rate limit")).cooldown == 60.0


# ------------------------------------------------- failures seen in the real run
def test_overloaded_model_does_not_take_its_sibling_models_down(chain, monkeypatch):
    """Real log: gemini-3.6-flash answered 503 "high demand" and the provider-wide
    cooldown skipped gemini-2.5-flash, which was fine."""
    monkeypatch.setattr(settings, "gemini_fallback_models", "gem-lite")
    monkeypatch.setattr(settings, "llm_provider_order", "gemini")
    chain({
        ("generativelanguage", "gem-flash"): [ApiError(503, "This model is currently experiencing high demand")],
        ("generativelanguage", "gem-lite"): ["the lite model is fine"],
    })
    result = LLMClient().run("hi", max_retries=1)
    assert (result.provider, result.model) == ("gemini", "gem-lite")


def test_exhausted_prepaid_credits_disable_the_provider_for_the_hour(chain, monkeypatch):
    """Real log: Hugging Face answered 402 "no remaining credits" on every call."""
    monkeypatch.setattr(settings, "hf_token", "t")
    monkeypatch.setattr(settings, "llm_provider_order", "huggingface,groq")
    net = chain({
        ("huggingface", settings.hf_llm_model): [ApiError(402, "You have no remaining credits")],
        ("groq", "groq-big"): ["groq ok"],
    })
    client = LLMClient()
    assert client.run("one").provider == "groq"
    net.calls.clear()
    client.run("two")
    assert not any("huggingface" in c[0] for c in net.calls)


def test_non_json_reply_from_an_endpoint_is_diagnosed_and_skipped(chain, monkeypatch):
    """Real log: GitHub Models gave "'str' object has no attribute 'choices'"."""
    monkeypatch.setattr(settings, "github_token", "ghs")
    monkeypatch.setattr(settings, "llm_provider_order", "github,groq")
    chain({
        ("models.github.ai", settings.github_models_model): [("<<RAW-STRING-BODY>>", "stop")],
        ("groq", "groq-big"): ["groq ok"],
    })
    assert LLMClient().run("hi").provider == "groq"
    assert lp.classify(ValueError("non-JSON response from x: '<html>'")).kind == "bad_response"


def test_github_models_send_the_headers_the_api_expects(monkeypatch):
    monkeypatch.setattr(settings, "github_token", "ghs")
    monkeypatch.setattr(settings, "github_models_enabled", True)
    provider = lp.get_provider("github")
    assert provider.headers["X-GitHub-Api-Version"] == "2022-11-28"
    assert "json" in provider.headers["Accept"]


def test_github_models_are_opt_in(monkeypatch):
    """From Actions the endpoint answered a plain 'OK', so it is off unless enabled."""
    monkeypatch.setattr(settings, "github_token", "ghs")
    assert Settings().github_models_enabled is False
    monkeypatch.setattr(settings, "github_models_enabled", False)
    assert lp.get_provider("github") is None


def test_every_call_sets_max_tokens_so_long_json_is_not_cut_off(chain):
    """Real log: Groq's 8k-char prompt-forge answer ended mid-string ("Unterminated string")."""
    net = chain({("groq", "groq-big"): ["ok"]})
    LLMClient().run("hi")
    assert net.max_tokens == [settings.llm_max_tokens] and settings.llm_max_tokens >= 4096


def test_truncated_json_strikes_off_only_that_model_not_the_whole_provider(chain):
    """Real log: one truncated Groq answer excluded Groq entirely; its sibling model was never asked."""
    net = chain({
        ("groq", "groq-big"): [('{"a": "cut off mid str', "length")],
        ("groq", "groq-small"): ['{"a": "complete"}'],
        ("generativelanguage", "gem-flash"): ['{"a": "gemini"}'],
    })
    assert LLMClient().run_json("give json") == {"a": "complete"}
    assert [c[1] for c in net.calls] == ["groq-big", "groq-small"]  # gemini was not even needed


def test_unconfigured_providers_are_skipped_silently(chain, caplog):
    chain({("groq", "groq-big"): ["ok"]})
    import logging

    with caplog.at_level(logging.INFO):
        LLMClient().run("hi")
    assert "mistral" not in caplog.text and "openrouter" not in caplog.text


def test_reasoning_blocks_are_stripped_from_answers(chain):
    chain({("groq", "groq-big"): ['<think>let me {reason} about it</think>\n{"score": 90}']})
    result = LLMClient().run("hi")
    assert result.stdout == '{"score": 90}'
    assert LLMClient().run_json("hi") == {"score": 90}


def test_default_fallback_models_are_ones_the_real_keys_offer():
    """From the Drift Check's list_models output: llama-3.x on Groq and gemini-2.x were
    404/retired for this account; gpt-oss-20b / qwen3.8 / gemini-3.x are offered."""
    s = Settings()
    assert "llama-3.3-70b-versatile" not in s.groq_fallback_models and "gpt-oss-20b" in s.groq_fallback_models
    assert "gemini-2.0-flash" not in s.gemini_fallback_models and "gemini-3" in s.gemini_fallback_models


def test_short_retry_after_is_waited_out_on_the_same_model(chain, monkeypatch):
    """Real log: Groq said "try again in 1.4s"; hopping to weaker models/providers was worse."""
    sleeps: list[float] = []
    monkeypatch.setattr(lc.time, "sleep", lambda s: sleeps.append(s))
    net = chain({("groq", "groq-big"): [ApiError(429, "rate limit", {"retry-after": "1.4"}), "answer from the best model"]})
    result = LLMClient().run("hi")
    assert (result.provider, result.model, result.stdout) == ("groq", "groq-big", "answer from the best model")
    assert sleeps and 1.3 < sleeps[0] < 2.0
    assert net.calls == [("https://api.groq.com/openai/v1", "groq-big")] * 2  # no hop to groq-small / gemini


def test_long_retry_after_is_not_waited_out_inline(chain, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(lc.time, "sleep", lambda s: sleeps.append(s))
    chain({
        ("groq", "groq-big"): [ApiError(429, "rate limit", {"retry-after": "45"})],
        ("groq", "groq-small"): ["small"],
    })
    assert LLMClient().run("hi").model == "groq-small" and sleeps == []


def test_gone_models_are_permanently_skipped():
    assert lp.classify(ApiError(410, "Gone")).kind == "model"
