"""The performance work must be FASTER and still CORRECT: same results, same order, failures
isolated, shared limits respected, state not leaking between threads, and every optimisation
switchable off (=1) to the old sequential behaviour.

Timing tests use stubs that sleep for real (marked `real_time`) so overlap is observable; the
asserted ratios are deliberately conservative to stay stable on a loaded CI machine."""
from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from PIL import Image

from src.fetch.competitor_finder import CompetitorFinder
from src.generation import creative_forge as cf
from src.generation import image_providers as ip
from src.generation.virality_gate import ViralityGate
from src.shared import chat_http
from src.shared import llm_client as lc
from src.shared.chat_http import ChatHTTPError, HttpChatClient
from src.shared.concurrency import Outcome, parallel_map, slots
from src.shared.config import settings
from src.shared.llm_providers import classify

pytestmark = pytest.mark.real_time


class Gauge:
    """Counts how many callers are inside a section at once."""

    def __init__(self) -> None:
        self.now = 0
        self.peak = 0
        self._lock = threading.Lock()

    def __enter__(self):
        with self._lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        return self

    def __exit__(self, *exc):
        with self._lock:
            self.now -= 1


# ============================================================ primitives
def test_parallel_map_preserves_order_and_isolates_failures():
    def work(n):
        time.sleep(0.02 * (5 - n))  # later items finish FIRST: order must still follow the input
        if n == 2:
            raise ValueError("boom")
        return n * 10

    out = parallel_map(work, range(5), workers=5)
    assert [o.value for o in out if o.ok] == [0, 10, 30, 40]
    assert isinstance(out[2].error, ValueError) and not out[2].ok
    with pytest.raises(ValueError):
        out[2].unwrap()


def test_parallel_map_never_exceeds_the_worker_cap():
    gauge = Gauge()

    def work(_):
        with gauge:
            time.sleep(0.03)

    parallel_map(work, range(12), workers=3)
    assert gauge.peak == 3


def test_one_worker_is_the_old_sequential_loop_on_the_calling_thread():
    threads = set()
    parallel_map(lambda _: threads.add(threading.get_ident()), range(4), workers=1)
    assert threads == {threading.get_ident()}
    assert parallel_map(lambda x: x, [], workers=4) == []
    assert parallel_map(lambda x: x, [7], workers=4)[0].value == 7


def test_shared_slots_bound_work_across_independent_pools():
    gauge = Gauge()

    def work(_):
        with slots("unit-test-resource", 2), gauge:
            time.sleep(0.03)

    for _ in range(3):  # three separate "pipelines" competing for the same resource
        threading.Thread(target=lambda: parallel_map(work, range(4), workers=4)).start()
    time.sleep(0.5)
    assert gauge.peak <= 2


# ============================================================ virality gate scoring
class _Forge:
    def generate_variants(self, **kw):
        return [{"prompt_type": f"t{i}", "prompt_text": f"text {i}", "format": "reel"} for i in range(5)]

    def refine_with_hints(self, best, *a):
        return best


class _SlowPredictor:
    def __init__(self, latency=0.12, fail_on=None):
        self.latency, self.fail_on, self.gauge = latency, fail_on, Gauge()

    def score(self, text, *a, **k):
        with self.gauge:
            time.sleep(self.latency)
        base_text = text.partition("\n\nVisual deliverable:")[0]
        if base_text == self.fail_on:
            raise RuntimeError("predictor down")
        return {"virality_score": 70 + int(base_text[-1]), "virality_label": "M", "virality_probability": 0.5,
                "key_factors": [], "risks": [], "reasoning": "", "improvement_hints": []}


def _run_gate(conc, predictor):
    settings.llm_concurrency = conc
    gate = ViralityGate(threshold=60, max_attempts=1, forge=_Forge(), predictor=predictor)
    start = time.perf_counter()
    out = gate.run({}, {}, {}, ["t0", "t1", "t2", "t3", "t4"])
    return out, time.perf_counter() - start


def test_scoring_runs_concurrently_gives_identical_results_and_is_faster(monkeypatch):
    seq_pred, par_pred = _SlowPredictor(), _SlowPredictor()
    original = settings.llm_concurrency
    try:
        seq, t_seq = _run_gate(1, seq_pred)
        par, t_par = _run_gate(3, par_pred)
    finally:
        settings.llm_concurrency = original
    assert seq_pred.gauge.peak == 1 and par_pred.gauge.peak == 3  # the cap is honoured, and used
    assert [(p["prompt_text"], p["virality_score"]) for p in seq] == [(p["prompt_text"], p["virality_score"]) for p in par]
    assert t_par < t_seq * 0.7, f"sequential {t_seq:.2f}s vs concurrent {t_par:.2f}s"


def test_one_failed_score_does_not_disturb_the_others_under_concurrency():
    original = settings.llm_concurrency
    try:
        out, _ = _run_gate(3, _SlowPredictor(latency=0.02, fail_on="text 2"))
    finally:
        settings.llm_concurrency = original
    by_text = {p["prompt_text"].partition("\n\nVisual deliverable:")[0]: p for p in out}
    assert by_text["text 4"]["virality_score"] == 74 and by_text["text 3"]["virality_score"] == 73
    flagged = [p for p in out if p.get("warnings")]
    assert all(p["prompt_text"] == "text 2" for p in flagged)  # only the failed one is marked provisional


# ============================================================ shared LLM in-flight cap
def test_llm_calls_from_many_threads_respect_llm_concurrency(monkeypatch):
    gauge = Gauge()

    class Client:
        def __init__(self):
            self.chat = type("C", (), {"completions": type("X", (), {"create": staticmethod(self._create)})})

        @staticmethod
        def _create(model, messages, temperature, max_tokens=None):
            with gauge:
                time.sleep(0.04)
            return type("R", (), {"choices": [type("Ch", (), {"message": type("M", (), {"content": "ok"}), "finish_reason": "stop"})]})

    monkeypatch.setattr(settings, "groq_api_key", "k")
    monkeypatch.setattr(settings, "llm_provider_order", "groq")
    monkeypatch.setattr(settings, "llm_concurrency", 2)
    monkeypatch.setattr(lc, "make_client", lambda provider, timeout: Client())
    client = lc.LLMClient()
    outs = parallel_map(lambda i: client.run(f"p{i}", label="t"), range(8), workers=8)
    assert all(o.ok and o.value.stdout == "ok" for o in outs)
    assert gauge.peak == 2


# ============================================================ reel beats + per-prompt rendering
@pytest.fixture
def forge(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "logs_dir", tmp_path / "logs")
    monkeypatch.setattr(cf, "FONT_PATH", Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Poppins-Bold.ttf")
    f = cf.CreativeForge()
    monkeypatch.setattr(f, "_tts_beats", lambda beats, name: (_ for _ in ()).throw(cf.CreativeForgeError("no voice in test")))
    monkeypatch.setattr(f, "_assemble_video", lambda stills, durations, audio, out: Path(out).write_bytes(b"mp4"))
    monkeypatch.setattr(f, "_extract_thumbnail", lambda video, name: None)
    return f


def _fake_generate(latency, gauge=None, fail_for=()):
    import io

    def generate(prompt, width, height):
        if any(marker in prompt for marker in fail_for):
            return None
        with gauge or Gauge():
            time.sleep(latency)
        buf = io.BytesIO()
        Image.new("RGB", (width // 16, height // 16), (90, 140, 200)).save(buf, "PNG")
        return buf.getvalue(), "stub"

    return generate


REEL = {"format": "reel", "title": "Reel", "hook": "H", "reel_beats": ["one", "two", "three", "four", "five"]}


def test_reel_beats_render_concurrently_in_order_and_faster(forge, monkeypatch):
    results = {}
    for conc in (1, 3):
        gauge = Gauge()
        monkeypatch.setattr(settings, "image_concurrency", conc)
        monkeypatch.setattr(ip, "generate", _fake_generate(0.12, gauge))
        start = time.perf_counter()
        result = forge.render_prompt(dict(REEL), {"name": "Reel Biz"})
        results[conc] = (time.perf_counter() - start, gauge.peak, result)
    (t_seq, peak_seq, r_seq), (t_par, peak_par, r_par) = results[1], results[3]
    assert peak_seq == 1 and peak_par == 3
    assert r_seq["asset_type"] == r_par["asset_type"] == "reel"
    assert t_par < t_seq * 0.7, f"sequential {t_seq:.2f}s vs concurrent {t_par:.2f}s"


def test_a_failed_beat_is_skipped_and_the_rest_keep_their_order(forge, monkeypatch):
    stills = {}
    monkeypatch.setattr(settings, "image_concurrency", 3)
    monkeypatch.setattr(settings, "image_card_fallback", False)  # a failed beat has no fallback -> skipped
    monkeypatch.setattr(ip, "generate", _fake_generate(0.01, fail_for=("beat-two",)))
    monkeypatch.setattr(forge, "_assemble_video", lambda s, d, a, out: (stills.setdefault("s", list(s)), Path(out).write_bytes(b"mp4")))
    result = forge.render_prompt({**REEL, "reel_beats": ["beat-one", "beat-two", "beat-three"]}, {"name": "Order Biz"})
    assert result["asset_type"] == "reel" and len(stills["s"]) == 2
    assert [Path(s).name.split("-beat")[1][0] for s in stills["s"]] == ["0", "2"]  # beats 0 and 2, in order
    assert any("Beat 1 image skipped" in w for w in result["warnings"])


def test_warnings_and_brand_colour_do_not_leak_between_concurrent_renders(forge, monkeypatch):
    """Per-render state used to be instance attributes: two renders on two threads would share them."""
    monkeypatch.setattr(settings, "image_concurrency", 3)
    monkeypatch.setattr(ip, "generate", _fake_generate(0.05, fail_for=("FAILME",)))  # this prompt falls back to a card
    prompts = [
        {"format": "single image", "title": "ok one", "hook": "fine hook"},
        {"format": "single image", "title": "bad", "hook": "FAILME"},
        {"format": "single image", "title": "ok two", "hook": "fine hook"},
    ]
    biz = {"name": "Colour Bakery"}
    outs = parallel_map(lambda job: forge.render_prompt(job[1], biz, index=job[0]), list(enumerate(prompts)), workers=3)
    warnings = [o.unwrap()["warnings"] for o in outs]
    assert warnings[0] == [] and warnings[2] == []  # healthy renders carry NO warning from the failing one
    assert any("branded card" in w for w in warnings[1])


# ============================================================ competitors
class _Social:
    def __init__(self, latency, fail=()):
        self.latency, self.fail, self.gauge = latency, fail, Gauge()

    async def fetch_instagram(self, handle):
        with self.gauge:
            await asyncio.sleep(self.latency)
        if handle in self.fail:
            raise RuntimeError("actor failed")
        return {"posts": [{"caption": f"post by {handle}"}]}

    async def fetch_linkedin(self, handle):
        return await self.fetch_instagram(handle)


def _finder(social):
    finder = CompetitorFinder.__new__(CompetitorFinder)
    finder.social = social
    return finder


def test_competitor_scrapes_overlap_within_the_cap_keep_order_and_isolate_failures(monkeypatch):
    comps = lambda: [{"name": f"c{i}", "platform": "instagram", "handle": f"h{i}"} for i in range(6)]  # noqa: E731
    timings = {}
    for conc in (1, 3):
        social = _Social(0.1, fail={"h2"})
        monkeypatch.setattr(settings, "fetch_concurrency", conc)
        start = time.perf_counter()
        out = asyncio.run(_finder(social).fetch_competitor_posts(comps()))
        timings[conc] = (time.perf_counter() - start, social.gauge.peak, out)
    (t1, peak1, out1), (t3, peak3, out3) = timings[1], timings[3]
    assert peak1 == 1 and peak3 == 3
    assert [c["handle"] for c in out3] == [f"h{i}" for i in range(6)]  # order preserved
    assert out3[2]["top_posts"] == [] and out3[3]["top_posts"][0]["caption"] == "post by h3"  # only h2 failed
    assert [c["top_posts"] for c in out1] == [c["top_posts"] for c in out3]  # same answer either way
    assert t3 < t1 * 0.7, f"sequential {t1:.2f}s vs concurrent {t3:.2f}s"


# ============================================================ the thin HTTP chat client
_REAL_CLIENT = httpx.Client  # captured once: a second patch must not wrap the first


def _client_with(handler, monkeypatch):
    real = _REAL_CLIENT
    built = []

    def factory(**kw):
        built.append(kw)
        return real(transport=httpx.MockTransport(handler), **kw)

    chat_http.reset_clients()
    monkeypatch.setattr(chat_http.httpx, "Client", factory)
    return built


def _ok(content="hi", finish="stop"):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}]})


def test_chat_client_sends_the_openai_compatible_request(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(url=str(request.url), auth=request.headers["authorization"], extra=request.headers.get("x-title"),
                    body=__import__("json").loads(request.content))
        return _ok("hello", "length")

    _client_with(handler, monkeypatch)
    client = HttpChatClient("https://api.example/v1/", "KEY", 30, {"X-Title": "ai-ads"})
    reply = client.chat.completions.create("m1", [{"role": "user", "content": "p"}], 0.4, max_tokens=123)
    assert seen["url"] == "https://api.example/v1/chat/completions" and seen["auth"] == "Bearer KEY" and seen["extra"] == "ai-ads"
    assert seen["body"] == {"model": "m1", "messages": [{"role": "user", "content": "p"}], "temperature": 0.4, "max_tokens": 123}
    assert reply.choices[0].message.content == "hello" and reply.choices[0].finish_reason == "length"


def test_max_tokens_is_omitted_when_not_set(monkeypatch):
    bodies = []
    _client_with(lambda r: (bodies.append(__import__("json").loads(r.content)), _ok())[1], monkeypatch)
    HttpChatClient("https://a/v1", "k", 30).chat.completions.create("m", [], 0.1)
    assert "max_tokens" not in bodies[0]


def test_connections_are_pooled_across_calls_and_threads(monkeypatch):
    built = _client_with(lambda r: _ok(), monkeypatch)
    client = HttpChatClient("https://a/v1", "k", 30)
    parallel_map(lambda _: client.chat.completions.create("m", [], 0.1), range(10), workers=5)
    HttpChatClient("https://b/v1", "k2", 30).chat.completions.create("m", [], 0.1)
    assert len(built) == 1  # one pooled httpx client (was: a new SDK client + TLS handshake per call)


def test_http_errors_carry_status_headers_and_body_for_classification(monkeypatch):
    _client_with(lambda r: httpx.Response(429, headers={"retry-after": "7"}, text='{"error": "Rate limit: tokens per day exceeded"}'), monkeypatch)
    with pytest.raises(ChatHTTPError) as err:
        HttpChatClient("https://a/v1", "k", 30).chat.completions.create("m", [], 0.1)
    failure = classify(err.value)
    assert err.value.status_code == 429 and "tokens per day" in str(err.value)
    assert (failure.kind, failure.scope, failure.cooldown) == ("rate_limit", "model", 7.0)  # Retry-After honoured


@pytest.mark.parametrize("status,body,kind", [
    (404, '{"error": {"code": "model_not_found"}}', "model"),
    (401, "bad key", "auth"),
    (402, '{"error": "You have no remaining credits"}', "quota"),
    (503, "overloaded", "transient"),
    (410, "gone", "model"),
])
def test_error_replies_classify_exactly_as_before(monkeypatch, status, body, kind):
    _client_with(lambda r: httpx.Response(status, text=body), monkeypatch)
    with pytest.raises(ChatHTTPError) as err:
        HttpChatClient("https://a/v1", "k", 30).chat.completions.create("m", [], 0.1)
    assert classify(err.value).kind == kind


def test_non_api_replies_and_malformed_bodies_are_reported_not_crashed(monkeypatch):
    _client_with(lambda r: httpx.Response(200, headers={"content-type": "text/plain"}, text="OK\r\n"), monkeypatch)
    assert HttpChatClient("https://a/v1", "k", 30).chat.completions.create("m", [], 0.1) == "OK\r\n"  # llm_client diagnoses this
    _client_with(lambda r: httpx.Response(200, json={"unexpected": True}), monkeypatch)
    with pytest.raises(ValueError, match="malformed chat completion"):
        HttpChatClient("https://a/v1", "k", 30).chat.completions.create("m", [], 0.1)


def test_network_timeouts_surface_as_transient(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    _client_with(handler, monkeypatch)
    with pytest.raises(httpx.ReadTimeout) as err:
        HttpChatClient("https://a/v1", "k", 30).chat.completions.create("m", [], 0.1)
    assert classify(err.value).kind == "transient"


def test_the_openai_sdk_is_no_longer_a_dependency_or_import():
    import tomllib

    root = Path(__file__).resolve().parents[1]
    deps = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    assert not any(d.lower().startswith("openai") for d in deps)
    assert "openai" not in (root / "requirements.txt").read_text(encoding="utf-8").lower()
    for module in (root / "src").rglob("*.py"):
        text = module.read_text(encoding="utf-8")
        assert "from openai" not in text and "import openai" not in text, module
    _ = patch  # keep the import used when the file is edited
