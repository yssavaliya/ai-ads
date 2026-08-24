from src.scheduler.daily_job import DailyPipeline


def test_recommend_post_time_returns_string():
    pipe = DailyPipeline()
    out = pipe._recommend_post_time({"timezone": "Asia/Kolkata"}, {})
    assert isinstance(out, str)
    assert "at" in out
    assert "Kolkata" in out


def test_recommend_post_time_uses_data_driven_hour():
    pipe = DailyPipeline()
    from datetime import datetime, timedelta

    social = {
        "posts": [
            {"posted_at": (datetime.utcnow() - timedelta(days=i)).replace(hour=8).isoformat()}
            for i in range(5)
        ]
    }
    out = pipe._recommend_post_time({"timezone": "Asia/Kolkata"}, social)
    assert "08:" in out or "8:" in out


def test_attach_posting_brief_sets_caption_and_time():
    pipe = DailyPipeline()
    prompts = [
        {"hook": "Stop guessing.", "hashtags": ["#growth", "#ai"], "prompt_type": "educational"},
    ]
    pipe._attach_posting_brief({"timezone": "Asia/Kolkata", "instagram_handle": "aimagix"}, prompts, {})
    assert prompts[0]["caption"].startswith("Stop guessing.")
    assert "#growth" in prompts[0]["caption"]
    assert "recommended_post_time" in prompts[0]
    assert "first_comment" in prompts[0]
    assert "@aimagix" in prompts[0]["first_comment"]


def test_recommend_post_time_uses_audience_signal():
    pipe = DailyPipeline()
    out = pipe._recommend_post_time(
        {"timezone": "Asia/Kolkata"},
        {},
        {"personas": [{"best_posting_times": "Tue-Thu 6-9 PM IST"}]},
    )
    assert "09:00 PM" in out


def test_recommend_post_time_falls_back_without_audience():
    pipe = DailyPipeline()
    out = pipe._recommend_post_time({"timezone": "Asia/Kolkata"}, {})
    assert "at" in out


def test_dedupe_drops_near_duplicates():
    pipe = DailyPipeline()
    prompts = [
        {"prompt_text": "5 home workouts for busy professionals", "virality_score": 80},
        {"prompt_text": "car maintenance tips for beginners", "virality_score": 70},
    ]
    recent = ["5 home workouts for busy indian professionals #fitness"]
    kept = pipe._dedupe_prompts(prompts, recent)
    assert len(kept) == 1
    assert kept[0]["prompt_text"].startswith("car maintenance")


def test_dedupe_never_returns_empty():
    pipe = DailyPipeline()
    prompts = [{"prompt_text": "5 home workouts for busy professionals", "virality_score": 80}]
    recent = ["5 home workouts for busy indian professionals #fitness"]
    kept = pipe._dedupe_prompts(prompts, recent)
    assert len(kept) == 1


def test_weight_boost_sign():
    from src.generation.virality_gate import ViralityGate

    assert ViralityGate._weight_boost(None, {"format": "reel"}) == 0.0
    assert ViralityGate._weight_boost({"formats::reel": 1.5}, {"format": "reel"}) > 0
    assert ViralityGate._weight_boost({"formats::reel": 0.2}, {"format": "reel"}) < 0


def test_read_engagement_log_parses(tmp_path, monkeypatch):
    from src.shared.config import settings
    from src.scheduler.daily_job import DailyPipeline

    log = tmp_path / "engagement_log.jsonl"
    log.write_text(
        '{"prompt_id": 1, "outcome": "good", "rate": 0.05}\n'
        '{"prompt_id": 2, "outcome": "bad", "rate": 0.001}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    records = DailyPipeline._read_engagement_log()
    assert len(records) == 2
    assert records[0]["outcome"] == "good"


def test_read_engagement_log_missing_returns_empty(tmp_path, monkeypatch):
    from src.shared.config import settings
    from src.scheduler.daily_job import DailyPipeline

    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    assert DailyPipeline._read_engagement_log() == []
