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
    pipe._attach_posting_brief({"timezone": "Asia/Kolkata"}, prompts, {})
    assert prompts[0]["caption"].startswith("Stop guessing.")
    assert "#growth" in prompts[0]["caption"]
    assert "recommended_post_time" in prompts[0]
