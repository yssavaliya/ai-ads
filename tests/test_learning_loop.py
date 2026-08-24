from unittest.mock import MagicMock, patch

from src.learning.auto_learner import MIN_SIMILARITY, AutoLearner, caption_similarity
from src.learning.process_feedback import IG_POST_RE, parse_feedback_issues


def test_min_similarity_raised():
    assert 0.1 <= MIN_SIMILARITY <= 0.5


def test_containment_matches_trimmed_caption():
    """A user who posts a shortened/paraphrased version of the suggested
    caption must still match via the containment fallback."""
    suggested = "AI meets art in Ahmedabad watch Aimagix Studio turn ideas into viral magic"
    posted = "Aimagix Studio turn ideas into viral magic"
    assert caption_similarity(posted, suggested) >= MIN_SIMILARITY


def test_decide_outcome_absolute_bar_for_new_account():
    from src.learning.auto_learner import AutoLearner

    # Too little history: a post with any real reaction is good.
    assert AutoLearner._decide_outcome({"likes": 3, "comments": 0}, 100, 0.5, 1) == "good"
    assert AutoLearner._decide_outcome({"likes": 0, "comments": 2}, 100, 0.5, 1) == "good"
    # Total flop is bad even for a new account.
    assert AutoLearner._decide_outcome({"likes": 0, "comments": 0}, 100, 0.5, 1) == "bad"


def test_decide_outcome_uses_median_benchmark_with_enough_history():
    from src.learning.auto_learner import AutoLearner

    # Enough history: compare against the business's own median rate.
    post = {"likes": 10, "comments": 0}
    assert AutoLearner._decide_outcome(post, 1000, 0.02, 5) == "bad"   # rate 0.01 < 0.02
    assert AutoLearner._decide_outcome(post, 1000, 0.005, 5) == "good"  # rate 0.01 >= 0.005


def test_ig_post_regex_extracts_reel_and_post_urls():
    body = "here it is https://www.instagram.com/reel/Cxyz123/ thanks"
    assert IG_POST_RE.search(body).group(0) == "https://www.instagram.com/reel/Cxyz123/"
    body2 = "posted: https://www.instagram.com/p/AbCdEf/"
    assert IG_POST_RE.search(body2).group(0) == "https://www.instagram.com/p/AbCdEf/"


def test_explicit_link_bypasses_similarity(tmp_path):
    """A post_id matching a linked Engagement row matches even with zero
    caption overlap."""
    import os

    os.environ["DATABASE_PATH"] = str(tmp_path / "t.db")
    from src.storage.models import (
        Base,
        Engagement,
        Platform,
        Prompt,
        PromptType,
        SessionLocal,
        engine as db_engine,
    )

    Base.metadata.create_all(bind=db_engine)
    db = SessionLocal()
    prompt = Prompt(
        business_id=1,
        prompt_text="totally different words here",
        prompt_type=PromptType.EDUCATIONAL,
    )
    db.add(prompt)
    db.commit()
    db.add(Engagement(prompt_id=prompt.id, post_id="MATCHED1", platform=Platform.INSTAGRAM))
    db.commit()

    learner = AutoLearner()
    learner._db = db
    prompts = [
        {
            "id": prompt.id,
            "prompt_type": "educational",
            "prompt_text": "totally different words here",
            "sent_at": None,
            "audience_segment": None,
        }
    ]
    posts = [{"post_id": "MATCHED1", "caption": "no overlap at all", "likes": 10}]
    matched = learner._match(posts, prompts)
    assert len(matched) == 1
    assert matched[0][3]["post_id"] == "MATCHED1"
    db.close()


def test_parse_feedback_issue_includes_posted(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.json.return_value = [
        {
            "number": 7,
            "title": "[content-intel-feedback] prompt #12 [POSTED 📎]",
            "body": "**Prompt ID:** 12\nhttps://www.instagram.com/p/QWErtz123/",
        }
    ]
    fake_resp.raise_for_status = lambda: None

    monkeypatch.setattr("src.shared.config.settings.github_token", "tok")
    monkeypatch.setattr("src.shared.config.settings.github_repo", "o/r")
    with patch("httpx.get", return_value=fake_resp):
        issues = parse_feedback_issues()
    assert len(issues) == 1
    assert issues[0]["feedback"] == "posted"
    assert issues[0]["prompt_id"] == 12


def test_auto_learn_does_not_double_count_same_post(tmp_path, monkeypatch):
    """Regression: engagement.yml runs hourly, so the same post is re-scraped
    repeatedly. The Bayesian weights must update only on the FIRST sighting,
    or trials (and thus the weights) inflate every run."""
    import os
    from datetime import datetime, timedelta

    os.environ["DATABASE_PATH"] = str(tmp_path / "t.db")
    from src.shared.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    from src.learning.auto_learner import AutoLearner
    from src.learning.bayesian_updater import BayesianUpdater
    from src.storage.models import (
        Base,
        Business,
        Prompt,
        PromptType,
        SessionLocal,
        engine as db_engine,
    )

    Base.metadata.create_all(bind=db_engine)
    db = SessionLocal()
    try:
        biz = Business(name="InflateBiz", email="x@y.z", virality_threshold=75)
        db.add(biz)
        db.commit()
        db.refresh(biz)
        prompt = Prompt(
            business_id=biz.id,
            prompt_text="unique inflate phrase alpha beta gamma delta",
            prompt_type=PromptType.EDUCATIONAL,
            sent_at=datetime.utcnow() - timedelta(days=1),
        )
        db.add(prompt)
        db.commit()
        db.refresh(prompt)

        social = {
            "handle": "inflatebiz",
            "follower_count": 1000,
            "posts": [
                {
                    "post_id": "INFLATE1",
                    "caption": "unique inflate phrase alpha beta gamma delta",
                    "likes": 5,
                    "comments": 1,
                    "shares": 0,
                    "saves": 0,
                    "reach": 0,
                    "posted_at": (datetime.utcnow() - timedelta(hours=20)).isoformat(),
                }
            ],
        }
        learner = AutoLearner()
        learner.run_for_business(biz.id, social, db)
        learner.run_for_business(biz.id, social, db)  # re-scrape same post

        edu = BayesianUpdater().get_weights(biz.id)["topics"]["educational"]
        assert edu["trials"] == 1, edu
        assert edu["hits"] == 1, edu

        # Durable log must now actually be written (it was silently failing
        # before the settings import fix) and must NOT duplicate per re-scrape.
        from pathlib import Path as _Path

        log = _Path(tmp_path) / "engagement_log.jsonl"
        assert log.exists(), "engagement_log.jsonl was not written"
        inflate_lines = [l for l in log.read_text().splitlines() if "INFLATE1" in l]
        assert len(inflate_lines) == 1, inflate_lines
    finally:
        db.close()


def test_human_posted_does_not_double_count_on_rescrape(tmp_path, monkeypatch):
    """Regression: when a human confirms 'I posted this' (process_feedback
    stores post_id == URL shortcode and records a GOOD weight), the same post
    re-scraped from Instagram later carries Apify's internal id (different from
    the shortcode). The learner must still recognise it as the SAME post via the
    URL shortcode, so it (a) does NOT record the weight a second time and
    (b) does NOT create a duplicate Engagement row."""
    import os
    from datetime import datetime, timedelta

    os.environ["DATABASE_PATH"] = str(tmp_path / "t.db")
    from src.shared.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    from src.learning.auto_learner import AutoLearner
    from src.learning.bayesian_updater import BayesianUpdater
    from src.storage.models import (
        Base,
        Business,
        Engagement,
        Platform,
        Prompt,
        PromptType,
        SessionLocal,
        engine as db_engine,
    )

    Base.metadata.create_all(bind=db_engine)
    db = SessionLocal()
    try:
        biz = Business(name="PostedBiz", email="p@q.r", virality_threshold=75)
        db.add(biz)
        db.commit()
        db.refresh(biz)
        prompt = Prompt(
            business_id=biz.id,
            prompt_text="human confirmed posted phrase zeta eta theta",
            prompt_type=PromptType.EDUCATIONAL,
            sent_at=datetime.utcnow() - timedelta(days=1),
        )
        db.add(prompt)
        db.commit()
        db.refresh(prompt)

        # Simulate process_feedback handling a "I posted this" issue.
        short = "SHORTXYZ"
        db.add(
            Engagement(
                prompt_id=prompt.id,
                platform=Platform.INSTAGRAM,
                post_id=short,
                post_url=f"https://instagram.com/p/{short}",
                fetched_at=datetime.utcnow(),
            )
        )
        db.commit()
        BayesianUpdater().record(biz.id, "good", prompt_type="educational")

        # Later scrape: same post, but post_id is Apify's internal id while the
        # URL still points at the same shortcode.
        social = {
            "handle": "postedbiz",
            "follower_count": 1000,
            "posts": [
                {
                    "post_id": "apify_internal_999",
                    "url": f"https://instagram.com/p/{short}",
                    "caption": "human confirmed posted phrase zeta eta theta",
                    "likes": 40,
                    "comments": 3,
                    "shares": 1,
                    "saves": 2,
                    "reach": 500,
                    "posted_at": (datetime.utcnow() - timedelta(hours=20)).isoformat(),
                }
            ],
        }
        AutoLearner().run_for_business(biz.id, social, db)

        edu = BayesianUpdater().get_weights(biz.id)["topics"]["educational"]
        assert edu["trials"] == 1, edu  # only the human GOOD, not a second count
        assert edu["hits"] == 1, edu

        rows = db.query(Engagement).filter(Engagement.prompt_id == prompt.id).all()
        assert len(rows) == 1, [r.post_id for r in rows]  # no duplicate row
        # The single row was refreshed with real scraped engagement.
        assert rows[0].likes == 40
    finally:
        db.close()
