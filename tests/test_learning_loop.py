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
