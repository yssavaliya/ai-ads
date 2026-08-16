from datetime import datetime, timedelta

from src.learning.auto_learner import AutoLearner, engagement_rate, jaccard_similarity, tokenize


def test_tokenize():
    assert "hello" in tokenize("Hello world! #fitness")
    assert "#fitness" in tokenize("Hello world! #fitness")
    assert "hi" not in tokenize("hi")  # too short


def test_jaccard_similarity():
    a = "5 home workouts for busy indian professionals"
    b = "5 home workouts for busy indian moms"
    c = "completely unrelated topic about cars"
    assert jaccard_similarity(a, b) > jaccard_similarity(a, c)
    assert jaccard_similarity("", "anything") == 0.0


def test_engagement_rate():
    post = {"likes": 100, "comments": 10, "shares": 5, "saves": 20, "reach": 1000}
    rate = engagement_rate(post, None)
    assert rate == (100 + 20 + 15 + 40) / 1000


def test_benchmark_median():
    posts = [
        {"likes": 10, "comments": 0, "shares": 0, "saves": 0, "reach": 100},
        {"likes": 30, "comments": 0, "shares": 0, "saves": 0, "reach": 100},
        {"likes": 20, "comments": 0, "shares": 0, "saves": 0, "reach": 100},
    ]
    assert abs(AutoLearner()._benchmark_median(posts, None) - 0.2) < 1e-6


def test_match_by_date_and_similarity():
    now = datetime.utcnow()
    prompt = {
        "id": 1,
        "prompt_type": "educational",
        "prompt_text": "5 home workouts for busy indian professionals",
        "sent_at": now - timedelta(days=2),
    }
    good_post = {
        "post_id": "p1",
        "caption": "5 home workouts for busy indian professionals #fitness",
        "likes": 50,
        "comments": 5,
        "posted_at": (now - timedelta(days=1)).isoformat(),
    }
    old_post = {
        "post_id": "p2",
        "caption": "5 home workouts for busy indian professionals #fitness",
        "likes": 50,
        "posted_at": (now - timedelta(days=10)).isoformat(),  # before prompt sent
    }
    unrelated_post = {
        "post_id": "p3",
        "caption": "totally unrelated car maintenance tips",
        "likes": 50,
        "posted_at": (now - timedelta(days=1)).isoformat(),
    }
    learner = AutoLearner()
    matched = learner._match([good_post, old_post, unrelated_post], [prompt])
    assert len(matched) == 1
    assert matched[0][3]["post_id"] == "p1"


def test_no_match_when_never_posted():
    now = datetime.utcnow()
    prompt = {"id": 1, "prompt_type": "educational", "prompt_text": "some prompt", "sent_at": now - timedelta(days=2)}
    assert AutoLearner()._match([], [prompt]) == []