"""The Engagement job must explain a '0 learned' run: why nothing was learned has to be visible on the
run page (GitHub log downloads are not always available, and a green run with no state change looks
exactly like a run that did nothing)."""
from __future__ import annotations

from pathlib import Path

from src.learning import auto_learner as al
from src.learning.auto_learner import (
    MATURITY_HOURS,
    AutoLearner,
    format_summary,
    write_step_summary,
)
from tests.test_self_learning import _learn, _post, _prompt, lab  # noqa: F401  (lab is a fixture)


# ------------------------------------------------------------------ counts that explain a zero
def test_counts_explain_every_post_that_was_not_learned(lab):  # noqa: F811
    db, biz = lab
    posts = []
    _, c1 = _prompt(db, biz, "reel", 1)
    posts.append(_post(c1, 50, 1))  # mature, matched -> learned
    _, c2 = _prompt(db, biz, "reel", 2)
    posts.append(_post(c2, 50, 2, age_hours=MATURITY_HOURS - 20))  # matched but too new
    _, c3 = _prompt(db, biz, "reel", 3)
    posts.append(_post(c3, -1, 3, hidden=True))  # matched, likes hidden
    posts.append(_post("an unrelated holiday photo of the beach", 40, 4))  # matches no prompt

    res = _learn(db, biz, posts)
    assert res["posts"] == 4 and res["candidates"] == 3
    assert res["matched"] == 3 and res["unmatched"] == 1
    assert res["learned"] == 1 and res["deferred"] == 1 and res["unusable"] == 1
    assert res["hidden_likes"] == 1
    # nothing unaccounted for
    assert res["matched"] == res["learned"] + res["deferred"] + res["unusable"] + res["refreshed"]


def test_an_empty_scrape_reports_the_scrape_error_not_just_zero(lab):  # noqa: F811
    db, biz = lab
    res = AutoLearner().run_for_business(biz.id, {"handle": "x", "error": "no apify token", "posts": []}, db)
    assert res == {"posts": 0, "matched": 0, "learned": 0, "scrape_error": "no apify token", "followers": None}
    quiet = AutoLearner().run_for_business(biz.id, {"handle": "x", "posts": []}, db)
    assert quiet["scrape_error"] is None  # an empty profile is not an error


# ------------------------------------------------------------------ run_auto_learning reports skips
def test_businesses_that_cannot_be_processed_are_listed_with_a_reason(lab, monkeypatch):  # noqa: F811
    db, biz = lab
    monkeypatch.setattr(al, "SessionLocal", lambda: db)
    monkeypatch.setattr(db, "close", lambda: None)
    monkeypatch.setattr(al, "load_businesses", lambda: [
        {"name": "Learning Lab Bakery", "instagram_handle": ""},
        {"name": "Never Ran Co", "instagram_handle": "neverran"},
    ])
    results = al.run_auto_learning()
    assert results["Learning Lab Bakery"] == {"skipped": "no instagram_handle configured"}
    assert "not in the database" in results["Never Ran Co"]["skipped"]


def test_one_failing_business_is_reported_and_does_not_stop_the_run(lab, monkeypatch):  # noqa: F811
    db, biz = lab
    monkeypatch.setattr(al, "SessionLocal", lambda: db)
    monkeypatch.setattr(db, "close", lambda: None)
    monkeypatch.setattr(al, "load_businesses", lambda: [{"name": "Learning Lab Bakery", "instagram_handle": "lab"}])

    async def boom(self, handle, limit=20):
        raise RuntimeError("actor crashed")

    monkeypatch.setattr("src.fetch.social_scraper.SocialScraper.fetch_instagram", boom)
    results = al.run_auto_learning()
    assert results["Learning Lab Bakery"]["error"] == "actor crashed"
    assert results["Learning Lab Bakery"]["learned"] == 0


# ------------------------------------------------------------------ the markdown
def test_summary_table_shows_each_business_and_every_explaining_column():
    md = format_summary({
        "Alpha": {"posts": 12, "candidates": 5, "matched": 4, "unmatched": 8, "learned": 3, "deferred": 1,
                  "unusable": 0, "refreshed": 0, "hidden_likes": 2},
        "Beta": {"posts": 0, "matched": 0, "learned": 0, "scrape_error": "no apify token"},
        "Gamma": {"skipped": "no instagram_handle configured"},
        "Delta": {"matched": 0, "learned": 0, "error": "actor crashed"},
    })
    for label in ("Posts found", "Prompts eligible", "Matched", f"Too new (<{MATURITY_HOURS}h)", "Likes hidden", "Learned"):
        assert label in md
    assert "| Alpha | 12 | 5 | 4 | 1 | 2 | 0 | 0 | 3 |" in md
    assert "**Total learned this run: 3**" in md and "Nothing was learned" not in md
    assert "**Beta** scrape returned no posts: no apify token" in md
    assert "**Gamma** skipped: no instagram_handle configured" in md
    assert "**Delta** failed: actor crashed" in md


def test_a_run_that_learned_nothing_says_so_and_points_at_the_columns():
    md = format_summary({"Alpha": {"posts": 6, "candidates": 2, "matched": 2, "learned": 0, "deferred": 2}})
    assert "Total learned this run: 0" in md and "Nothing was learned" in md
    assert format_summary({}).endswith("No active business to process.")


def test_zero_is_printed_as_zero_and_a_missing_count_as_a_dash():
    rows = [ln for ln in format_summary({"A": {"posts": 0, "learned": 0}}).splitlines() if ln.startswith("| A")]
    cells = [c.strip() for c in rows[0].split("|")]
    assert cells[2] == "0"  # posts=0 is data
    assert cells[3] == "-"  # candidates was never computed


# ------------------------------------------------------------------ GITHUB_STEP_SUMMARY
def test_summary_is_appended_to_the_github_step_summary_file(tmp_path, monkeypatch):
    target = tmp_path / "summary.md"
    target.write_text("earlier step\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))
    assert write_step_summary({"Alpha": {"posts": 1, "matched": 1, "learned": 1}}) is True
    text = target.read_text(encoding="utf-8")
    assert text.startswith("earlier step\n") and "| Alpha |" in text  # appended, not overwritten


def test_no_summary_is_written_outside_actions_or_when_the_file_is_unwritable(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    assert write_step_summary({"A": {}}) is False
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "missing-dir" / "s.md"))
    assert write_step_summary({"A": {}}) is False  # a reporting problem must never fail the learning run


def test_the_module_entrypoint_used_by_the_workflow_writes_the_summary(tmp_path, monkeypatch):
    """engagement.yml runs `python -m src.learning.auto_learner`; execute that real entrypoint."""
    import runpy

    target = tmp_path / "s.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(target))
    monkeypatch.setattr(al, "run_auto_learning", lambda: {"Alpha": {"posts": 3, "matched": 0, "learned": 0}})
    monkeypatch.setattr("logging.basicConfig", lambda **kw: None)
    # run_name="__main__" re-executes the module top to bottom, so stub the one slow dependency at its source
    monkeypatch.setattr("src.fetch.social_scraper.SocialScraper.fetch_instagram", None, raising=False)
    import src.storage.models as models

    monkeypatch.setattr(models, "SessionLocal", lambda: _NoBusinessesDb())
    monkeypatch.setattr("src.shared.utils.load_businesses", lambda: [])
    runpy.run_path(al.__file__, run_name="__main__")
    text = target.read_text(encoding="utf-8")
    assert "### Engagement sync" in text and "No active business to process." in text


class _NoBusinessesDb:
    def close(self):
        return None


def test_engagement_workflow_runs_the_entrypoint_that_writes_the_summary():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "engagement.yml").read_text(encoding="utf-8")
    assert "run_pipeline.sh src.learning.auto_learner" in workflow
    entry = Path(al.__file__).read_text(encoding="utf-8").split('if __name__ == "__main__":')[1]
    assert "write_step_summary(outcome)" in entry


# ------------------------------------------------------------------ Apify reports failures as ITEMS
def test_an_actor_error_item_is_an_error_not_an_empty_profile():
    from src.fetch.social_scraper import SocialScraper

    norm = SocialScraper._normalize_instagram
    private = norm([{"error": "no_items", "errorDescription": "Empty or private data for provided input"}])
    assert private["posts"] == [] and private["error"] == "no_items: Empty or private data for provided input"
    assert norm([])["error"] == "the actor returned no items"
    assert norm([{"followersCount": 10, "latestPosts": []}])["error"] is None  # a real profile with no posts is not an error
    ok = norm([{"followersCount": 10, "latestPosts": [{"shortCode": "A", "likesCount": 5, "caption": "hi"}]}])
    assert ok["error"] is None and len(ok["posts"]) == 1


def test_a_profile_with_no_posts_is_explained_in_the_summary():
    md = format_summary({
        "New": {"posts": 0, "matched": 0, "learned": 0, "scrape_error": None, "followers": 120},
        "Gone": {"posts": 0, "matched": 0, "learned": 0, "scrape_error": None, "followers": None},
    })
    assert "**New** has no posts to learn from yet: profile found (120 followers)." in md
    assert "**Gone** has no posts to learn from yet: no profile data came back." in md
