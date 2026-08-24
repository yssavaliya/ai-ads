from unittest.mock import patch

from src.storage.models import Base, Business, Prompt, PromptType, SessionLocal, engine as _models_engine
from sqlalchemy import create_engine as _ce
from sqlalchemy.orm import sessionmaker as _sm


CANNED_VARIANT = {
    "prompt_text": "Regenerated idea: try a 15s Reel showing a before/after AI edit",
    "prompt_type": "educational",
    "format": "reel",
    "hook": "See this 15s AI edit trick",
    "hashtags": ["#AI", "#Ahmedabad"],
    "post_idea": "Show a before/after AI edit",
    "image_prompt": "cinematic before after edit",
    "title": "AI Edit Reel",
    "visual_prompt": "cinematic",
}


def _tmp_db(tmp_path):
    eng = _ce(f"sqlite:///{tmp_path / 'fb.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    return _sm(bind=eng, autoflush=False, autocommit=False)()


def test_regenerate_prompt_creates_and_emails(tmp_path, monkeypatch):
    """REGENERATE must actually generate a replacement prompt, score it, render
    an asset, persist it, and email it — without crashing on the predictor's
    5-argument score() signature."""
    import src.generation.creative_forge as cf
    import src.generation.prompt_forge as pf
    import src.intelligence.virality_predictor as _vp
    import src.delivery.email_sender as es
    import src.learning.process_feedback as pfmod
    import src.shared.utils as utils

    monkeypatch.setattr(utils, "load_businesses", lambda: [
        {"name": "RegenBiz", "email": "r@g.com", "instagram_handle": "regenbiz"}
    ])
    monkeypatch.setattr(pf.PromptForge, "generate_variants", classmethod(
        lambda cls, *a, **k: [dict(CANNED_VARIANT)]
    ))
    monkeypatch.setattr(_vp.ViralityPredictor, "score", lambda self, *a, **k: {
        "virality_score": 82, "virality_label": "high", "virality_probability": 0.8,
        "key_factors": ["x"], "risks": [], "reasoning": "r", "improvement_hints": [],
    })
    monkeypatch.setattr(cf.CreativeForge, "render_prompt", lambda self, *a, **k: {
        "asset_path": "x.mp4", "asset_type": "reel", "thumbnail": None,
    })

    sent = {}
    def fake_send(self, business, saved):
        sent["calls"] = sent.get("calls", 0) + 1
        sent["n"] = len(saved)
        return True
    monkeypatch.setattr(es.EmailSender, "send_daily_prompts", fake_send)

    db = _tmp_db(tmp_path)
    try:
        biz = Business(
            name="RegenBiz", email="r@g.com", virality_threshold=75,
            profile_json={"name": "RegenBiz"}, audience_personas_json={"best_posting_times": ["19:00"]},
        )
        db.add(biz)
        db.commit()
        db.refresh(biz)
        orig = Prompt(
            business_id=biz.id,
            prompt_text="original weak idea",
            prompt_type=PromptType.EDUCATIONAL,
            sent_at=__import__("datetime").datetime.utcnow(),
        )
        db.add(orig)
        db.commit()
        db.refresh(orig)

        ok = pfmod.regenerate_prompt(orig, db)
        assert ok is True, "regenerate should succeed"
        assert sent.get("calls") == 1, sent
        assert sent.get("n") == 1, sent
        new_count = db.query(Prompt).filter(Prompt.business_id == biz.id).count()
        assert new_count == 2, new_count  # original + regenerated
    finally:
        db.close()
