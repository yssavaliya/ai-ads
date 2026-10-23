from __future__ import annotations

import re
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy import (
    Enum as SQLEnum,
)
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

from src.shared.config import settings
from src.shared.utils import utc_now_naive


class Base(DeclarativeBase):
    pass


def slugify(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-") or "business"


class PromptType(StrEnum):
    EDUCATIONAL = "educational"
    STORYTELLING = "storytelling"
    SOCIAL_PROOF = "social_proof"
    PROMOTIONAL = "promotional"
    ENGAGEMENT = "engagement"
    BEHIND_SCENES = "behind_scenes"
    USER_GENERATED = "user_generated"
    TREND_JACKING = "trend_jacking"


class FeedbackType(StrEnum):
    GOOD = "good"
    BAD = "bad"
    REGENERATE = "regenerate"


class Platform(StrEnum):
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    TIKTOK = "tiktok"


class Business(Base):
    __tablename__ = "businesses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    website_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    instagram_handle: Mapped[str | None] = mapped_column(String(100), nullable=True)
    linkedin_handle: Mapped[str | None] = mapped_column(String(100), nullable=True)
    twitter_handle: Mapped[str | None] = mapped_column(String(100), nullable=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    timezone: Mapped[str] = mapped_column(String(50), default="Asia/Kolkata")
    virality_threshold: Mapped[int] = mapped_column(Integer, default=75)
    prompt_types: Mapped[list[str]] = mapped_column(JSON, default=lambda: [
        "educational", "storytelling", "social_proof"
    ])
    profile_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    audience_personas_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utc_now_naive, onupdate=utc_now_naive
    )

    @property
    def weights_key(self) -> str:
        """Key for data/weights.json. Derived from the name, NOT the row id:
        agent.db lives in the Actions cache and ids are reassigned if it is
        evicted, which used to attach learned weights to the wrong business."""
        return slugify(self.name)

    prompts: Mapped[list[Prompt]] = relationship(back_populates="business", cascade="all, delete-orphan")
    competitors: Mapped[list[Competitor]] = relationship(back_populates="business", cascade="all, delete-orphan")


class Prompt(Base):
    __tablename__ = "prompts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # Durable random identity. The integer id is recycled if the cached database is rebuilt, so
    # nothing external (feedback links) may refer to it.
    uid: Mapped[str] = mapped_column(String(32), default=lambda: uuid.uuid4().hex, index=True)
    business_id: Mapped[int] = mapped_column(Integer, ForeignKey("businesses.id"), nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_type: Mapped[PromptType] = mapped_column(SQLEnum(PromptType), nullable=False)
    virality_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    virality_label: Mapped[str | None] = mapped_column(String(20), nullable=True)
    virality_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    key_factors: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    risks: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    generation_attempt: Mapped[int] = mapped_column(Integer, default=1)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    feedback: Mapped[FeedbackType | None] = mapped_column(SQLEnum(FeedbackType), nullable=True)
    feedback_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    github_issue_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Set when the auto-learner has counted this prompt's post toward the weights (exactly once).
    auto_learned_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    audience_segment: Mapped[str | None] = mapped_column(String(255), nullable=True)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    asset_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    asset_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    format: Mapped[str | None] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)

    business: Mapped[Business] = relationship(back_populates="prompts")
    engagements: Mapped[list[Engagement]] = relationship(back_populates="prompt", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_prompts_business_sent", "business_id", "sent_at"),
        Index("ix_prompts_feedback", "feedback"),
    )


class Engagement(Base):
    __tablename__ = "engagements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prompt_id: Mapped[int] = mapped_column(Integer, ForeignKey("prompts.id"), nullable=False)
    platform: Mapped[Platform] = mapped_column(SQLEnum(Platform), nullable=False)
    post_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    post_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    likes: Mapped[int] = mapped_column(Integer, default=0)
    comments: Mapped[int] = mapped_column(Integer, default=0)
    shares: Mapped[int] = mapped_column(Integer, default=0)
    saves: Mapped[int] = mapped_column(Integer, default=0)
    reach: Mapped[int] = mapped_column(Integer, default=0)
    impressions: Mapped[int] = mapped_column(Integer, default=0)
    profile_visits: Mapped[int] = mapped_column(Integer, default=0)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)
    posted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    prompt: Mapped[Prompt] = relationship(back_populates="engagements")

    __table_args__ = (
        Index("ix_engagements_prompt_fetched", "prompt_id", "fetched_at"),
        UniqueConstraint("prompt_id", "platform", "post_id", name="uq_engagement_prompt_platform_post"),
    )


class Competitor(Base):
    __tablename__ = "competitors"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    business_id: Mapped[int] = mapped_column(Integer, ForeignKey("businesses.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    platform: Mapped[Platform] = mapped_column(SQLEnum(Platform), nullable=False)
    handle: Mapped[str] = mapped_column(String(100), nullable=False)
    profile_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    follower_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_fetched_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    top_posts_json: Mapped[list[dict] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive, onupdate=utc_now_naive)

    business: Mapped[Business] = relationship(back_populates="competitors")

    __table_args__ = (
        UniqueConstraint("business_id", "platform", "handle", name="uq_competitor_business_platform_handle"),
    )


class Trend(Base):
    __tablename__ = "trends"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hashtag: Mapped[str] = mapped_column(String(100), nullable=False)
    platform: Mapped[Platform] = mapped_column(SQLEnum(Platform), nullable=False)
    volume: Mapped[int | None] = mapped_column(Integer, nullable=True)
    growth_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    related_keywords: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)

    __table_args__ = (
        UniqueConstraint("hashtag", "platform", "fetched_at", name="uq_trend_hashtag_platform_date"),
        Index("ix_trends_platform_fetched", "platform", "fetched_at"),
    )


engine = create_engine(
    f"sqlite:///{settings.database_path}",
    connect_args={"check_same_thread": False},
    echo=settings.log_level == "DEBUG",
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=10000")
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()


_init_done = False


def init_db() -> None:
    global _init_done
    if _init_done:
        return
    Base.metadata.create_all(bind=engine)
    _migrate()
    _init_done = True


def _migrate(bind: Any = None) -> None:
    """Lightweight additive migrations for SQLite (create_all won't ALTER)."""
    import logging

    migrations = [
        ("prompts", "audience_segment", "VARCHAR(255)"),
        ("prompts", "caption", "TEXT"),
        ("prompts", "asset_path", "VARCHAR(500)"),
        ("prompts", "asset_type", "VARCHAR(20)"),
        ("prompts", "format", "VARCHAR(40)"),
        ("prompts", "uid", "VARCHAR(32)"),
        ("prompts", "auto_learned_at", "DATETIME"),
    ]
    log = logging.getLogger(__name__)
    with (bind or engine).connect() as conn:
        for table, column, ddl in migrations:
            try:
                conn.exec_driver_sql(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                log.info("Migration: added %s.%s", table, column)
            except OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    log.warning("Migration %s.%s failed: %s", table, column, exc)
        # rows created before `uid` existed get a random one (never derived from the id)
        conn.exec_driver_sql("UPDATE prompts SET uid = lower(hex(randomblob(16))) WHERE uid IS NULL OR uid = ''")
        conn.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Guarantee schema + additive migrations run on import of this module, so no
# entry point can ever query a stale (pre-migration) cached agent.db.
init_db()
