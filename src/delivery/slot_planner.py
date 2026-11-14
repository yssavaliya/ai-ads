"""Pick the actual publish time for a post.

Inputs are pure data (no network), so every rule is unit-tested:
- the wanted local hour (measured window, experiment arm, or fallback) in the business timezone;
- the times already taken in the Buffer queue;
- a minimum gap between posts (an account should not publish two ads inside a day) and a lead time so Buffer
  always has time to process the post.
The result is a timezone-aware UTC datetime, or None when no slot fits inside the horizon.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

DEFAULT_TZ = "Asia/Kolkata"
DEFAULT_HOUR = 19
MIN_GAP_HOURS = 20.0
LEAD_MINUTES = 90
HORIZON_DAYS = 10


def parse_due(value: Any) -> datetime | None:
    """A Buffer ``dueAt`` (ISO string) as an aware UTC datetime."""
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _zone(name: str | None) -> ZoneInfo:
    try:
        return ZoneInfo(name or DEFAULT_TZ)
    except Exception:  # noqa: BLE001 - unknown timezone name in config
        return ZoneInfo(DEFAULT_TZ)


def plan_slot(
    now: datetime,
    hour_local: int | None,
    tz_name: str | None = None,
    taken: list[datetime] | None = None,
    min_gap_hours: float = MIN_GAP_HOURS,
    lead_minutes: int = LEAD_MINUTES,
    horizon_days: int = HORIZON_DAYS,
    minute: int = 0,
) -> datetime | None:
    """First future slot at ``hour_local`` (business timezone) that keeps ``min_gap_hours`` from every taken time."""
    zone = _zone(tz_name)
    hour = DEFAULT_HOUR if hour_local is None else max(0, min(23, int(hour_local)))
    now_utc = now.astimezone(UTC) if now.tzinfo else now.replace(tzinfo=UTC)
    earliest = now_utc + timedelta(minutes=lead_minutes)
    busy = [t.astimezone(UTC) for t in (taken or []) if isinstance(t, datetime)]
    gap = timedelta(hours=min_gap_hours)

    day = now_utc.astimezone(zone).date()
    for offset in range(horizon_days + 1):
        local = datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone) + timedelta(days=offset)
        # re-anchor after adding days so DST changes keep the wall-clock hour
        local = datetime(local.year, local.month, local.day, hour, minute, tzinfo=zone)
        candidate = local.astimezone(UTC)
        if candidate < earliest:
            continue
        if all(abs(candidate - other) >= gap for other in busy):
            return candidate
    return None


def describe(slot: datetime, tz_name: str | None = None) -> str:
    local = slot.astimezone(_zone(tz_name))
    return local.strftime("%a %d %b %Y, %I:%M %p ") + (tz_name or DEFAULT_TZ).replace("_", " ")
