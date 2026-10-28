"""Delivery channels: how finished content reaches people.

DELIVERY_MODE (default "site"):
    site   publish to the static website (replaces e-mail)
    email  the legacy SMTP e-mails only
    both   site + e-mail

Pipelines talk to one `Delivery` object; it fans every call out to the configured
channels and reports success if at least one of them delivered.
"""
from __future__ import annotations

import logging
from typing import Any

from src.shared.config import settings

logger = logging.getLogger(__name__)

METHODS = ("send_daily_prompts", "send_weekly_report", "send_onboarding", "send_lead_digest")


class DeliveryResult:
    """Outcome per channel. Truthy when AT LEAST ONE channel delivered (so `if result:` keeps its
    old meaning), but partial failures stay visible: `partial` / `failed` / `errors`."""

    def __init__(self, channels: dict[str, bool], errors: dict[str, str] | None = None) -> None:
        self.channels = channels
        self.errors = errors or {}

    @property
    def ok(self) -> bool:
        return any(self.channels.values())

    @property
    def failed(self) -> list[str]:
        return [name for name, done in self.channels.items() if not done]

    @property
    def partial(self) -> bool:
        return self.ok and bool(self.failed)

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        return f"DeliveryResult({self.channels})"


class Delivery:
    def __init__(self, channels: list[Any]) -> None:
        self.channels = channels

    @property
    def names(self) -> list[str]:
        return [getattr(c, "name", type(c).__name__) for c in self.channels]

    def _fan_out(self, method: str, *args: Any) -> DeliveryResult:
        outcome: dict[str, bool] = {}
        errors: dict[str, str] = {}
        for channel in self.channels:
            label = getattr(channel, "name", type(channel).__name__)
            try:
                outcome[label] = bool(getattr(channel, method)(*args))
            except Exception as exc:  # noqa: BLE001 - one channel failing must not block the others
                logger.error("::error::Delivery via %s failed in %s: %s", label, method, exc)
                outcome[label], errors[label] = False, str(exc)[:200]
            if not outcome[label]:
                logger.warning("::warning::Channel %s did not deliver (%s)", label, method)
        result = DeliveryResult(outcome, errors)
        if result.partial:
            logger.warning("::warning::PARTIAL delivery: ok=%s failed=%s", [n for n, d in outcome.items() if d], result.failed)
        return result

    def send_daily_prompts(self, business: dict[str, Any], prompts: list[dict[str, Any]]) -> DeliveryResult:
        return self._fan_out("send_daily_prompts", business, prompts)

    def send_weekly_report(self, business: dict[str, Any], report: dict[str, Any]) -> DeliveryResult:
        return self._fan_out("send_weekly_report", business, report)

    def send_onboarding(self, business: dict[str, Any], profile: dict[str, Any]) -> DeliveryResult:
        return self._fan_out("send_onboarding", business, profile)

    def send_lead_digest(self, to: str, leads: list[dict[str, Any]]) -> DeliveryResult:
        return self._fan_out("send_lead_digest", to, leads)


MODES = ("site", "email", "both", "buffer")


def get_delivery(mode: str | None = None) -> Delivery:
    raw_mode = (mode or settings.delivery_mode or "site").strip().lower()
    if raw_mode not in MODES and not ("," in raw_mode and "buffer" in raw_mode):
        raise ValueError(f"Unknown DELIVERY_MODE {raw_mode!r}; expected one of {', '.join(MODES)}")
    selected = [m.strip() for m in raw_mode.split(",") if m.strip()]
    for m in selected:
        if m not in MODES:
            raise ValueError(f"Unknown DELIVERY_MODE {m!r}; expected one of {', '.join(MODES)}")

    channels: list[Any] = []
    if any(m in ("site", "both") for m in selected):
        from src.delivery.site_publisher import SitePublisher

        channels.append(SitePublisher())
    if any(m in ("email", "both") for m in selected):
        from src.delivery.email_sender import EmailSender

        channels.append(EmailSender())
    if any(m == "buffer" for m in selected):
        from src.delivery.buffer_publisher import BufferPublisher

        channels.append(BufferPublisher())
    return Delivery(channels)
