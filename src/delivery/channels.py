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


class Delivery:
    def __init__(self, channels: list[Any]) -> None:
        self.channels = channels

    @property
    def names(self) -> list[str]:
        return [getattr(c, "name", type(c).__name__) for c in self.channels]

    def _fan_out(self, method: str, *args: Any) -> bool:
        delivered = False
        for channel in self.channels:
            label = getattr(channel, "name", type(channel).__name__)
            try:
                ok = bool(getattr(channel, method)(*args))
            except Exception as exc:  # noqa: BLE001 - one channel failing must not block the others
                logger.error("::error::Delivery via %s failed in %s: %s", label, method, exc)
                ok = False
            delivered = delivered or ok
        return delivered

    def send_daily_prompts(self, business: dict[str, Any], prompts: list[dict[str, Any]]) -> bool:
        return self._fan_out("send_daily_prompts", business, prompts)

    def send_weekly_report(self, business: dict[str, Any], report: dict[str, Any]) -> bool:
        return self._fan_out("send_weekly_report", business, report)

    def send_onboarding(self, business: dict[str, Any], profile: dict[str, Any]) -> bool:
        return self._fan_out("send_onboarding", business, profile)

    def send_lead_digest(self, to: str, leads: list[dict[str, Any]]) -> bool:
        return self._fan_out("send_lead_digest", to, leads)


def get_delivery(mode: str | None = None) -> Delivery:
    mode = (mode or settings.delivery_mode or "site").strip().lower()
    channels: list[Any] = []
    if mode in ("site", "both") or mode not in ("email",):
        from src.delivery.site_publisher import SitePublisher

        channels.append(SitePublisher())
    if mode in ("email", "both"):
        from src.delivery.email_sender import EmailSender

        channels.append(EmailSender())
    return Delivery(channels)
