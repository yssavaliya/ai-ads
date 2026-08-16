from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from email.utils import formatdate
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.shared.config import settings

logger = logging.getLogger(__name__)


class EmailSender:
    """Sends emails via SMTP (Gmail app password by default)."""

    def __init__(self, host: str | None = None, port: int | None = None) -> None:
        self.host = host or settings.smtp_host
        self.port = port or settings.smtp_port
        self.user = settings.smtp_user
        self.password = settings.smtp_password
        self.from_addr = settings.smtp_from or self.user
        self._env = Environment(
            loader=FileSystemLoader(settings.templates_dir),
            autoescape=select_autoescape(["html"]),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.user and self.password)

    def send(self, to: str, subject: str, html: str, plain: str | None = None) -> bool:
        if not self.enabled:
            logger.warning("SMTP not configured; skipping email to %s: %s", to, subject)
            return False

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = to
        msg["Date"] = formatdate(localtime=True)
        msg.set_content(plain or "Open this email in an HTML-capable client.")
        msg.add_alternative(html, subtype="html")

        try:
            with smtplib.SMTP(self.host, self.port, timeout=60) as server:
                if settings.smtp_tls:
                    server.starttls()
                server.login(self.user, self.password)
                server.send_message(msg)
            logger.info("Email sent to %s: %s", to, subject)
            return True
        except Exception as exc:
            logger.error("Failed to send email to %s: %s", to, exc)
            return False

    def send_daily_prompts(self, business: dict[str, Any], prompts: list[dict[str, Any]]) -> bool:
        template = self._env.get_template("daily_prompt.html")
        html = template.render(business=business, prompts=prompts, date_str=self._today_str())
        subject = f"🎯 Tomorrow's {len(prompts)} Viral Content Prompts for {business['name']}"
        return self.send(business["email"], subject, html)

    def send_weekly_report(self, business: dict[str, Any], report: dict[str, Any]) -> bool:
        template = self._env.get_template("weekly_report.html")
        html = template.render(business=business, report=report, date_str=self._today_str())
        subject = f"📊 Weekly Intelligence Report for {business['name']}"
        return self.send(business["email"], subject, html)

    def send_onboarding(self, business: dict[str, Any], profile: dict[str, Any]) -> bool:
        template = self._env.get_template("welcome.html")
        html = template.render(business=business, profile=profile, date_str=self._today_str())
        subject = f"🚀 {business['name']} is live — here's your content intelligence setup"
        return self.send(business["email"], subject, html)

    def send_lead_digest(self, to: str, leads: list[dict[str, Any]]) -> bool:
        template = self._env.get_template("leads_digest.html")
        html = template.render(leads=leads, date_str=self._today_str())
        subject = f"🔍 {len(leads)} new business leads with sample viral prompts"
        return self.send(to, subject, html)

    @staticmethod
    def _today_str() -> str:
        import datetime

        return datetime.date.today().isoformat()