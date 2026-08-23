from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from email.utils import formatdate
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.shared.config import settings

logger = logging.getLogger(__name__)


class EmailSender:
    """Sends emails via SMTP (Gmail app password by default).
    Supports inline images (cid:) and file attachments.
    When DRY_RUN is set, renders everything to logs/preview/ instead of sending."""

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

    @property
    def dry_run(self) -> bool:
        return bool(settings.dry_run)

    def send(
        self,
        to: str,
        subject: str,
        html: str,
        plain: str | None = None,
        inline_images: dict[str, str] | None = None,
        attachments: list[str] | None = None,
    ) -> bool:
        """inline_images: {cid: file_path}; attachments: [file_path]."""
        if self.dry_run:
            return self._write_preview(to, subject, html, inline_images or {}, attachments or [])
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
            from imghdr import what  # noqa: PLC0415 - stdlib image type sniffing

            for cid, path in (inline_images or {}).items():
                img_path = Path(path)
                if not img_path.exists():
                    logger.warning("::warning::Inline image missing, skipping cid=%s: %s", cid, path)
                    continue
                subtype = what(img_path) or "png"
                msg.get_payload()[1].add_related(
                    img_path.read_bytes(),
                    maintype="image",
                    subtype=subtype,
                    cid=f"<{cid}>",
                )
        except ImportError:
            # Python 3.13 removed imghdr; fall back to png
            for cid, path in (inline_images or {}).items():
                img_path = Path(path)
                if not img_path.exists():
                    continue
                msg.get_payload()[1].add_related(
                    img_path.read_bytes(), maintype="image", subtype="png", cid=f"<{cid}>"
                )
        except Exception as exc:
            logger.error("::warning::Inline image attach failed: %s", exc)

        for path in attachments or []:
            att_path = Path(path)
            if not att_path.exists():
                logger.warning("::warning::Attachment missing, skipping: %s", path)
                continue
            maintype, subtype = self._guess_type(att_path)
            try:
                msg.add_attachment(
                    att_path.read_bytes(),
                    maintype=maintype,
                    subtype=subtype,
                    filename=att_path.name,
                )
            except Exception as exc:
                logger.error("::warning::Attachment failed for %s: %s", path, exc)

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

        inline: dict[str, str] = {}
        attachments: list[str] = []
        for i, p in enumerate(prompts):
            asset = p.get("asset_path")
            if not asset:
                continue
            cid = f"asset{i}"
            if p.get("asset_type") == "reel":
                thumb = p.get("thumbnail")
                if thumb and Path(thumb).exists():
                    inline[cid] = thumb
                attachments.append(asset)
                p["asset_cid"] = cid
            else:
                if Path(asset).exists():
                    inline[cid] = asset
                p["asset_cid"] = cid

        html = template.render(business=business, prompts=prompts, date_str=self._today_str())
        return self.send(business["email"], subject, html, inline_images=inline, attachments=attachments)

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

        inline: dict[str, str] = {}
        attachments: list[str] = []
        li, pi = 0, 0
        for lead in leads:
            for p in lead.get("prompts") or []:
                asset = p.get("asset_path")
                if not asset:
                    continue
                if p.get("asset_type") == "reel":
                    thumb = p.get("thumbnail")
                    cid = f"lead-asset{li}"
                    li += 1
                    if thumb and Path(thumb).exists():
                        inline[cid] = thumb
                    attachments.append(asset)
                    p["asset_cid"] = cid
                else:
                    cid = f"lead-asset{li}"
                    li += 1
                    if Path(asset).exists():
                        inline[cid] = asset
                    p["asset_cid"] = cid
                pi += 1

        html = template.render(leads=leads, date_str=self._today_str())
        return self.send(to, subject, html, inline_images=inline, attachments=attachments)

    def _write_preview(
        self,
        to: str,
        subject: str,
        html: str,
        inline_images: dict[str, str],
        attachments: list[str],
    ) -> bool:
        """DRY_RUN: write rendered email + referenced assets to logs/preview/."""
        preview_dir = Path(settings.logs_dir) / "preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        slug = "".join(c if c.isalnum() else "-" for c in subject)[:60].strip("-").lower()
        out_html = preview_dir / f"{slug}.html"
        body = html
        for cid, path in inline_images.items():
            src = Path(path)
            if src.exists():
                import base64
                import mimetypes

                mime = mimetypes.guess_type(str(src))[0] or "image/png"
                b64 = base64.b64encode(src.read_bytes()).decode()
                body = body.replace(f"cid:{cid}", f"data:{mime};base64,{b64}")
        out_html.write_text(body, encoding="utf-8")
        import shutil

        copied = []
        for path in attachments:
            src = Path(path)
            if src.exists():
                dst = preview_dir / src.name
                shutil.copy2(src, dst)
                copied.append(dst.name)
        for path in inline_images.values():
            src = Path(path)
            if src.exists() and not (preview_dir / src.name).exists():
                shutil.copy2(src, preview_dir / src.name)
        logger.info("DRY RUN: preview written to %s (attachments: %d)", out_html, len(copied))
        print(f"[dry-run] email preview: {out_html} ({len(attachments)} attachments)")
        return True

    @staticmethod
    def _guess_type(path: Path) -> tuple[str, str]:
        suffix = path.suffix.lower()
        if suffix == ".mp4":
            return ("video", "mp4")
        if suffix == ".wav":
            return ("audio", "wav")
        if suffix == ".jpg" or suffix == ".jpeg":
            return ("image", "jpeg")
        return ("application", "octet-stream")

    @staticmethod
    def _today_str() -> str:
        import datetime

        return datetime.date.today().isoformat()
