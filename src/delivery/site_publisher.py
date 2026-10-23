"""Static-site delivery: the replacement for e-mail.

Every pipeline "delivers" by calling the same methods the e-mail sender had
(send_daily_prompts / send_weekly_report / send_onboarding / send_lead_digest).
Here they update JSON manifests inside `settings.site_dir` (a checkout of the
published `gh-pages` branch), copy optimised images / Reels next to them, and
re-render the whole site from the manifests. The workflows then push that folder.

Layout (all paths relative to the site dir):
    index.html                         landing page (lists businesses unless SITE_SECRET is set)
    b/<key>/index.html                 one business: latest prompts, archive, weekly reports
    b/<key>/weekly-<date>.html         weekly deep-dive
    leads/index.html (or l-<hash>/ with SITE_SECRET)     lead digests
    assets/<key>/<date>/<id>.jpg|mp4   media
    data/*.json                        manifests (the source of truth; HTML is derived)
    static/site.css, static/site.js
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import logging
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from src.shared.config import settings
from src.shared.privacy import redact, redact_text
from src.shared.utils import utc_now_naive
from src.storage.models import slugify

logger = logging.getLogger(__name__)

MAX_VIDEO_BYTES = 25 * 1024 * 1024
FEED_JPEG_QUALITY = 86
THUMB_WIDTH = 540
WEEKLY_KEEP = 26
DIGEST_KEEP = 8
NOTICE_KEEP = 7


def _today() -> str:
    return date.today().isoformat()


class SitePublisher:
    """Delivery channel that writes a static website."""

    name = "site"

    def __init__(self, site_dir: Path | str | None = None) -> None:
        self.site_dir = Path(site_dir or settings.site_dir)
        self._env = Environment(
            loader=FileSystemLoader(Path(settings.templates_dir) / "site"),
            autoescape=select_autoescape(["html"]),
        )
        self._env.filters["pretty_type"] = lambda v: str(v or "").replace("_", " ").title()
        self._env.filters["pretty_date"] = _pretty_date

    # ------------------------------------------------------------------
    # keys / urls
    # ------------------------------------------------------------------
    @staticmethod
    def path_key(name: str) -> str:
        """URL segment for a business. With SITE_SECRET it carries a keyed hash, so
        the page cannot be guessed from the name and the index does not list it."""
        slug = slugify(name)
        secret = settings.site_secret
        if not secret:
            return slug
        digest = hmac.new(secret.encode(), slug.encode(), hashlib.sha256).hexdigest()[:10]
        return f"{slug}-{digest}"

    def leads_key(self) -> str:
        secret = settings.site_secret
        if not secret:
            return "leads"
        return "l-" + hmac.new(secret.encode(), b"leads", hashlib.sha256).hexdigest()[:10]

    def base_url(self) -> str:
        if settings.site_base_url:
            return settings.site_base_url.rstrip("/")
        repo = settings.github_repo or ""
        if "/" not in repo:
            return ""
        owner, name = repo.split("/", 1)
        return f"https://{owner}.github.io" + ("" if name.lower() == f"{owner.lower()}.github.io" else f"/{name}")

    def business_url(self, name: str) -> str:
        return f"{self.base_url()}/b/{self.path_key(name)}/"

    # ------------------------------------------------------------------
    # manifests
    # ------------------------------------------------------------------
    def _data_dir(self) -> Path:
        d = self.site_dir / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _manifest_path(self, key: str) -> Path:
        return self._data_dir() / f"{key}.json"

    def _load_manifest(self, key: str) -> dict[str, Any]:
        path = self._manifest_path(key)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
            except (OSError, ValueError) as exc:
                logger.warning("Corrupt site manifest %s (%s); starting fresh", path, exc)
        return {"business": {}, "entries": [], "weekly": [], "notices": [], "about": None}

    def _open(self, name: str) -> tuple[str, str, dict[str, Any]]:
        """(manifest slug, current URL key, manifest). The manifest is named by the STABLE slug;
        only URLs/asset folders carry the secret-derived key, so rotating SITE_SECRET re-homes
        content instead of orphaning it (audit C-02)."""
        slug = slugify(name)
        pk = self.path_key(name)
        manifest = self._load_manifest(slug)
        self._rehome(pk, manifest)
        return slug, pk, manifest

    def _rehome(self, pk: str, manifest: dict[str, Any]) -> None:
        old = manifest.get("path_key")
        manifest["path_key"] = pk
        if not old or old == pk:
            return
        src, dst = self.site_dir / "assets" / old, self.site_dir / "assets" / pk
        if src.exists():
            if dst.exists():
                shutil.copytree(src, dst, dirs_exist_ok=True)
                shutil.rmtree(src, ignore_errors=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
        for entry in manifest.get("entries", []):
            asset = entry.get("asset")
            if isinstance(asset, dict):
                for field, value in list(asset.items()):
                    if isinstance(value, str):
                        asset[field] = value.replace(f"assets/{old}/", f"assets/{pk}/")
        logger.info("Re-homed site content %s -> %s", old, pk)

    def _save_manifest(self, key: str, manifest: dict[str, Any]) -> None:
        self._manifest_path(key).write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")

    def _touch_dirty(self) -> None:
        (self.site_dir / ".dirty").write_text(utc_now_naive().isoformat(), encoding="utf-8")

    @staticmethod
    def _business_meta(business: dict[str, Any]) -> dict[str, Any]:
        # never publish the contact e-mail address
        return {
            "name": business.get("name", ""),
            "instagram_handle": business.get("instagram_handle") or "",
            "website_url": business.get("website_url") or "",
        }

    # ------------------------------------------------------------------
    # assets
    # ------------------------------------------------------------------
    def _asset_dir(self, key: str, day: str) -> Path:
        d = self.site_dir / "assets" / key / day
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _import_assets(self, key: str, day: str, entry_id: str, prompt: dict[str, Any]) -> dict[str, Any] | None:
        """Copy the rendered media next to the site, optimised for the web."""
        src = prompt.get("asset_path")
        if not src or not Path(src).exists():
            return None
        out_dir = self._asset_dir(key, day)
        rel_base = f"assets/{key}/{day}"
        try:
            if prompt.get("asset_type") == "reel":
                size = Path(src).stat().st_size
                if size > MAX_VIDEO_BYTES:
                    logger.warning("Reel %s is %.1fMB (> %dMB); publishing its poster only", src, size / 1e6, MAX_VIDEO_BYTES // 2**20)
                    video = None
                else:
                    shutil.copy2(src, out_dir / f"{entry_id}.mp4")
                    video = f"{rel_base}/{entry_id}.mp4"
                poster_src = prompt.get("thumbnail")
                poster = self._save_jpeg(poster_src, out_dir / f"{entry_id}-poster.jpg") if poster_src and Path(poster_src).exists() else None
                return {"type": "reel", "video": video, "poster": f"{rel_base}/{poster}" if poster else None,
                        "thumb": f"{rel_base}/{poster}" if poster else None}
            full = self._save_jpeg(src, out_dir / f"{entry_id}.jpg")
            thumb = self._save_jpeg(src, out_dir / f"{entry_id}-thumb.jpg", width=THUMB_WIDTH)
            return {"type": "image", "image": f"{rel_base}/{full}", "thumb": f"{rel_base}/{thumb}"}
        except Exception as exc:  # noqa: BLE001 - a bad asset must not lose the prompt text
            logger.warning("::warning::Could not publish asset %s: %s", src, exc)
            return None

    @staticmethod
    def _save_jpeg(src: str | None, dest: Path, width: int | None = None) -> str:
        from PIL import Image

        img = Image.open(src).convert("RGB")  # type: ignore[arg-type]
        if width and img.width > width:
            img = img.resize((width, round(img.height * width / img.width)), Image.Resampling.LANCZOS)
        img.save(dest, "JPEG", quality=FEED_JPEG_QUALITY, optimize=True, progressive=True)
        return dest.name

    # ------------------------------------------------------------------
    # delivery API (same names as EmailSender so pipelines are channel-agnostic)
    # ------------------------------------------------------------------
    def send_daily_prompts(self, business: dict[str, Any], prompts: list[dict[str, Any]]) -> bool:
        slug, key, manifest = self._open(business["name"])
        manifest["business"] = self._business_meta(business)
        day = _today()

        if not prompts:
            manifest["notices"] = ([{"date": day, "message": "No prompts could be generated today (the AI providers were unavailable). The next scheduled run will try again."}]
                                   + [n for n in manifest.get("notices", []) if n.get("date") != day])[:NOTICE_KEEP]
        for i, p in enumerate(prompts):
            pid = p.get("prompt_id")
            entry_id = f"{day}-{pid if pid is not None else i}"
            entry = {
                "id": entry_id,
                "date": day,
                "created_at": utc_now_naive().isoformat(timespec="seconds"),
                "prompt_id": pid,
                "title": p.get("title") or "",
                "prompt_type": p.get("prompt_type") or "",
                "score": p.get("virality_score"),
                "below_threshold": bool(p.get("below_threshold")),
                "threshold": p.get("threshold"),
                "label": p.get("virality_label") or "",
                "hook": p.get("hook") or "",
                "angle": p.get("angle") or "",
                "prompt_text": p.get("prompt_text") or "",
                "caption": p.get("caption") or "",
                "hashtags": list(p.get("hashtags") or []),
                "first_comment": p.get("first_comment") or "",
                "recommended_post_time": p.get("recommended_post_time") or "",
                "format": p.get("format") or "",
                "platform_focus": p.get("platform_focus") or "",
                "audience_segment": p.get("audience_segment") or "",
                "reasoning": p.get("reasoning") or "",
                "key_factors": list(p.get("key_factors") or []),
                "risks": list(p.get("risks") or []),
                "warnings": list(p.get("warnings") or []),
                "asset": self._import_assets(key, day, entry_id, p),
                "feedback_links": {k: v for k, v in (p.get("feedback_links") or {}).items() if v and v != "#"},
            }
            # Defence in depth (audit C-02/H-01): model output must not carry contact details onto a
            # public page even if one slipped through (e.g. an address echoed from a scraped profile).
            entry.update(redact({k: v for k, v in entry.items()
                                 if k not in ("id", "date", "created_at", "prompt_id", "asset", "feedback_links")}))
            # a regenerated/re-run prompt replaces its earlier entry
            manifest["entries"] = [e for e in manifest["entries"] if e.get("id") != entry_id]
            manifest["entries"].append(entry)

        self._prune(key, manifest)
        self._save_manifest(slug, manifest)
        self._touch_dirty()
        self.rebuild()
        logger.info("Site updated for %s: %d new entries -> %s", business["name"], len(prompts), self.business_url(business["name"]))
        return True

    def send_weekly_report(self, business: dict[str, Any], report: dict[str, Any]) -> bool:
        slug, key, manifest = self._open(business["name"])
        manifest["business"] = self._business_meta(business)
        day = _today()
        manifest["weekly"] = ([{"date": day, "report": redact(report)}] + [w for w in manifest.get("weekly", []) if w.get("date") != day])[:WEEKLY_KEEP]
        self._save_manifest(slug, manifest)
        self._touch_dirty()
        self.rebuild()
        return True

    def send_onboarding(self, business: dict[str, Any], profile: dict[str, Any]) -> bool:
        slug, key, manifest = self._open(business["name"])
        manifest["business"] = self._business_meta(business)
        manifest["about"] = {
            "onboarded": _today(),
            "title": redact_text((profile or {}).get("title") or ""),
            "description": redact_text(((profile or {}).get("meta_description") or "")[:300]),
            "pages": (profile or {}).get("page_count") or 0,
            "headings": [redact_text(str(h)) for h in ((profile or {}).get("headings") or [])[:8]],
        }
        self._save_manifest(slug, manifest)
        self._touch_dirty()
        self.rebuild()
        return True

    def send_lead_digest(self, to: str, leads: list[dict[str, Any]]) -> bool:  # noqa: ARG002 - `to` kept for interface parity
        key = self.leads_key()
        path = self._data_dir() / "_leads.json"
        digests: list[dict[str, Any]] = []
        if path.exists():
            try:
                digests = json.loads(path.read_text(encoding="utf-8")).get("digests", [])
            except (OSError, ValueError):
                digests = []
        day = _today()
        published = []
        for li, lead in enumerate(leads):
            lead_prompts = []
            for pi, p in enumerate(lead.get("prompts") or []):
                entry_id = f"{day}-lead{li}-{pi}"
                lead_prompts.append({
                    "title": redact_text(p.get("title") or ""), "prompt_type": p.get("prompt_type") or "",
                    "score": p.get("virality_score"), "prompt_text": redact_text(p.get("prompt_text") or ""),
                    "caption": redact_text(p.get("caption") or ""),
                    "asset": self._import_assets("leads", day, entry_id, p),
                })
            published.append({
                "name": redact_text(lead.get("name") or ""), "instagram_handle": lead.get("instagram_handle") or "",
                "website_url": lead.get("website_url") or "", "niche": lead.get("niche") or "",
                "followers": lead.get("followers") or lead.get("follower_count") or 0,
                "opportunity_score": lead.get("opportunity_score"),
                "reasons": [redact_text(str(r)) for r in (lead.get("opportunity_reasons") or [])],
                "prompt_error": redact_text(lead.get("prompt_error") or ""),
                "prompts": lead_prompts,
            })
        digests = ([{"date": day, "leads": published}] + [d for d in digests if d.get("date") != day])[:DIGEST_KEEP]
        path.write_text(json.dumps({"digests": digests, "key": key}, indent=1, ensure_ascii=False), encoding="utf-8")
        keep_days = {d["date"] for d in digests}
        lead_assets = self.site_dir / "assets" / "leads"
        if lead_assets.exists():  # digests beyond DIGEST_KEEP no longer reference their media
            for day_dir in lead_assets.iterdir():
                if day_dir.is_dir() and day_dir.name not in keep_days:
                    shutil.rmtree(day_dir, ignore_errors=True)
        self._touch_dirty()
        self.rebuild()
        return True

    # ------------------------------------------------------------------
    # retention
    # ------------------------------------------------------------------
    def _prune(self, key: str, manifest: dict[str, Any]) -> None:
        cutoff = (date.today() - timedelta(days=settings.site_retention_days)).isoformat()
        manifest["entries"] = [e for e in manifest["entries"] if e.get("date", "") >= cutoff]
        keep_days = {e["date"] for e in manifest["entries"]}
        assets = self.site_dir / "assets" / key
        if assets.exists():
            for day_dir in assets.iterdir():
                if day_dir.is_dir() and day_dir.name not in keep_days:
                    shutil.rmtree(day_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # rendering
    # ------------------------------------------------------------------
    def rebuild(self) -> None:
        """Re-render every page from the manifests (idempotent) and delete whatever is no longer
        part of the current site: pages of removed businesses, pages under an old SITE_SECRET,
        stale lead pages and orphaned media."""
        self.site_dir.mkdir(parents=True, exist_ok=True)
        self._copy_static()
        businesses = []
        live: set[str] = set()
        for path in sorted(self._data_dir().glob("*.json")):
            if path.name.startswith("_"):
                continue
            manifest = self._load_manifest(path.stem)
            if not manifest.get("business"):
                continue
            name = manifest["business"]["name"]
            slug, pk = slugify(name), self.path_key(name)
            if path.stem != slug:  # legacy: manifest was named by its (secret-derived) path key
                target = self._manifest_path(slug)
                if target.exists():
                    path.unlink()
                    continue
                path.rename(target)
            self._rehome(pk, manifest)
            self._save_manifest(slug, manifest)
            businesses.append(self._render_business(pk, manifest))
            live.add(pk)
        has_leads = self._render_leads()
        self._sweep(live, self.leads_key() if has_leads else None)
        self._render_index(businesses, has_leads)

    def _sweep(self, live_business_keys: set[str], leads_key: str | None) -> list[str]:
        removed: list[str] = []

        def drop(path: Path) -> None:
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path.relative_to(self.site_dir).as_posix())

        pages = self.site_dir / "b"
        if pages.exists():
            for d in pages.iterdir():
                if d.is_dir() and d.name not in live_business_keys:
                    drop(d)
        assets = self.site_dir / "assets"
        if assets.exists():
            for d in assets.iterdir():
                if d.is_dir() and d.name not in live_business_keys | {"leads"}:
                    drop(d)
        for d in self.site_dir.iterdir():
            if d.is_dir() and (d.name == "leads" or d.name.startswith("l-")) and d.name != leads_key:
                drop(d)
        if removed:
            self._touch_dirty()  # the deploy must happen even if nothing else changed
            logger.warning("::warning::Removed stale site content: %s", ", ".join(removed))
        return removed

    def _copy_static(self) -> None:
        src = Path(settings.templates_dir) / "site" / "static"
        dest = self.site_dir / "static"
        dest.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*"):
            shutil.copy2(f, dest / f.name)
        (self.site_dir / ".nojekyll").write_text("", encoding="utf-8")
        (self.site_dir / "robots.txt").write_text("User-agent: *\nDisallow: /\n", encoding="utf-8")

    def _write(self, rel: str, html: str) -> None:
        path = self.site_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")

    def _render_business(self, key: str, manifest: dict[str, Any]) -> dict[str, Any]:
        entries = sorted(manifest["entries"], key=lambda e: e.get("created_at", ""), reverse=True)
        days: dict[str, list[dict[str, Any]]] = {}
        for e in entries:
            days.setdefault(e["date"], []).append(e)
        grouped = [{"date": d, "entries": days[d]} for d in sorted(days, reverse=True)]
        types = sorted({e["prompt_type"] for e in entries if e.get("prompt_type")})
        meta = manifest["business"]
        html = self._env.get_template("business.html").render(
            root="../../", business=meta, days=grouped, types=types, weekly=manifest.get("weekly", []),
            notices=manifest.get("notices", []), about=manifest.get("about"), total=len(entries),
            retention=settings.site_retention_days,
        )
        self._write(f"b/{key}/index.html", html)
        for w in manifest.get("weekly", []):
            page = self._env.get_template("weekly.html").render(
                root="../../", business=meta, report=w["report"], date=w["date"], back="index.html"
            )
            self._write(f"b/{key}/weekly-{w['date']}.html", page)
        return {"key": key, "name": meta["name"], "count": len(entries),
                "latest": grouped[0]["date"] if grouped else "", "weekly": len(manifest.get("weekly", []))}

    def _render_leads(self) -> bool:
        path = self._data_dir() / "_leads.json"
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        key = self.leads_key()
        html = self._env.get_template("leads.html").render(root="../", digests=data.get("digests", []))
        self._write(f"{key}/index.html", html)
        return True

    def _render_index(self, businesses: list[dict[str, Any]], has_leads: bool) -> None:
        listed = [] if settings.site_secret else businesses
        html = self._env.get_template("index.html").render(
            root="", businesses=listed, private=bool(settings.site_secret),
            leads_href=(f"{self.leads_key()}/" if has_leads and not settings.site_secret else ""),
        )
        self._write("index.html", html)

    # ------------------------------------------------------------------
    def summary(self) -> list[str]:
        """Lines for the workflow summary: where the owner finds the pages."""
        lines = []
        base = self.base_url()
        for path in sorted(self._data_dir().glob("*.json")):
            if path.name.startswith("_"):
                continue
            manifest = self._load_manifest(path.stem)
            if manifest.get("business"):
                lines.append(f"- {manifest['business']['name']}: {base}/b/{self.path_key(manifest['business']['name'])}/")
        if (self._data_dir() / "_leads.json").exists():
            lines.append(f"- Leads: {base}/{self.leads_key()}/")
        return lines


def _pretty_date(value: str) -> str:
    try:
        d = datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return str(value)
    if d == date.today():
        return "Today"
    if d == date.today() - timedelta(days=1):
        return "Yesterday"
    return d.strftime("%a %d %b %Y")
