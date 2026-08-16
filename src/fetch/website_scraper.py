from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WebsiteScraper:
    """Scrapes a business website: products/services, positioning, tone,
    value propositions, and key pages (about, blog, contact)."""

    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    TIMEOUT = 30
    MAX_PAGES = 12

    def scrape(self, url: str) -> dict[str, Any]:
        base = self._normalize_url(url)
        if not base:
            raise ValueError(f"Invalid website URL: {url!r}")

        pages = self._crawl(base)
        if not pages:
            raise RuntimeError(f"No pages could be fetched from {base}")

        homepage_html = pages[0]["html"]
        soup = BeautifulSoup(homepage_html, "lxml")

        title = self._clean(soup.title.get_text() if soup.title else "")
        meta_description = self._clean(
            soup.find("meta", attrs={"name": "description"}).get("content", "")
            if soup.find("meta", attrs={"name": "description"})
            else ""
        )

        headings = [
            self._clean(h.get_text())
            for h in soup.find_all(["h1", "h2", "h3"])[:30]
            if self._clean(h.get_text())
        ]

        text_blocks = [self._clean(p.get_text()) for p in soup.find_all("p")[:60]]
        text_blocks = [t for t in text_blocks if len(t) > 20]

        links = self._extract_links(pages)

        all_text = " ".join(
            self._clean(p["html_text"]) for p in pages if p.get("html_text")
        )

        return {
            "url": base,
            "title": title,
            "meta_description": meta_description,
            "headings": headings,
            "text_blocks": text_blocks[:40],
            "links": links[:50],
            "page_count": len(pages),
            "pages": [{"url": p["url"], "title": p["title"]} for p in pages[:10]],
            "full_text_sample": all_text[:15000],
            "scraped_at": None,
        }

    def _crawl(self, base: str) -> list[dict[str, Any]]:
        visited: set[str] = set()
        results: list[dict[str, Any]] = []
        queue = [base]

        while queue and len(results) < self.MAX_PAGES:
            page_url = queue.pop(0)
            if page_url in visited:
                continue
            visited.add(page_url)

            try:
                html = self._fetch(page_url)
            except Exception as exc:
                logger.warning("Failed to fetch %s: %s", page_url, exc)
                continue

            soup = BeautifulSoup(html, "lxml")
            title = self._clean(soup.title.get_text() if soup.title else "")
            text = " ".join(self._clean(p.get_text()) for p in soup.find_all("p"))
            results.append({"url": page_url, "html": html, "title": title, "html_text": text})

            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                full = urljoin(page_url, href)
                if self._same_domain(base, full) and full not in visited:
                    queue.append(full)
        return results

    def _fetch(self, url: str) -> str:
        with httpx.Client(
            headers={"User-Agent": self.USER_AGENT},
            timeout=self.TIMEOUT,
            follow_redirects=True,
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text

    def _normalize_url(self, url: str) -> str | None:
        if not url:
            return None
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        parsed = urlparse(url)
        if not parsed.netloc:
            return None
        return url

    def _same_domain(self, base: str, candidate: str) -> bool:
        return urlparse(base).netloc == urlparse(candidate).netloc

    def _extract_links(self, pages: list[dict[str, Any]]) -> list[str]:
        links: list[str] = []
        for page in pages:
            soup = BeautifulSoup(page["html"], "lxml")
            for a in soup.find_all("a", href=True):
                full = urljoin(page["url"], a["href"])
                if full not in links and self._same_domain(pages[0]["url"], full):
                    links.append(full)
        return links

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(text.split())