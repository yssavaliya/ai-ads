from __future__ import annotations

import ipaddress
import logging
import re
import socket
from typing import Any
from urllib.parse import urldefrag, urljoin, urlparse

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
    MAX_BYTES = 2_000_000  # per page; never slurp a huge download into memory
    SKIP_EXTENSIONS = (
        ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".zip", ".mp4",
        ".mp3", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".ico", ".css", ".js",
    )
    SOCIAL_HOSTS = ("instagram.com", "linkedin.com")

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
        meta_tag = soup.find("meta", attrs={"name": "description"})
        meta_description = self._clean(str(meta_tag.get("content", "")) if meta_tag else "")

        headings = [
            self._clean(h.get_text())
            for h in soup.find_all(["h1", "h2", "h3"])[:30]
            if self._clean(h.get_text())
        ]

        text_blocks = [self._clean(p.get_text()) for p in soup.find_all("p")[:60]]
        text_blocks = [t for t in text_blocks if len(t) > 20]

        links = self._extract_links(pages)
        social_links = self._extract_social_links(pages)
        contact_emails = self._extract_contact_emails(pages)

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
            "social_links": social_links,
            "contact_emails": contact_emails,
            "page_count": len(pages),
            "pages": [{"url": p["url"], "title": p["title"]} for p in pages[:10]],
            "full_text_sample": all_text[:15000],
            "scraped_at": None,
        }

    def _crawl(self, base: str) -> list[dict[str, Any]]:
        visited: set[str] = set()
        results: list[dict[str, Any]] = []
        queue = [urldefrag(base)[0]]
        queued: set[str] = set(queue)

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
                href = str(a["href"]).strip()
                full = urldefrag(urljoin(page_url, href))[0]
                if (
                    full.startswith(("http://", "https://"))
                    and not urlparse(full).path.lower().endswith(self.SKIP_EXTENSIONS)
                    and self._same_domain(base, full)
                    and full not in visited
                    and full not in queued
                ):
                    queued.add(full)
                    queue.append(full)
        return results

    MAX_REDIRECTS = 5

    @staticmethod
    def _resolve_public(url: str) -> str:
        """SSRF guard: URLs come from web search results and user input, so refuse anything that
        resolves to a loopback/private/link-local address (e.g. cloud metadata at 169.254.169.254).
        Returns the validated IP: the caller connects to THAT address, so a DNS answer that changes
        between this check and the request (DNS rebinding) cannot redirect the request inward."""
        host = urlparse(url).hostname
        if not host:
            raise ValueError(f"URL has no host: {url!r}")
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror as exc:
            raise ValueError(f"cannot resolve {host}") from exc
        addresses = [ipaddress.ip_address(str(info[4][0]).split("%")[0]) for info in infos]
        for ip in addresses:
            if not ip.is_global:
                raise ValueError(f"refusing non-public address {ip} for {host}")
        if not addresses:
            raise ValueError(f"cannot resolve {host}")
        return str(addresses[0])

    @classmethod
    def _assert_public_url(cls, url: str) -> None:
        cls._resolve_public(url)

    @staticmethod
    def _pin(url: str, ip: str) -> tuple[str, dict[str, str], dict[str, str]]:
        """The same request aimed at the validated IP: (url, headers, extensions). Host and TLS SNI /
        certificate checking keep using the original hostname."""
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if host == ip:
            return url, {}, {}  # already a literal address
        literal = f"[{ip}]" if ":" in ip else ip
        port = f":{parsed.port}" if parsed.port else ""
        pinned = parsed._replace(netloc=f"{literal}{port}").geturl()
        return pinned, {"Host": parsed.netloc.split("@")[-1]}, {"sni_hostname": host}

    def _fetch(self, url: str) -> str:
        # Redirects are followed by hand so every hop is checked (and pinned), not just the first.
        with httpx.Client(
            headers={"User-Agent": self.USER_AGENT},
            timeout=self.TIMEOUT,
            follow_redirects=False,
        ) as client:
            current = url
            for _ in range(self.MAX_REDIRECTS + 1):
                target, headers, extensions = self._pin(current, self._resolve_public(current))
                with client.stream("GET", target, headers=headers, extensions=extensions) as resp:
                    if resp.is_redirect and resp.headers.get("location"):
                        current = urljoin(current, resp.headers["location"])
                        continue
                    resp.raise_for_status()
                    ctype = resp.headers.get("content-type", "").lower()
                    if ctype and "html" not in ctype and "xml" not in ctype and "text" not in ctype:
                        raise ValueError(f"non-HTML content-type {ctype!r}")
                    body = b""
                    for chunk in resp.iter_bytes():
                        body += chunk
                        if len(body) >= self.MAX_BYTES:
                            break
                    encoding = resp.encoding or "utf-8"
                return body.decode(encoding, errors="replace")
        raise ValueError(f"too many redirects fetching {url}")

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

    @staticmethod
    def _host(url: str) -> str:
        host = str(urlparse(url).netloc).lower().split("@")[-1].split(":")[0]
        return host[4:] if host.startswith("www.") else host

    def _same_domain(self, base: str, candidate: str) -> bool:
        """example.com and www.example.com are the same site."""
        return self._host(base) == self._host(candidate)

    def _extract_social_links(self, pages: list[dict[str, Any]]) -> list[str]:
        """Outbound Instagram/LinkedIn links found on the crawled pages (the
        same-domain-only link list can never contain these)."""
        found: list[str] = []
        for page in pages:
            soup = BeautifulSoup(page["html"], "lxml")
            for a in soup.find_all("a", href=True):
                full = urljoin(page["url"], str(a["href"]).strip())
                host = self._host(full)
                if any(host == h or host.endswith("." + h) for h in self.SOCIAL_HOSTS) and full not in found:
                    found.append(full)
        return found[:20]

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
    def _extract_contact_emails(pages: list[dict[str, Any]]) -> list[str]:
        """Pull mailto: addresses (contact pages first)."""

        found: list[str] = []
        ordered = sorted(
            pages,
            key=lambda p: ("contact" not in (p.get("url") or "").lower(),),
        )
        mailto_re = re.compile(r"mailto:([^\"'?>\s]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})", re.I)
        for page in ordered:
            for match in mailto_re.findall(page.get("html", "")):
                email = match.strip().rstrip(".")
                if email.lower() not in [e.lower() for e in found]:
                    found.append(email)
        return found[:5]

    @staticmethod
    def _clean(text: str) -> str:
        return " ".join(text.split())
