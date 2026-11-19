"""Shared types for the API clients. Each API lives in its own module."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass


@dataclass
class ImageRecord:
    """One candidate image plus everything the manifest needs about it."""

    id: str
    source_api: str
    image_url: str
    page_url: str
    title: str
    author: str
    license: str  # normalized: cc0 | pdm | by | other
    license_url: str
    label: str
    date_downloaded: str = ""
    file: str = ""
    status: str = "pending"  # pending | no_hand | multi_hand | kept | download_failed
    download_url: str = ""  # what we actually fetch (e.g. a resized thumbnail); image_url if empty

    @property
    def group(self) -> str:
        """Train/test split key: same author (or same image) never lands on both sides."""
        return f"{self.source_api}:{self.author.strip().lower() or self.id}"


def canonical_url(url: str) -> str:
    """Drop Wikimedia's utm_* tracking query so the same file dedupes across APIs."""
    return url.split("?", 1)[0] if "wikimedia.org/" in url else url


def wikimedia_thumb(url: str, width: int = 960) -> str | None:
    """Standard-size thumbnail URL for an upload.wikimedia.org original (Wikimedia asks bots to use these)."""
    m = re.fullmatch(
        r"(https://upload\.wikimedia\.org/wikipedia/[^/]+)/(\w/\w\w)/([^/?]+)", canonical_url(url)
    )
    return f"{m[1]}/thumb/{m[2]}/{m[3]}/{width}px-{m[3]}" if m else None


def normalize_license(raw: str) -> str:
    """Map API license strings to cc0 | pdm | by | other.

    Openverse uses e.g. "cc0", "pdm", "by", "by-sa".
    Wikimedia Commons uses e.g. "cc0", "pd", "cc-by-4.0", "cc-by-sa-3.0".
    """
    s = raw.strip().lower()
    if s in ("cc0", "cc-zero") or s.startswith("cc0"):
        return "cc0"
    if s in ("pdm", "pd", "public domain") or s.startswith("pd-"):
        return "pdm"
    if s == "by" or (s.startswith("cc-by-") and s[6:7].isdigit()):
        return "by"
    return "other"


class Throttle:
    """Blocks so that calls are at least `interval` seconds apart."""

    def __init__(self, interval: float) -> None:
        self.interval = interval
        self._last = 0.0

    def wait(self) -> None:
        delay = self._last + self.interval - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()
