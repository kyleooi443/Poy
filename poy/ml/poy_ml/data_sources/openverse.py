"""Openverse image search API client (https://api.openverse.org/v1/)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import requests

from poy_ml import config
from poy_ml.data_sources import ImageRecord, Throttle, canonical_url, normalize_license

API = "https://api.openverse.org/v1/images/"
OK_FILETYPES = {"jpg", "jpeg", "png", "webp", None}
_throttle = Throttle(config.OPENVERSE_MIN_INTERVAL_S)


def license_param(allowed: Iterable[str]) -> str:
    """Openverse's own license codes for our normalized set (pdm/cc0/by happen to match)."""
    return ",".join(sorted(allowed))


def parse_page(payload: dict[str, Any], label: str) -> list[ImageRecord]:
    """Turn one page of API results into records (no license filtering here)."""
    out = []
    for r in payload.get("results", []):
        if r.get("filetype") not in OK_FILETYPES or not r.get("url"):
            continue
        out.append(
            ImageRecord(
                id=f"openverse-{r['id']}",
                source_api="openverse",
                image_url=canonical_url(r["url"]),
                page_url=r.get("foreign_landing_url") or "",
                title=(r.get("title") or "").strip(),
                author=(r.get("creator") or "").strip(),
                license=normalize_license(r.get("license") or ""),
                license_url=r.get("license_url") or "",
                label=label,
            )
        )
    return out


def search(
    session: requests.Session, label: str, query: str, allowed: Iterable[str], cache_dir: Path
) -> Iterator[ImageRecord]:
    """Yield records for one query. Pages are cached on disk to save the 200/day anonymous quota."""
    for page in range(1, config.OPENVERSE_MAX_PAGES + 1):
        params = {"q": query, "license": license_param(allowed), "page_size": 20, "page": page}
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        cache = cache_dir / f"openverse_{key}.json"
        if cache.exists():
            payload = json.loads(cache.read_text())
        else:
            _throttle.wait()
            resp = session.get(API, params=params, timeout=30)
            if resp.status_code == 429:
                print(f"  openverse: rate limited ({resp.headers.get('retry-after', '?')}s); stopping")
                return
            if resp.status_code in (400, 401) and page > 1:
                return  # past the last page an anonymous user may see
            resp.raise_for_status()
            payload = resp.json()
            cache.write_text(json.dumps(payload))
        yield from parse_page(payload, label)
        if page >= int(payload.get("page_count") or 0):
            return
