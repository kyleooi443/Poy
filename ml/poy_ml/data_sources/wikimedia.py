"""Wikimedia Commons MediaWiki Action API client (https://commons.wikimedia.org/w/api.php)."""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import requests

from poy_ml import config
from poy_ml.data_sources import ImageRecord, Throttle, canonical_url, normalize_license

API = "https://commons.wikimedia.org/w/api.php"
THUMB_WIDTH = 800
MAX_PAGES = 4  # 50 results per page
_throttle = Throttle(config.WIKIMEDIA_MIN_INTERVAL_S)


def _meta(info: dict[str, Any], key: str) -> str:
    raw = str(info.get("extmetadata", {}).get(key, {}).get("value", ""))
    return html.unescape(re.sub(r"<[^>]+>", "", raw)).strip()  # Artist is often an HTML link


def parse_page(payload: dict[str, Any], label: str) -> list[ImageRecord]:
    """Turn one API response into records (no license filtering here)."""
    out = []
    for page in payload.get("query", {}).get("pages", {}).values():
        infos = page.get("imageinfo") or []
        if not infos or not infos[0].get("mime", "").startswith(("image/jpeg", "image/png")):
            continue
        info = infos[0]
        out.append(
            ImageRecord(
                id=f"wikimedia-{page['pageid']}",
                source_api="wikimedia_commons",
                image_url=canonical_url(info["url"]),
                page_url=canonical_url(info.get("descriptionurl", "")),
                title=page.get("title", "").removeprefix("File:"),
                author=_meta(info, "Artist"),
                license=normalize_license(_meta(info, "License") or _meta(info, "LicenseShortName")),
                # PD files often have no license URL; the file page holds the PD rationale.
                license_url=_meta(info, "LicenseUrl") or canonical_url(info.get("descriptionurl", "")),
                label=label,
                download_url=info.get("thumburl", ""),
            )
        )
    return out


def search(
    session: requests.Session, label: str, query: str, allowed: Iterable[str], cache_dir: Path
) -> Iterator[ImageRecord]:
    """Yield records for one full-text file search. Responses are cached on disk."""
    del allowed  # Commons cannot filter by license server-side; the caller filters.
    cont: dict[str, str] = {}
    for _ in range(MAX_PAGES):
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "1",
            "maxlag": "5",
            "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap",
            "gsrnamespace": "6",
            "gsrlimit": "50",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime",
            "iiextmetadatafilter": "License|LicenseShortName|LicenseUrl|Artist",
            "iiurlwidth": str(THUMB_WIDTH),
            **cont,
        }
        key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
        cache = cache_dir / f"wikimedia_{key}.json"
        if cache.exists():
            payload = json.loads(cache.read_text())
        else:
            _throttle.wait()
            resp = session.get(API, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            if "error" in payload:  # e.g. maxlag: back off and stop rather than hammer the API
                print(f"  wikimedia: {payload['error'].get('code')}; stopping this query")
                return
            cache.write_text(json.dumps(payload))
        yield from parse_page(payload, label)
        if "continue" not in payload:
            return
        cont = {k: str(v) for k, v in payload["continue"].items()}
