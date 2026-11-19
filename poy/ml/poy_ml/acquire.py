"""Step 1: search the official APIs, keep allowed licenses, download images, write the manifest.

Usage: python -m poy_ml.acquire [--allow-cc-by] [--max-per-class N]
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import time
from collections import Counter
from collections.abc import Iterable
from urllib.parse import urlsplit

import requests
from PIL import Image

from poy_ml import config, manifest
from poy_ml.data_sources import (
    ImageRecord,
    Throttle,
    canonical_url,
    openverse,
    wikimedia,
    wikimedia_thumb,
)

SOURCES = (openverse, wikimedia)
MAX_SIDE = 1024
_download_throttle = Throttle(config.DOWNLOAD_MIN_INTERVAL_S)


def allowed_licenses(allow_cc_by: bool) -> frozenset[str]:
    return config.DEFAULT_LICENSES | (config.CC_BY_LICENSES if allow_cc_by else frozenset())


def filter_new(
    candidates: Iterable[ImageRecord], allowed: frozenset[str], seen_urls: set[str]
) -> Iterable[ImageRecord]:
    """Drop disallowed licenses and images already seen (under any label or source)."""
    for rec in candidates:
        if rec.license in allowed and rec.image_url not in seen_urls:
            seen_urls.add(rec.image_url)
            yield rec


class RateLimited(Exception):
    """A server kept answering 429 after we waited as asked; the caller stops using that host."""


def candidate_urls(rec: ImageRecord) -> list[str]:
    """Prefer a thumbnail; the original is only a fallback (e.g. originals smaller than the thumb)."""
    first = rec.download_url or wikimedia_thumb(rec.image_url) or rec.image_url
    return list(dict.fromkeys([first, rec.image_url]))


def _fetch(session: requests.Session, url: str, attempts: int = 3) -> bytes:
    """GET with the polite throttle. On 429, wait as long as the server asks, up to `attempts` times."""
    for attempt in range(attempts):
        _download_throttle.wait()
        resp = session.get(url, timeout=60, stream=True)
        if resp.status_code != 429:
            break
        wait = min(int(resp.headers.get("retry-after", "60") or 60), 600)
        if attempt == attempts - 1:
            raise RateLimited(f"{url} ({attempts} times in a row)")
        print(f"    429 from server, waiting {wait}s as asked")
        time.sleep(wait)
    resp.raise_for_status()
    data = bytearray()
    deadline = time.monotonic() + 90  # `timeout` is per read; a trickling server could stall forever
    for chunk in resp.iter_content(1 << 16):  # iter_content turns network errors into RequestException
        data += chunk
        if time.monotonic() > deadline:
            raise requests.Timeout(f"download took over 90 s: {url}")
        if len(data) > config.MAX_IMAGE_BYTES:
            break
    return bytes(data)


def download(session: requests.Session, rec: ImageRecord) -> bool:
    """Fetch, shrink to MAX_SIDE and save as JPEG. Returns False on failure; raises RateLimited."""
    try:
        urls = candidate_urls(rec)
        try:
            data = _fetch(session, urls[0])
        except requests.HTTPError:
            if len(urls) == 1:
                raise
            data = _fetch(session, urls[1])
        if len(data) > config.MAX_IMAGE_BYTES:
            return False
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except (requests.RequestException, OSError) as exc:
        print(f"    skip {rec.id}: {exc}")
        return False
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    rel = f"raw/{rec.label}/{rec.id}.jpg"
    (config.DATA_DIR / rel).parent.mkdir(parents=True, exist_ok=True)
    img.save(config.DATA_DIR / rel, "JPEG", quality=90)
    rec.file = rel
    rec.date_downloaded = dt.date.today().isoformat()
    rec.status = "pending"
    return True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-cc-by", action="store_true", help="also keep CC BY (needs attribution)")
    ap.add_argument("--max-per-class", type=int, default=400)
    args = ap.parse_args(argv)
    allowed = allowed_licenses(args.allow_cc_by)

    records = manifest.read(config.MANIFEST_PATH)
    for rec in records:  # honor a narrower license choice than the previous run
        if rec.license not in allowed and rec.file:
            (config.DATA_DIR / rec.file).unlink(missing_ok=True)
    records = [r for r in records if r.license in allowed]
    for r in records:
        r.image_url = canonical_url(r.image_url)
    known = {r.file for r in records}
    for path in config.RAW_DIR.glob("*/*.jpg"):  # files from an interrupted run without metadata
        if str(path.relative_to(config.DATA_DIR)) not in known:
            path.unlink()
    seen = {r.image_url for r in records}
    cache_dir = config.DATA_DIR / "api_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = config.USER_AGENT
    blocked: set[str] = set()  # hosts that kept answering 429: leave them alone for this run
    for label, queries in config.QUERIES.items():
        count = sum(r.label == label for r in records)
        for query in queries:
            for source in SOURCES:
                if count >= args.max_per_class:
                    break
                print(f"[{label}] {source.__name__.rsplit('.', 1)[-1]}: {query!r}")
                found = source.search(session, label, query, allowed, cache_dir)
                try:
                    for rec in filter_new(found, allowed, seen):
                        host = urlsplit(candidate_urls(rec)[0]).netloc
                        if count >= args.max_per_class:
                            break
                        if host in blocked:
                            seen.discard(rec.image_url)  # may be fetched on a later run
                            continue
                        try:
                            ok = download(session, rec)
                        except RateLimited as exc:
                            print(f"    rate limited: {exc}; skipping {host} for the rest of this run")
                            blocked.add(host)
                            seen.discard(rec.image_url)
                            continue
                        if ok:
                            records.append(rec)
                            count += 1
                            if count % 20 == 0:
                                manifest.write(config.MANIFEST_PATH, records)  # save progress
                except requests.RequestException as exc:  # search API hiccup: next query
                    print(f"    search failed: {exc}")
    if blocked:
        print(f"\nRate-limited hosts skipped: {sorted(blocked)}. Run `make data` again later to resume.")
    manifest.write(config.MANIFEST_PATH, records)
    print("downloaded per class:", dict(Counter(r.label for r in records)))
    print(f"manifest: {config.MANIFEST_PATH}")


if __name__ == "__main__":
    main()
