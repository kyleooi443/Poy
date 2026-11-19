"""Read/write data/manifest.csv, the committed record of every downloaded image."""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

from poy_ml.data_sources import ImageRecord

FIELDS = [f.name for f in dataclasses.fields(ImageRecord)]
REQUIRED = (
    "source_api",
    "image_url",
    "page_url",
    "title",
    "author",
    "license",
    "license_url",
    "date_downloaded",
    "label",
)


def write(path: Path, records: list[ImageRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for r in sorted(records, key=lambda r: (r.label, r.id)):
            w.writerow(dataclasses.asdict(r))


def read(path: Path) -> list[ImageRecord]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [ImageRecord(**row) for row in csv.DictReader(fh)]
