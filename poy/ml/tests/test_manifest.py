import csv

import pytest

from poy_ml import attribution, config, manifest
from poy_ml.data_sources import ImageRecord


def make(i, **kw):
    base = dict(
        id=f"x-{i}",
        source_api="openverse",
        image_url=f"http://i/{i}",
        page_url=f"http://p/{i}",
        title=f"T|{i}",
        author="Ann",
        license="cc0",
        license_url="http://l",
        label="rock",
        date_downloaded="2026-11-19",
        status="kept",
    )
    return ImageRecord(**{**base, **kw})


def test_roundtrip(tmp_path):
    recs = [make(2, label="paper"), make(1, status="no_hand")]
    path = tmp_path / "m.csv"
    manifest.write(path, recs)
    back = manifest.read(path)
    assert sorted(back, key=lambda r: r.id) == sorted(recs, key=lambda r: r.id)
    with path.open() as fh:
        header = next(csv.reader(fh))
    assert set(manifest.REQUIRED) <= set(header)


def test_read_missing_is_empty(tmp_path):
    assert manifest.read(tmp_path / "nope.csv") == []


def test_attribution_lists_only_kept_and_escapes():
    md = attribution.render_markdown([make(1), make(2, status="no_hand"), make(3, license="by")])
    assert "x-2" not in md and "http://p/2" not in md
    assert "T\\|1" in md
    assert "1 of them are CC BY" in md
    assert len(attribution.render_json([make(1), make(2, status="multi_hand")])) == 1


@pytest.mark.skipif(not config.MANIFEST_PATH.exists(), reason="no committed manifest yet")
def test_committed_manifest_is_complete_and_licensed():
    recs = manifest.read(config.MANIFEST_PATH)
    assert recs, "manifest is empty"
    allowed = config.DEFAULT_LICENSES | config.CC_BY_LICENSES
    for r in recs:
        for f in manifest.REQUIRED:
            if f not in ("author", "title"):  # some public-domain works have no known author
                assert getattr(r, f), f"{r.id} is missing {f}"
        assert r.license in allowed, r.id
        assert r.label in config.LABELS
        assert r.status in ("kept", "no_hand", "multi_hand", "pending")
    assert len({r.image_url for r in recs}) == len(recs), "duplicate images"


@pytest.mark.skipif(not config.MANIFEST_PATH.exists(), reason="no committed manifest yet")
def test_attribution_file_is_up_to_date():
    expected = attribution.render_markdown(manifest.read(config.MANIFEST_PATH))
    assert config.ATTRIBUTION_MD.read_text(encoding="utf-8") == expected, "run make data (attribution)"
