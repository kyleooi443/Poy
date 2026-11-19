import json
from pathlib import Path

import numpy as np

from poy_ml import acquire, landmarks
from poy_ml.data_sources import ImageRecord, normalize_license, openverse, wikimedia

FIX = Path(__file__).parent / "fixtures"


def rec(i, license="cc0", url=None, label="rock", status="pending", file=""):
    return ImageRecord(
        id=f"t-{i}",
        source_api="test",
        image_url=url or f"http://x/{i}.jpg",
        page_url="p",
        title="t",
        author="a",
        license=license,
        license_url="",
        label=label,
        status=status,
        file=file,
    )


def test_normalize_license():
    assert normalize_license("cc0") == "cc0"
    assert normalize_license("pdm") == "pdm"
    assert normalize_license("pd") == "pdm"
    assert normalize_license("by") == "by"
    assert normalize_license("cc-by-4.0") == "by"
    assert normalize_license("cc-by-sa-4.0") == "other"
    assert normalize_license("by-sa") == "other"
    assert normalize_license("by-nc") == "other"
    assert normalize_license("") == "other"


def test_allowed_licenses_cc_by_is_opt_in():
    assert acquire.allowed_licenses(False) == {"cc0", "pdm"}
    assert acquire.allowed_licenses(True) == {"cc0", "pdm", "by"}


def test_filter_drops_disallowed_and_duplicates():
    seen = {"http://x/old.jpg"}
    cands = [
        rec(1),
        rec(2, "by"),
        rec(3, "other"),
        rec(4, url="http://x/old.jpg"),
        rec(5, url="http://x/1.jpg"),
    ]
    kept = list(acquire.filter_new(cands, acquire.allowed_licenses(False), seen))
    assert [r.id for r in kept] == ["t-1"]
    kept_by = list(acquire.filter_new([rec(2, "by")], acquire.allowed_licenses(True), set()))
    assert [r.id for r in kept_by] == ["t-2"]


def test_openverse_parse():
    recs = openverse.parse_page(json.loads((FIX / "openverse_page.json").read_text()), "rock")
    assert len(recs) == 1  # the svg is skipped
    r = recs[0]
    assert (r.source_api, r.license, r.author, r.label) == ("openverse", "cc0", "D. Benjamin Miller", "rock")
    assert r.image_url.startswith("https://upload.wikimedia.org/")


def test_wikimedia_parse_strips_html_and_tracking():
    recs = wikimedia.parse_page(json.loads((FIX / "wikimedia_page.json").read_text()), "paper")
    by_id = {r.id: r for r in recs}
    pd = by_id["wikimedia-1"]
    assert pd.license == "pdm" and pd.author == "Some Author"
    assert "?" not in pd.image_url and pd.license_url == "https://commons.wikimedia.org/wiki/File:A.jpg"
    assert pd.download_url.endswith("thumb.jpg?x=1")
    assert by_id["wikimedia-2"].license == "other"  # cc-by-sa


def test_hand_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(landmarks.config, "DATA_DIR", tmp_path)
    records = []
    for i in range(4):
        (tmp_path / f"{i}.jpg").write_bytes(b"x")
        records.append(rec(i, file=f"{i}.jpg", label=("rock", "paper", "scissors", "rock")[i]))
    records.append(rec(9, status="no_hand"))  # already rejected earlier: skipped
    one = np.random.default_rng(0).uniform(size=(21, 3))
    fake = {"0.jpg": [one], "1.jpg": [], "2.jpg": [one, one], "3.jpg": [one]}

    out = landmarks.process(records, lambda p: fake[Path(p).name])

    assert [r.status for r in records[:4]] == ["kept", "no_hand", "multi_hand", "kept"]
    assert out["X"].shape == (2, 63) and list(out["y"]) == ["rock", "rock"]
    assert not (tmp_path / "1.jpg").exists() and not (tmp_path / "2.jpg").exists()  # rejected files deleted
    assert (tmp_path / "0.jpg").exists()
    assert "rock" in landmarks.report(records)


def test_group_key_uses_author_then_id():
    r = rec(1)
    assert r.group == "test:a"
    r.author = ""
    assert r.group == "test:t-1"


def test_candidate_urls_prefer_wikimedia_thumbnails():
    r = rec(1, url="https://upload.wikimedia.org/wikipedia/commons/3/3d/A_b.jpg")
    assert acquire.candidate_urls(r) == [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/A_b.jpg/960px-A_b.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/3/3d/A_b.jpg",
    ]
    assert acquire.candidate_urls(rec(2, url="https://flickr/x.jpg")) == ["https://flickr/x.jpg"]


def test_rate_limit_waits_then_gives_up(monkeypatch):
    class Resp:
        status_code = 429
        headers = {"retry-after": "60"}

    calls = []
    session = type("S", (), {"get": lambda self, url, **kw: calls.append(url) or Resp()})()
    monkeypatch.setattr(acquire._download_throttle, "interval", 0)
    monkeypatch.setattr(acquire.time, "sleep", lambda s: calls.append(f"sleep {s}"))
    import pytest

    with pytest.raises(acquire.RateLimited):
        acquire.download(session, rec(1, url="https://upload.wikimedia.org/wikipedia/commons/3/3d/A.jpg"))
    assert calls[1:] == ["sleep 60", calls[2], "sleep 60", calls[4]]  # waits as asked, then gives up
    assert len(calls) == 5
