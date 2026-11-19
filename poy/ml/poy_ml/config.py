"""Paths, labels and data-source settings shared by the whole pipeline."""

import os
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = ML_DIR.parent
DATA_DIR = ML_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.csv"
LANDMARKS_PATH = DATA_DIR / "landmarks.npz"
REPORTS_DIR = ML_DIR / "reports"
WEB_PUBLIC = ROOT_DIR / "web" / "public"
MODEL_OUT = WEB_PUBLIC / "models" / "model.json"
HAND_LANDMARKER_PATH = WEB_PUBLIC / "models" / "hand_landmarker.task"
ATTRIBUTION_MD = ROOT_DIR / "ATTRIBUTION.md"
ATTRIBUTION_JSON = WEB_PUBLIC / "attributions.json"
PARITY_FIXTURE = ROOT_DIR / "fixtures" / "parity.json"

LABELS: tuple[str, ...] = ("rock", "paper", "scissors")
FEATURE_SPEC_VERSION = 1
MODEL_FORMAT_VERSION = 1

# Search phrases per label. Labels are weak (taken from the query), see README "Known limitations".
QUERIES: dict[str, tuple[str, ...]] = {
    "rock": ("rock paper scissors rock hand", "fist", "clenched fist"),
    "paper": ("rock paper scissors paper hand", "open hand", "open palm"),
    "scissors": ("rock paper scissors scissors hand", "victory sign", "peace sign hand"),
}

# Licenses kept by default; CC BY only with --allow-cc-by (then ATTRIBUTION.md lists the authors).
DEFAULT_LICENSES = frozenset({"cc0", "pdm"})
CC_BY_LICENSES = frozenset({"by"})

# Wikimedia's policy asks bots for contact info: set POY_CONTACT to your email or repo URL.
USER_AGENT = (
    "HandoffRPS/0.1 (educational rock-paper-scissors gesture dataset"
    + (f"; {os.environ['POY_CONTACT']}" if os.environ.get("POY_CONTACT") else "")
    + ") python-requests"
)
OPENVERSE_MIN_INTERVAL_S = 3.5  # anonymous burst limit is 20 requests/minute
OPENVERSE_MAX_PAGES = 12  # anonymous users get at most 240 results (12 pages of 20)
WIKIMEDIA_MIN_INTERVAL_S = 1.0
DOWNLOAD_MIN_INTERVAL_S = 1.0
MAX_IMAGE_BYTES = 8_000_000
