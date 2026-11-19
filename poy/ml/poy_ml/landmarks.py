"""Step 2: run MediaPipe HandLandmarker (Tasks API) on every manifest image, keep only one-hand images.

Writes data/landmarks.npz (features, labels, groups, ids) and updates each manifest row's status.
Usage: python -m poy_ml.landmarks
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

from poy_ml import config, manifest
from poy_ml.data_sources import ImageRecord
from poy_ml.features import landmarks_to_features

# Returns one (21, 3) array per detected hand.
Detector = Callable[[str], list[np.ndarray]]


def mediapipe_detector() -> tuple[Detector, HandLandmarker]:
    if not config.HAND_LANDMARKER_PATH.exists():
        raise SystemExit(f"Missing {config.HAND_LANDMARKER_PATH}. Run `make setup` first.")
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(config.HAND_LANDMARKER_PATH)),
        running_mode=RunningMode.IMAGE,
        num_hands=2,  # so we can reject photos with several hands (ambiguous label)
        min_hand_detection_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    def detect(path: str) -> list[np.ndarray]:
        result = landmarker.detect(mp.Image.create_from_file(path))
        return [np.array([[p.x, p.y, p.z] for p in hand]) for hand in result.hand_landmarks]

    return detect, landmarker


def classify(hands: list[np.ndarray]) -> str:
    """Manifest status for an image given its detected hands."""
    return "kept" if len(hands) == 1 else ("no_hand" if not hands else "multi_hand")


def process(records: list[ImageRecord], detect: Detector) -> dict[str, np.ndarray]:
    """Detect hands, set statuses, delete files that are not kept. Returns the feature arrays."""
    feats, labels, groups, ids = [], [], [], []
    for rec in records:
        if rec.status not in ("pending", "kept"):
            continue
        path = config.DATA_DIR / rec.file
        if not path.exists():
            continue  # kept earlier but images were wiped; landmarks need the file
        hands = detect(str(path))
        rec.status = classify(hands)
        if rec.status != "kept":
            path.unlink(missing_ok=True)
            continue
        feats.append(landmarks_to_features(hands[0]))
        labels.append(rec.label)
        groups.append(rec.group)
        ids.append(rec.id)
    return {
        "X": np.array(feats).reshape(-1, 63),
        "y": np.array(labels, dtype=str),
        "groups": np.array(groups, dtype=str),
        "ids": np.array(ids, dtype=str),
    }


def report(records: list[ImageRecord]) -> str:
    lines = ["label     downloaded  had_hand  kept(one hand)"]
    for label in config.LABELS:
        rows = [r for r in records if r.label == label]
        st = Counter(r.status for r in rows)
        had = st["kept"] + st["multi_hand"]
        lines.append(f"{label:<9} {len(rows):>10}  {had:>8}  {st['kept']:>14}")
    return "\n".join(lines)


def main() -> None:
    records = manifest.read(config.MANIFEST_PATH)
    if not records:
        raise SystemExit("Manifest is empty. Run `make data` (acquire step) first.")
    detect, landmarker = mediapipe_detector()
    with landmarker:
        arrays = process(records, detect)
    manifest.write(config.MANIFEST_PATH, records)
    np.savez(config.LANDMARKS_PATH, **arrays)
    text = report(records)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "data_counts.txt").write_text(text + "\n")
    print(text)
    kept = Counter(arrays["y"].tolist())
    if min((kept[c] for c in config.LABELS), default=0) < 150:
        print(
            "\nWARNING: fewer than ~150 usable images in at least one class. The base model will be"
            "\nweak; use the 'Train on my hand' page in the web app (in-browser personalization)."
        )


if __name__ == "__main__":
    main()
