"""Step 3: compare models on a source-grouped split, evaluate the best exportable one, export model.json.

Split: images by the same author/source never appear in both train and test (GroupShuffleSplit),
so near-duplicate photo series cannot inflate accuracy. Mirrored copies are added on both sides
(after splitting) so left and right hands are covered.
Usage: python -m poy_ml.train
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPClassifier

from poy_ml import config, export
from poy_ml.evaluate import evaluate, format_report
from poy_ml.features import mirror_features

SEED = 0
MIN_PER_CLASS = 5
CANDIDATES: dict[str, ClassifierMixin] = {
    "logistic_regression": LogisticRegression(C=1.0, max_iter=5000),
    "mlp": MLPClassifier(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=3000, random_state=SEED),
    "random_forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
}
EXPORTABLE = ("logistic_regression", "mlp")


def augment(X: npt.NDArray, y: npt.NDArray, g: npt.NDArray) -> tuple[npt.NDArray, ...]:
    return np.vstack([X, mirror_features(X)]), np.concatenate([y, y]), np.concatenate([g, g])


def grouped_split(
    y: npt.NDArray, groups: npt.NDArray, test_size: float = 0.25
) -> tuple[npt.NDArray, npt.NDArray]:
    split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    train_idx, test_idx = next(split.split(np.zeros(len(y)), y, groups))
    return train_idx, test_idx


def cv_macro_f1(model: ClassifierMixin, X: npt.NDArray, y: npt.NDArray, g: npt.NDArray) -> float:
    """Grouped CV on the training part only (the test split stays untouched for the final report)."""
    k = min(5, len(set(g)))
    scores = []
    for tr, va in GroupKFold(n_splits=k).split(X, y, g):
        Xa, ya, _ = augment(X[tr], y[tr], g[tr])
        m = clone(model).fit(Xa, ya)
        scores.append(
            f1_score(
                np.concatenate([y[va], y[va]]), m.predict(augment(X[va], y[va], g[va])[0]), average="macro"
            )
        )
    return float(np.mean(scores))


def run(X: npt.NDArray, y: npt.NDArray, groups: npt.NDArray) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (model.json document, full report)."""
    counts = Counter(y.tolist())
    if any(counts[c] < MIN_PER_CLASS for c in config.LABELS):
        raise SystemExit(f"Too little data to train: {dict(counts)} (need {MIN_PER_CLASS}+ per class).")
    tr, te = grouped_split(y, groups)
    cv = {name: cv_macro_f1(m, X[tr], y[tr], groups[tr]) for name, m in CANDIDATES.items()}
    best = max(EXPORTABLE, key=lambda n: cv[n])

    Xtr, ytr, _ = augment(X[tr], y[tr], groups[tr])
    Xte, yte, _ = augment(X[te], y[te], groups[te])
    labels = list(config.LABELS)
    test = {
        name: evaluate(yte, clone(m).fit(Xtr, ytr).predict(Xte), labels) for name, m in CANDIDATES.items()
    }

    # Final export: same config refit on all data; metrics above are from the held-out split.
    Xall, yall, _ = augment(X, y, groups)
    final = clone(CANDIDATES[best]).fit(Xall, yall)
    metrics = {
        "selected": best,
        "holdout": test[best],
        "split": "GroupShuffleSplit by source_api:author, 25% test; mirrored copies on both sides",
        "n_images": {c: counts[c] for c in labels},
        "n_groups": {"train": len(set(groups[tr])), "test": len(set(groups[te]))},
    }
    report = {"cv_macro_f1": cv, "test": test, **metrics}
    return export.model_to_dict(final, metrics), report


def main() -> None:
    if not config.LANDMARKS_PATH.exists():
        raise SystemExit("No landmarks yet. Run `make data` first.")
    d = np.load(config.LANDMARKS_PATH)
    doc, report = run(d["X"], d["y"], d["groups"])
    export.save(doc)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "eval.json").write_text(json.dumps(report, indent=2))
    lines = ["model comparison (grouped CV macro-F1 on train | holdout macro-F1):"]
    for name in CANDIDATES:
        flag = "  <- exported" if name == report["selected"] else ""
        lines.append(
            f"  {name:<20} {report['cv_macro_f1'][name]:.3f} | {report['test'][name]['macro_f1']:.3f}{flag}"
        )
    lines += ["", f"holdout report for {report['selected']}:", format_report(report["holdout"])]
    text = "\n".join(lines)
    (config.REPORTS_DIR / "eval.txt").write_text(text + "\n")
    print(text)
    print(f"\nwrote {config.MODEL_OUT}")


if __name__ == "__main__":
    main()
