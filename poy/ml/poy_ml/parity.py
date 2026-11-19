"""Generate fixtures/parity.json: inputs plus Python's outputs, checked by both test suites.

Python's test recomputes everything from the inputs; web/tests/parity.test.js does the same in
JavaScript. Both must match the stored numbers, which proves the two implementations agree.
Usage: python -m poy_ml.parity
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from poy_ml import config, export
from poy_ml.features import landmarks_to_features, mirror_features


def _cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    cases = []
    for i in range(8):
        wrist = rng.uniform(0.2, 0.8, size=3)
        pts = wrist + rng.normal(0, 0.12, size=(21, 3))
        pts[0] = wrist
        cases.append({"name": f"random_{i}", "landmarks": pts.round(6).tolist()})
    cases.append({"name": "all_points_equal", "landmarks": [[0.5, 0.5, 0.0]] * 21})
    flat = rng.uniform(0, 1, size=(21, 3))
    flat[:, 1] = flat[0, 1]  # every point at the wrist's height: scale comes from x only
    cases.append({"name": "horizontal_line", "landmarks": flat.round(6).tolist()})
    return cases


def _expected(doc: dict[str, Any], X: np.ndarray) -> dict[str, Any]:
    proba = export.predict_proba(doc, X)
    return {"model": doc, "proba": proba.tolist(), "pred": [doc["labels"][i] for i in proba.argmax(1)]}


def build() -> dict[str, Any]:
    rng = np.random.default_rng(42)
    cases = _cases(rng)
    for c in cases:
        f = landmarks_to_features(c["landmarks"])
        c["features"] = f.tolist()
        c["mirrored"] = mirror_features(f).tolist()
    X = np.array([c["features"] for c in cases])

    Xfit = rng.normal(size=(90, 63))
    yfit = np.array(config.LABELS * 30)
    Xfit[np.arange(90), np.arange(90) % 3] += 3  # make the toy classes separable
    lr = LogisticRegression(max_iter=2000).fit(Xfit, yfit)
    mlp = MLPClassifier(hidden_layer_sizes=(8, 5), max_iter=2000, random_state=0).fit(Xfit, yfit)
    return {
        "cases": cases,
        "logistic_regression": _expected(export.model_to_dict(lr, {}), X),
        "mlp": _expected(export.model_to_dict(mlp, {}), X),
    }


def main() -> None:
    config.PARITY_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    config.PARITY_FIXTURE.write_text(json.dumps(build(), indent=1))
    print(f"wrote {config.PARITY_FIXTURE}")


if __name__ == "__main__":
    main()
