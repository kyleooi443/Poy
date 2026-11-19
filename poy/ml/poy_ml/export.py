"""Model <-> model.json, the single versioned contract the web app loads.

Also contains a NumPy re-implementation of inference on the JSON, so tests can prove the exported
file reproduces sklearn exactly, and the parity fixture can be generated from the same code path.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from poy_ml import config
from poy_ml.features import FEATURE_SPEC

FORMAT = "poy-gesture-model"


def model_to_dict(model: LogisticRegression | MLPClassifier, metrics: dict[str, Any]) -> dict[str, Any]:
    if isinstance(model, LogisticRegression):
        body: dict[str, Any] = {
            "type": "logistic_regression",
            "coef": model.coef_.tolist(),  # [n_classes][63]
            "intercept": model.intercept_.tolist(),
        }
    elif isinstance(model, MLPClassifier):
        if model.activation != "relu" or model.out_activation_ != "softmax":
            raise ValueError("only relu MLPs with softmax output can be exported")
        body = {
            "type": "mlp",
            "activation": "relu",
            # weights[i] is [n_in][n_out]
            "layers": [
                {"weights": w.tolist(), "bias": b.tolist()}
                for w, b in zip(model.coefs_, model.intercepts_, strict=True)
            ],
        }
    else:
        raise TypeError(f"cannot export {type(model).__name__} to plain JavaScript")
    return {
        "format": FORMAT,
        "format_version": config.MODEL_FORMAT_VERSION,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "labels": [str(c) for c in model.classes_],  # output order of the probabilities
        "feature_spec": {**FEATURE_SPEC, "version": config.FEATURE_SPEC_VERSION},
        "model": body,
        "metrics": metrics,
    }


def validate(doc: dict[str, Any]) -> None:
    """Raise ValueError if doc is not a model file this code (and classifier.js) can run."""
    if doc.get("format") != FORMAT or doc.get("format_version") != config.MODEL_FORMAT_VERSION:
        raise ValueError("unknown model format or version")
    if doc["feature_spec"].get("version") != config.FEATURE_SPEC_VERSION:
        raise ValueError("feature spec version mismatch")
    n_in, n_out = doc["feature_spec"]["n_features"], len(doc["labels"])
    m = doc["model"]
    if m["type"] == "logistic_regression":
        shapes = [np.shape(m["coef"]) == (n_out, n_in), np.shape(m["intercept"]) == (n_out,)]
    elif m["type"] == "mlp":
        dims = [n_in] + [len(layer["bias"]) for layer in m["layers"]]
        shapes = [dims[-1] == n_out] + [
            np.shape(layer["weights"]) == (dims[i], dims[i + 1]) for i, layer in enumerate(m["layers"])
        ]
    else:
        raise ValueError(f"unknown model type {m['type']!r}")
    if not all(shapes):
        raise ValueError("weight shapes do not match feature_spec/labels")


def _softmax(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def predict_proba(doc: dict[str, Any], X: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Inference straight from the JSON document; mirrors web/src/classifier.js."""
    h = np.atleast_2d(np.asarray(X, dtype=np.float64))
    m = doc["model"]
    if m["type"] == "logistic_regression":
        return _softmax(h @ np.asarray(m["coef"]).T + np.asarray(m["intercept"]))
    layers = m["layers"]
    for i, layer in enumerate(layers):
        h = h @ np.asarray(layer["weights"]) + np.asarray(layer["bias"])
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    return _softmax(h)


def save(doc: dict[str, Any], path: Path = config.MODEL_OUT) -> None:
    validate(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, separators=(",", ":")))
