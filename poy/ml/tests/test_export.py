import json

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from poy_ml import config, export, train

LABELS = list(config.LABELS)


def toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array(LABELS * (n // 3))
    X = rng.normal(size=(n, 63))
    X[np.arange(n), np.arange(n) % 3] += 3
    return X, y


@pytest.mark.parametrize(
    "model",
    [
        LogisticRegression(max_iter=2000),
        MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=0),
    ],
)
def test_json_inference_matches_sklearn(model, tmp_path):
    X, y = toy()
    model.fit(X, y)
    doc = export.model_to_dict(model, {"note": 1})
    path = tmp_path / "model.json"
    export.save(doc, path)
    loaded = json.loads(path.read_text())
    export.validate(loaded)
    assert loaded["labels"] == list(model.classes_)
    assert loaded["feature_spec"]["n_features"] == 63
    assert loaded["format_version"] == config.MODEL_FORMAT_VERSION
    assert np.allclose(export.predict_proba(loaded, X), model.predict_proba(X), atol=1e-9)


def test_unexportable_model_rejected():
    X, y = toy()
    with pytest.raises(TypeError):
        export.model_to_dict(RandomForestClassifier(n_estimators=3).fit(X, y), {})


def test_validate_rejects_bad_shapes():
    X, y = toy()
    doc = export.model_to_dict(LogisticRegression(max_iter=2000).fit(X, y), {})
    doc["model"]["coef"] = doc["model"]["coef"][:2]
    with pytest.raises(ValueError):
        export.validate(doc)


def test_shipped_model_is_valid():
    if not config.MODEL_OUT.exists():
        pytest.skip("no model.json yet")
    doc = json.loads(config.MODEL_OUT.read_text())
    export.validate(doc)
    assert sorted(doc["labels"]) == sorted(LABELS)
    assert doc["metrics"]["holdout"]["per_class"].keys() == set(LABELS)


def test_train_run_splits_by_group_and_reports():
    X, y = toy(120)
    groups = np.array([f"g{i % 20}" for i in range(120)])
    tr, te = train.grouped_split(y, groups)
    assert not set(groups[tr]) & set(groups[te])
    doc, report = train.run(X, y, groups)
    export.validate(doc)
    assert report["selected"] in train.EXPORTABLE
    assert set(report["test"]) == set(train.CANDIDATES)
    cm = np.array(report["holdout"]["confusion_matrix"])
    assert cm.shape == (3, 3) and cm.sum() == report["holdout"]["n_test"]


def test_train_refuses_tiny_dataset():
    X, y = toy(6)
    with pytest.raises(SystemExit):
        train.run(X, y, np.array([str(i) for i in range(6)]))
