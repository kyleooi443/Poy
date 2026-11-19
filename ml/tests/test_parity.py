"""Python half of the parity test. web/tests/parity.test.js checks the same fixture in JavaScript."""

import json

import numpy as np
import pytest

from poy_ml import config, export
from poy_ml.features import landmarks_to_features, mirror_features

FIX = json.loads(config.PARITY_FIXTURE.read_text())


@pytest.mark.parametrize("case", FIX["cases"], ids=lambda c: c["name"])
def test_features_match_fixture(case):
    f = landmarks_to_features(case["landmarks"])
    assert np.allclose(f, case["features"], atol=1e-12, rtol=0)
    assert np.allclose(mirror_features(f), case["mirrored"], atol=1e-12, rtol=0)


@pytest.mark.parametrize("kind", ["logistic_regression", "mlp"])
def test_predictions_match_fixture(kind):
    exp = FIX[kind]
    X = np.array([landmarks_to_features(c["landmarks"]) for c in FIX["cases"]])
    proba = export.predict_proba(exp["model"], X)
    assert np.allclose(proba, exp["proba"], atol=1e-12, rtol=0)
    assert [exp["model"]["labels"][i] for i in proba.argmax(1)] == exp["pred"]
