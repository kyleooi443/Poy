// JavaScript half of the parity test; ml/tests/test_parity.py checks the same fixture in Python.
import { describe, expect, it } from "vitest";
import fixture from "../../fixtures/parity.json";
import { predictProba } from "../src/classifier.js";
import { landmarksToFeatures, mirrorFeatures } from "../src/features.js";

const TOL = 1e-12;
const close = (a, b) => a.forEach((v, i) => expect(Math.abs(v - b[i])).toBeLessThan(TOL));

describe.each(fixture.cases)("features: $name", (c) => {
  it("matches Python", () => {
    const f = landmarksToFeatures(c.landmarks);
    close(f, c.features);
    close(mirrorFeatures(f), c.mirrored);
  });
});

describe.each(["logistic_regression", "mlp"])("predictions: %s", (kind) => {
  it("matches Python probabilities and labels", () => {
    const exp = fixture[kind];
    fixture.cases.forEach((c, i) => {
      const p = predictProba(exp.model, landmarksToFeatures(c.landmarks));
      close(p, exp.proba[i]);
      expect(exp.model.labels[p.indexOf(Math.max(...p))]).toBe(exp.pred[i]);
    });
  });
});
