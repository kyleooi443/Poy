import { describe, expect, it } from "vitest";
import fixture from "../../fixtures/parity.json";
import {
  Smoother,
  ensembleProba,
  loadModel,
  predictProba,
  softmax,
  trainKnn,
  validateModel,
} from "../src/classifier.js";

const lr = fixture.logistic_regression.model;
const sum = (a) => a.reduce((x, y) => x + y, 0);

it("softmax is stable for large inputs", () => {
  const p = softmax([1000, 1001, 999]);
  expect(sum(p)).toBeCloseTo(1, 12);
  expect(p.every(Number.isFinite)).toBe(true);
});

describe("validateModel / loadModel", () => {
  it("accepts exported models", () => {
    expect(validateModel(lr)).toBe(lr);
    expect(validateModel(fixture.mlp.model)).toBe(fixture.mlp.model);
  });
  it("rejects wrong format, feature version and shapes", () => {
    expect(() => validateModel({ ...lr, format: "x" })).toThrow(/format/);
    expect(() => validateModel({ ...lr, feature_spec: { version: 99 } })).toThrow(/feature/);
    expect(() => validateModel({ ...lr, labels: ["a"] })).toThrow(/weights/);
  });
  it("reports a missing file clearly", async () => {
    const fetch404 = async () => ({ ok: false, status: 404 });
    await expect(loadModel("models/model.json", fetch404)).rejects.toThrow(/missing/);
  });
  it("loads a valid file", async () => {
    const fetchOk = async () => ({ ok: true, json: async () => lr });
    await expect(loadModel("m.json", fetchOk)).resolves.toBe(lr);
  });
});

describe("k-NN personalization", () => {
  const labels = ["rock", "paper", "scissors"];
  const sample = (label, v) => ({
    label,
    features: Array.from({ length: 63 }, (_, i) => (i === 4 ? v : 0)),
  });
  const samples = [
    sample("rock", 0),
    sample("rock", 0.1),
    sample("paper", 5),
    sample("scissors", -5),
  ];

  it("trains with mirrored copies and predicts the nearest class", () => {
    const m = trainKnn(samples, labels, 2);
    expect(m.model.X).toHaveLength(8);
    const p = predictProba(m, sample("?", 0.05).features);
    expect(labels[p.indexOf(Math.max(...p))]).toBe("rock");
    expect(sum(p)).toBeCloseTo(1, 12);
  });

  it("refuses empty input and ignores bad samples", () => {
    expect(() => trainKnn([], labels)).toThrow();
    expect(() => trainKnn([{ label: "rock", features: [1, 2] }], labels)).toThrow();
  });

  it("ensemble averages models by label name", () => {
    const knn = trainKnn(samples, labels, 1);
    const swapped = trainKnn(samples, ["scissors", "paper", "rock"], 1);
    const out = ensembleProba([knn, swapped], sample("?", 0).features);
    expect(out.rock).toBeCloseTo(1, 12);
    expect(sum(Object.values(out))).toBeCloseTo(1, 12);
  });
});

describe("Smoother", () => {
  it("averages recent frames and forgets old ones", () => {
    const s = new Smoother(3, 0.6);
    s.push({ rock: 1, paper: 0 });
    s.push({ rock: 0, paper: 1 });
    s.push({ rock: 0, paper: 1 });
    expect(s.current()).toMatchObject({ label: "paper", sure: true });
    s.push({ rock: 1, paper: 0 }); // window is now paper, paper, rock
    expect(s.current().confidence).toBeCloseTo(2 / 3, 12);
  });
  it("reports no hand when most frames had none, and unsure when split", () => {
    const s = new Smoother(4, 0.6);
    s.push({ rock: 0.5, paper: 0.5 });
    expect(s.current().sure).toBe(false);
    s.push(null);
    expect(s.current()).toBeNull();
  });
});
