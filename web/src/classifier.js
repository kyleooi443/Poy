/**
 * Runs model.json (exported by ml/poy_ml/export.py) and the in-browser k-NN, plus smoothing.
 * Must stay numerically identical to export.predict_proba (see tests/parity.test.js).
 */
import { FEATURE_SPEC_VERSION, N_FEATURES, mirrorFeatures } from "./features.js";

/**
 * @typedef {{ type: "logistic_regression", coef: number[][], intercept: number[] }} LogReg
 * @typedef {{ type: "mlp", activation: "relu", layers: { weights: number[][], bias: number[] }[] }} Mlp
 * @typedef {{ type: "knn", k: number, X: number[][], y: number[] }} Knn
 * @typedef {{ labels: string[], model: LogReg | Mlp | Knn, feature_spec?: { version: number } }} Model
 * @typedef {{ label: string, features: number[] }} Sample
 */

export const MODEL_FORMAT = "poy-gesture-model";
export const MODEL_FORMAT_VERSION = 1;

/** @param {number[]} z @returns {number[]} */
export function softmax(z) {
  const m = Math.max(...z);
  const e = z.map((v) => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

/**
 * Check a parsed model.json. Throws with a readable message.
 * @param {any} doc @returns {Model}
 */
export function validateModel(doc) {
  if (doc?.format !== MODEL_FORMAT || doc.format_version !== MODEL_FORMAT_VERSION)
    throw new Error("model.json has an unknown format or version");
  if (doc.feature_spec?.version !== FEATURE_SPEC_VERSION)
    throw new Error("model.json was built for a different feature version");
  const m = doc.model;
  const nOut = doc.labels?.length;
  const ok =
    m?.type === "logistic_regression"
      ? m.coef.length === nOut && m.coef.every((r) => r.length === N_FEATURES)
      : m?.type === "mlp"
        ? m.layers[0].weights.length === N_FEATURES && m.layers.at(-1).bias.length === nOut
        : false;
  if (!ok) throw new Error("model.json weights do not match its labels/features");
  return doc;
}

/**
 * Fetch and validate the exported model.
 * @param {string} url @param {typeof fetch} [fetchFn] @returns {Promise<Model>}
 */
export async function loadModel(url, fetchFn = fetch) {
  const res = await fetchFn(url);
  if (!res.ok) throw new Error(`model file missing (${res.status} for ${url})`);
  return validateModel(await res.json());
}

/** @param {number[]} x @param {number[][]} w [n_in][n_out] @param {number[]} b */
const dense = (x, w, b) => b.map((bj, j) => x.reduce((acc, xi, i) => acc + xi * w[i][j], bj));

/** @param {Knn} m @param {number[]} x @param {number} nLabels */
function knnProba(m, x, nLabels) {
  const dists = m.X.map((row, i) => [row.reduce((a, v, j) => a + (v - x[j]) ** 2, 0), m.y[i]]);
  dists.sort((a, b) => a[0] - b[0]);
  const votes = new Array(nLabels).fill(0);
  const near = dists.slice(0, m.k);
  for (const [, y] of near) votes[y] += 1 / near.length;
  return votes;
}

/**
 * Class probabilities in the order of model.labels.
 * @param {Model} doc @param {number[]} x 63 features @returns {number[]}
 */
export function predictProba(doc, x) {
  const m = doc.model;
  if (m.type === "logistic_regression")
    return softmax(m.coef.map((row, c) => row.reduce((a, w, i) => a + w * x[i], m.intercept[c])));
  if (m.type === "mlp") {
    let h = x;
    m.layers.forEach((layer, i) => {
      h = dense(h, layer.weights, layer.bias);
      if (i < m.layers.length - 1) h = h.map((v) => Math.max(v, 0));
    });
    return softmax(h);
  }
  if (m.type === "knn") return knnProba(m, x, doc.labels.length);
  throw new Error(`unknown model type ${/** @type {any} */ (m).type}`);
}

/**
 * Probabilities keyed by label, averaged over any number of models (labels may be ordered differently).
 * @param {Model[]} models @param {number[]} x @returns {Record<string, number>}
 */
export function ensembleProba(models, x) {
  /** @type {Record<string, number>} */
  const out = {};
  for (const doc of models) {
    predictProba(doc, x).forEach((p, i) => {
      out[doc.labels[i]] = (out[doc.labels[i]] ?? 0) + p / models.length;
    });
  }
  return out;
}

/**
 * Train a k-NN on recorded samples, adding mirrored copies so both hands work.
 * @param {Sample[]} samples @param {string[]} labels @param {number} [k] @returns {Model}
 */
export function trainKnn(samples, labels, k = 5) {
  const usable = samples.filter(
    (s) => labels.includes(s.label) && s.features.length === N_FEATURES,
  );
  if (!usable.length) throw new Error("no samples to train on");
  const X = usable.flatMap((s) => [s.features, mirrorFeatures(s.features)]);
  const y = usable.flatMap((s) => [labels.indexOf(s.label), labels.indexOf(s.label)]);
  return { labels: [...labels], model: { type: "knn", k: Math.min(k, X.length), X, y } };
}

/**
 * Averages the last `size` predictions. Frames without a hand count as "nothing" so the label
 * disappears quickly when the hand leaves.
 */
export class Smoother {
  /** @param {number} [size] @param {number} [minConfidence] */
  constructor(size = 8, minConfidence = 0.55) {
    this.size = size;
    this.minConfidence = minConfidence;
    /** @type {(Record<string, number> | null)[]} */
    this.frames = [];
  }

  /** @param {Record<string, number> | null} proba null when no hand is visible */
  push(proba) {
    this.frames.push(proba);
    if (this.frames.length > this.size) this.frames.shift();
  }

  reset() {
    this.frames = [];
  }

  /** null = no hand in most recent frames; `sure` = confident enough to play.
   * @returns {{ label: string, confidence: number, sure: boolean } | null} */
  current() {
    const seen = this.frames.filter((f) => f !== null);
    if (seen.length * 2 <= this.frames.length || !seen.length) return null; // hand missing in most frames
    /** @type {Record<string, number>} */
    const avg = {};
    for (const f of seen)
      for (const [k, v] of Object.entries(f)) avg[k] = (avg[k] ?? 0) + v / seen.length;
    const [label, confidence] = Object.entries(avg).sort((a, b) => b[1] - a[1])[0];
    return { label, confidence, sure: confidence >= this.minConfidence };
  }
}
