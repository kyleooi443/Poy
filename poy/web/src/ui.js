/** DOM wiring only. Rules live in game.js, features in features.js, models in classifier.js. */
import "./styles.css";
import { CameraError, startCamera } from "./camera.js";
import { ensembleProba, loadModel, Smoother, trainKnn } from "./classifier.js";
import { FEATURE_SPEC_VERSION, landmarksToFeatures } from "./features.js";
import { MOVES, isMove, newGame, playRound } from "./game.js";
import { createHandTracker } from "./handTracker.js";

/** @typedef {import("./classifier.js").Model} Model */
/** @typedef {import("./classifier.js").Sample} Sample */

const BASE = import.meta.env.BASE_URL;
const RECORD_MS = 2000;
const OUTCOME_TEXT = { win: "You win!", lose: "Computer wins.", draw: "Draw." };

/** @param {string} id */
const $ = (id) => /** @type {HTMLElement} */ (document.getElementById(id));
const video = /** @type {HTMLVideoElement} */ ($("video"));
const useBase = /** @type {HTMLInputElement} */ ($("use-base"));
const usePersonal = /** @type {HTMLInputElement} */ ($("use-personal"));

const state = {
  /** @type {MediaStream | null} */ stream: null,
  /** @type {Awaited<ReturnType<typeof createHandTracker>> | null} */ tracker: null,
  /** @type {Model | null} */ baseModel: null,
  /** @type {Model | null} */ personalModel: null,
  /** @type {Sample[]} */ samples: [],
  /** @type {{ label: string, until: number } | null} */ recording: null,
  smoother: new Smoother(),
  game: newGame(),
};

/** @param {string} msg */
const setStatus = (msg) => ($("status").textContent = msg);

function activeModels() {
  return /** @type {Model[]} */ (
    [useBase.checked && state.baseModel, usePersonal.checked && state.personalModel].filter(Boolean)
  );
}

// ---- screens -------------------------------------------------------------------------------
function showScreen() {
  const name = location.hash.slice(1) || "play";
  const known = ["play", "train", "about"].includes(name) ? name : "play";
  document.querySelectorAll("[data-screen]").forEach((el) => {
    /** @type {HTMLElement} */ (el).hidden = el.getAttribute("data-screen") !== known;
  });
  document.querySelectorAll("[data-nav]").forEach((a) => {
    if (a.getAttribute("data-nav") === known) a.setAttribute("aria-current", "page");
    else a.removeAttribute("aria-current");
  });
  $("camera-panel").hidden = known === "about";
}

// ---- camera + tracking loop ------------------------------------------------------------------
async function onStartCamera() {
  const btn = /** @type {HTMLButtonElement} */ ($("start-camera"));
  btn.disabled = true;
  setStatus("Starting camera…");
  try {
    state.stream ??= await startCamera(video);
    setStatus("Loading hand tracker…");
    state.tracker ??= await createHandTracker(BASE);
    setStatus(activeModels().length ? "" : "No gesture model loaded yet. Use Train on my hand.");
    btn.hidden = true;
    requestAnimationFrame(tick);
  } catch (err) {
    setStatus(
      err instanceof CameraError
        ? err.message
        : `Could not start: ${/** @type {Error} */ (err).message}`,
    );
    btn.disabled = false;
  }
}

/** @param {number} now */
function tick(now) {
  const landmarks = state.tracker?.detect(video, now) ?? null;
  const features = landmarks && landmarksToFeatures(landmarks);
  if (features && state.recording) {
    if (now < state.recording.until) {
      state.samples.push({ label: state.recording.label, features });
      updateCounts();
    } else stopRecording();
  } else if (state.recording && now >= state.recording.until) stopRecording();
  const models = activeModels();
  state.smoother.push(features && models.length ? ensembleProba(models, features) : null);
  renderSeeing(landmarks !== null, models.length > 0);
  requestAnimationFrame(tick);
}

/** @param {boolean} handNow @param {boolean} haveModel */
function renderSeeing(handNow, haveModel) {
  const cur = state.smoother.current();
  let text = "No hand visible";
  if (handNow && !haveModel) text = "Hand found, but no model. Train on my hand.";
  else if (cur)
    text = `Seeing: ${cur.label}${cur.sure ? "" : "?"} (${Math.round(cur.confidence * 100)}%)`;
  $("seeing").textContent = text;
}

// ---- play -----------------------------------------------------------------------------------
function onPlayRound() {
  const cur = state.smoother.current();
  if (!state.tracker) return setStatus("Start the camera first.");
  if (!cur) return setStatus("I can't see your hand. Hold it up to the camera.");
  if (!cur.sure || !isMove(cur.label))
    return setStatus("Not sure what that is. Hold rock, paper or scissors steady.");
  setStatus("");
  state.game = playRound(state.game, cur.label);
  const last = state.game.rounds.at(-1);
  if (last)
    $("result").textContent =
      `You: ${last.player}. Computer: ${last.computer}. ${OUTCOME_TEXT[last.outcome]}`;
  renderScore();
}

function renderScore() {
  $("score-player").textContent = String(state.game.score.player);
  $("score-computer").textContent = String(state.game.score.computer);
  $("score-draws").textContent = String(state.game.score.draws);
}

// ---- train on my hand ---------------------------------------------------------------------------
/** @param {string} label */
function startRecording(label) {
  if (!state.tracker) return setStatus("Start the camera first.");
  state.recording = { label, until: performance.now() + RECORD_MS };
  document.querySelectorAll("[data-record]").forEach((b) => {
    b.setAttribute("aria-pressed", String(b.getAttribute("data-record") === label));
  });
  setStatus(`Recording ${label}… keep your hand in view.`);
}

function stopRecording() {
  state.recording = null;
  document
    .querySelectorAll("[data-record]")
    .forEach((b) => b.setAttribute("aria-pressed", "false"));
  setStatus("Recorded. Record more, or press Train my model.");
}

function updateCounts() {
  for (const label of MOVES) {
    const el = document.querySelector(`[data-count="${label}"]`);
    if (el) el.textContent = String(state.samples.filter((s) => s.label === label).length);
  }
}

function onTrain() {
  const missing = MOVES.filter((m) => !state.samples.some((s) => s.label === m));
  if (missing.length) return setStatus(`Record some samples for: ${missing.join(", ")}.`);
  state.personalModel = trainKnn(state.samples, [...MOVES]);
  state.smoother.reset();
  setStatus(`Trained on ${state.samples.length} samples. Go to Play and try it.`);
}

function onDownload() {
  if (!state.samples.length) return setStatus("No samples recorded yet.");
  const doc = {
    format: "poy-samples",
    feature_spec_version: FEATURE_SPEC_VERSION,
    created: new Date().toISOString(),
    samples: state.samples,
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(doc)], { type: "application/json" }));
  const a = Object.assign(document.createElement("a"), {
    href: url,
    download: "handoff-samples.json",
  });
  a.click();
  URL.revokeObjectURL(url);
}

function onClear() {
  state.samples = [];
  state.personalModel = null;
  updateCounts();
  setStatus("Samples cleared.");
}

// ---- about --------------------------------------------------------------------------------
async function loadAbout() {
  try {
    const res = await fetch(`${BASE}attributions.json`);
    if (!res.ok) throw new Error(String(res.status));
    /** @type {{ title: string, author: string, license: string, license_url: string, page_url: string }[]} */
    const items = await res.json();
    const list = $("attributions");
    for (const it of items) {
      const li = document.createElement("li");
      const link = Object.assign(document.createElement("a"), {
        href: it.page_url,
        textContent: it.title,
      });
      const lic = Object.assign(document.createElement("a"), {
        href: it.license_url,
        textContent: it.license,
      });
      li.append(link, ` by ${it.author || "unknown"}, `, lic);
      list.append(li);
    }
    if (!items.length) list.textContent = "No dataset images were used.";
  } catch {
    $("attributions").textContent = "Attribution list not found (attributions.json).";
  }
}

// ---- boot -----------------------------------------------------------------------------------
async function boot() {
  showScreen();
  window.addEventListener("hashchange", showScreen);
  $("start-camera").addEventListener("click", onStartCamera);
  $("play-round").addEventListener("click", onPlayRound);
  $("reset-score").addEventListener("click", () => {
    state.game = newGame();
    $("result").textContent = "";
    renderScore();
  });
  document
    .querySelectorAll("[data-record]")
    .forEach((b) =>
      b.addEventListener("click", () =>
        startRecording(/** @type {string} */ (b.getAttribute("data-record"))),
      ),
    );
  $("train-knn").addEventListener("click", onTrain);
  $("download-samples").addEventListener("click", onDownload);
  $("clear-samples").addEventListener("click", onClear);
  document.addEventListener("keydown", (e) => {
    const t = /** @type {HTMLElement} */ (e.target);
    // Space on a focused control should activate that control, not play.
    if (e.code !== "Space" || e.repeat || t.closest("button, input, a, textarea, select")) return;
    if ($("screen-play").hidden) return;
    e.preventDefault();
    onPlayRound();
  });
  loadAbout();
  try {
    state.baseModel = await loadModel(`${BASE}models/model.json`);
    const acc = state.baseModel.metrics?.holdout?.accuracy;
    $("model-info").textContent =
      `Base model: ${state.baseModel.model.type}` +
      (acc != null ? `, held-out accuracy ${(acc * 100).toFixed(0)}%.` : ".");
  } catch (err) {
    useBase.checked = false;
    useBase.disabled = true;
    $("model-info").textContent = "Base model not available.";
    setStatus(`${/** @type {Error} */ (err).message}. You can still play after Train on my hand.`);
  }
}

boot();
