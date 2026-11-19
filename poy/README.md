# Handoff: webcam rock-paper-scissors

Show rock, paper or scissors to your webcam, press **Play round** (or <kbd>Space</kbd>), and play
against the computer. Everything runs in your browser: **no video is uploaded or stored.**

```
ml/  (offline, Python)                         web/  (static site, plain JS + Vite)
acquire -> landmarks -> train -> export  --->  public/models/model.json  ---> classifier.js
  Openverse + Wikimedia Commons APIs            (the only file the app loads from the pipeline)
```

The two halves share one contract: the **feature function** (21 hand landmarks -> 63 numbers) in
`ml/poy_ml/features.py` and `web/src/features.js`, and the **model file** `model.json`. A parity
test with shared fixtures (`fixtures/parity.json`) proves both sides compute the same features and
the same predictions.

## Quick start (beginner friendly)

You need:

- **Python 3.12** (or [uv](https://docs.astral.sh/uv/), which `make setup` uses automatically if installed)
- **Node.js 20 or newer** (comes with `npm`)
- `make` and `curl` (preinstalled on macOS and most Linux; on Windows use WSL)
- A webcam and a recent Chrome, Edge, Firefox or Safari

Then, from this `poy/` folder:

```bash
make setup    # creates ml/.venv, installs pinned Python + npm packages, downloads the hand model
make test     # linters + all Python and JavaScript tests
make serve    # starts the site at http://localhost:5173
```

Open http://localhost:5173, press **Start camera**, allow camera access, and play.
The repository already contains a trained `web/public/models/model.json`, so you do not need to
download data to play. To rebuild the model yourself:

```bash
make data     # download images via the APIs, detect hands, write manifest.csv + ATTRIBUTION.md
make train    # compare models, evaluate on a held-out split, export model.json
```

`make data` takes a while on purpose: it waits between requests to respect API rate limits.
If a server rate-limits you, it skips that server for the run; run `make data` again later and it
resumes where it stopped (search results are cached in `ml/data/api_cache/`).
Set `POY_CONTACT=you@example.com` (or your repo URL) before `make data` so the User-Agent carries
contact information, as Wikimedia's robot policy asks.

### All commands

| Command | What it does |
|---|---|
| `make setup` | Python venv + pinned deps, `npm ci`, downloads `hand_landmarker.task` |
| `make data` | acquire images (CC0 / Public Domain only) -> run MediaPipe -> manifest + attribution |
| `make data ALLOW_CC_BY=1` | same, but also keeps CC BY images (credited in `ATTRIBUTION.md`) |
| `make train` | model comparison, evaluation report (`ml/reports/`), export `model.json` |
| `make test` | ruff + pytest, eslint + prettier + vitest (includes the parity test) |
| `make serve` | development server |
| `make build` | static site in `web/dist/` |
| `make fixtures` | regenerate `fixtures/parity.json` (only after changing the feature/model contract) |
| `cd web && npm run smoke` | optional browser smoke test in real headless Chrome (after `make build`) |

## Using the app

- **Play**: the label on the video says what the model sees (`Seeing: rock (87%)`; a `?` means
  "not sure"). Predictions are averaged over the last 8 frames. Press **Play round** or <kbd>Space</kbd>.
- **Train on my hand**: the dataset model is weak (see below), so this page lets you record your
  own samples. Press *Record rock* and hold the gesture for 2 seconds; repeat for paper and scissors
  (a few times each, both hands, different angles), then **Train my model**. A k-nearest-neighbours
  model is trained in the page (mirrored copies are added automatically). You can use it alone or
  averaged with the base model. **Download my samples** saves them as JSON on your computer.
  Samples live only in the tab's memory: nothing is sent anywhere.
- **About**: privacy note, base-model accuracy and image attributions.

Error messages are shown for: camera permission denied, no camera, camera busy, no hand visible,
missing `hand_landmarker.task`, missing or invalid `model.json` (you can still play with your own
trained model in that case).

## How it works

### Features (the shared contract)

For the 21 MediaPipe hand landmarks `(x, y, z)`: subtract the wrist (landmark 0) from every point,
then divide everything by the largest wrist-to-point distance measured in `x` and `y`. Result: 63
numbers, `x0,y0,z0,x1,...`. Mirroring a hand = negating every `x`, used for augmentation so both
hands work.

### Model file (`model.json`, format version 1)

```json
{
  "format": "poy-gesture-model", "format_version": 1, "created": "...",
  "labels": ["paper", "rock", "scissors"],
  "feature_spec": { "name": "wrist_origin_max_xy_distance", "n_features": 63, "version": 1, ... },
  "model": { "type": "logistic_regression", "coef": [[...63]...], "intercept": [...] }
        // or { "type": "mlp", "activation": "relu", "layers": [{ "weights": [in][out], "bias": [...] }] }
  "metrics": { "holdout": { "accuracy": ..., "per_class": {...}, "confusion_matrix": [...] }, ... }
}
```

### Data policy

- Only official APIs: the [Openverse API](https://api.openverse.org/v1/) and the
  [Wikimedia Commons API](https://commons.wikimedia.org/w/api.php). No web pages are scraped.
- Default licenses: **CC0** and **Public Domain** only. `ALLOW_CC_BY=1` also keeps **CC BY**.
  Share-alike, non-commercial and no-derivatives licenses are never kept.
- Requests are throttled (Openverse: 3.5 s apart, under the anonymous 20/minute limit;
  Commons API: 1 s; image downloads: 1 s, Wikimedia images as standard-size thumbnails as their
  policy asks). On HTTP 429 the client waits the server's `Retry-After`, and after three 429s in a
  row stops using that host for the run. A descriptive User-Agent is sent.
- Every downloaded image is recorded in `ml/data/manifest.csv` (source API, image URL, page URL,
  title, author, license, license URL, download date, label, and `status`:
  `kept` / `no_hand` / `multi_hand`). Images without exactly one detected hand are deleted.
- Images are **never committed** (`ml/data/` is git-ignored except the manifest).
  `ATTRIBUTION.md` and `web/public/attributions.json` are generated from the manifest.
- Labels come from the search query (e.g. "clenched fist" -> rock). They are weak labels.

### Training and evaluation

- Train/test split by **image source**: `GroupShuffleSplit` on `source_api:author`, so photos from
  the same author (often near-duplicate series) never appear on both sides. Model selection uses
  grouped 5-fold CV on the training part only; the test split is used once for the final report.
- Candidates: logistic regression, a small MLP (32 hidden units), random forest. The best of the
  ones that can run in plain JavaScript (logistic regression or MLP) is exported, refit on all data.
- Reports: `ml/reports/eval.txt` and `eval.json` (per-class precision, recall, F1, confusion
  matrix), `ml/reports/data_counts.txt` (downloaded / had a hand / kept per class).

## Results

Latest run (2026-11-19, CC0 + Public Domain only). Full numbers are in `ml/reports/`.

| Class | Downloaded | Had a hand | Kept (exactly one hand) |
|---|---|---|---|
| rock | 293 | 42 | 27 |
| paper | 400 | 25 | 15 |
| scissors | 298 | 58 | 35 |

**This is far below the ~150 usable images per class the base model needs.** Openly licensed
photos of rock-paper-scissors hands are rare, most search hits show no clear hand, and Wikimedia's
image server rate-limited this machine part-way through (those images were skipped, not forced).

Held-out evaluation (23 test images by unseen authors, 46 with mirrored copies):

| Model | Grouped CV macro-F1 (train) | Holdout macro-F1 |
|---|---|---|
| logistic regression (exported) | 0.35 | 0.25 |
| MLP (32) | 0.33 | 0.35 |
| random forest (not exportable) | 0.43 | 0.37 |

Exported model holdout accuracy: **0.30, which is chance level**. Per class: rock P 0.80 / R 0.33,
paper P 0.00 / R 0.00, scissors P 0.17 / R 1.00. With this little test data, these numbers are
also very noisy. In practice: **use "Train on my hand"**. A minute of your own samples is
much better data than photos found through the search APIs.

## Deploying (static site)

The site is the `web/` folder; `npm run build` writes `web/dist/`. Commit
`web/public/models/model.json` and `web/public/models/hand_landmarker.task` (both are already in
the repo) so the host can build without running the Python pipeline. The MediaPipe WebAssembly
files are copied from `node_modules` into `web/public/wasm/` by `npm install` (postinstall).

**Vercel**

1. Push the repository to GitHub, then in Vercel choose *Add New -> Project* and import it.
2. Set **Root Directory** to `poy/web`. Framework preset: *Vite* (build `npm run build`, output `dist`).
3. Deploy. The camera works because Vercel serves over HTTPS.

**GitHub Pages**

1. In the repository: *Settings -> Pages -> Source: GitHub Actions*.
2. Add `.github/workflows/pages.yml`:

```yaml
name: pages
on: { push: { branches: [main] } }
permissions: { contents: read, pages: write, id-token: write }
jobs:
  deploy:
    runs-on: ubuntu-latest
    environment: github-pages
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 22 }
      - run: npm ci && npm run build
        working-directory: poy/web
      - uses: actions/upload-pages-artifact@v3
        with: { path: poy/web/dist }
      - uses: actions/deploy-pages@v4
```

The build uses relative paths (`base: "./"`), so it works under `https://<user>.github.io/<repo>/`.

## Project layout

```
poy/
  ATTRIBUTION.md       generated from the manifest
  Makefile
  fixtures/parity.json shared Python/JS parity fixture
  ml/                  requirements.txt (pinned), pyproject.toml (ruff, pytest)
    poy_ml/            config, data_sources/{openverse,wikimedia}, acquire, manifest, landmarks,
                       features, train, evaluate, export, attribution, parity
    tests/             pytest
    data/manifest.csv  committed; images are git-ignored
    reports/           evaluation output
  web/
    index.html         markup only
    src/               camera, handTracker, features, classifier, game (pure rules), ui, styles
    public/models/     model.json, hand_landmarker.task
    tests/             vitest (game, features, classifier, parity, camera) + e2e/smoke.mjs
```

## Known limitations

- **The base model is not useful yet** (see Results). The in-browser k-NN on your own samples is the
  intended way to play. More data (re-running `make data` later, `ALLOW_CC_BY=1`, or more queries in
  `ml/poy_ml/config.py`) is the way to improve it.
- **Labels are weak**: they come from search queries, and some kept images show the wrong gesture
  (e.g. an open hand found by "fist"). Nobody reviewed the images by hand.
- Features use MediaPipe's image-normalized coordinates, so the image aspect ratio slightly distorts
  hand shape (webcam 4:3 versus photos of any shape). This is part of the shared spec. Fixing it
  means a new feature version on both sides.
- Python (IMAGE mode on photos) and the browser (VIDEO mode on a webcam) run the same
  `hand_landmarker.task`, but tracking in VIDEO mode can give slightly different landmarks.
- The k-NN keeps all samples in memory and compares against each one every frame. That's fine for
  a few thousand samples, but it would slow down far beyond that.
- Python type hints are not checked by a type checker (no mypy in `make test`). JSDoc types are
  for editors only.
- Only one hand is tracked in the browser at a time.
