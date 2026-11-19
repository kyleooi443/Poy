# Handoff: simple documentation

Handoff is a rock-paper-scissors game you play with your webcam. A small machine-learning model
recognizes your hand gesture, and everything runs in the browser. It is my first ML project.
(The full details are in the [README](../README.md).)

## What it does

1. You show rock, paper or scissors to the webcam.
2. MediaPipe finds 21 points on your hand.
3. A small classifier turns those points into "rock", "paper" or "scissors".
4. You press **Play round** (or Space), the computer picks randomly, and the score updates.

No video is uploaded or stored. There is no server.

## The two parts

**1. The ML pipeline (Python, folder `ml/`)** runs once on a computer and produces one file,
`model.json`:

```
download images (official APIs only) -> find the hand (MediaPipe) -> 63 numbers per hand
-> train 3 models and compare -> evaluate -> export the best one to model.json
```

**2. The web app (JavaScript, folder `web/`)** loads `model.json` and does the same hand-to-numbers
step on every webcam frame, then predicts and plays the game.

They share two things: the **feature function** (hand points to 63 numbers, written in both Python
and JavaScript) and the **model file format**. A test checks that both languages give identical
results.

## The 63 numbers (features)

Take the 21 hand points (x, y, z). Move the wrist to the origin, then divide everything by the
largest wrist-to-point distance (using x and y). This makes the numbers independent of where the
hand is and how big it looks. Mirrored hands are added to the training data so left and right
hands both work.

## The screens

- **Play**: webcam, a live "Seeing: ..." label, the Play round button and the score.
- **Train on my hand**: record samples of each gesture, train a small k-nearest-neighbours model in
  the page, and download your samples as JSON. Samples stay in memory only.
- **About**: privacy note and image credits.

## Where the data comes from

Only official APIs (Openverse and Wikimedia Commons), never web scraping. Only CC0 and Public
Domain images are kept (CC BY is opt-in with `make data ALLOW_CC_BY=1`). Every image is listed in
`ml/data/manifest.csv`, and `ATTRIBUTION.md` is generated from it. Images are not committed.

## Results (honest version)

Only 77 usable images were found (27 rock, 15 paper, 35 scissors), far fewer than the ~150 per
class needed. The exported model scored **30% accuracy on held-out images, which is chance level**.
That is why the **Train on my hand** page exists: your own samples work much better than this base
model. I did not test with a real webcam; browser tests used a photo as a fake camera.

## How to run it

```bash
make setup   # install everything and download the hand model
make test    # run all checks (40 Python tests, 43 JavaScript tests)
make serve   # open http://localhost:5173
make data    # (optional) re-download images and rebuild the dataset
make train   # (optional) retrain and re-export model.json
```

Needs Python 3.12 (or `uv`), Node.js 20+, `make`, `curl` and a webcam.

## Folder map

```
README.md          full documentation
ATTRIBUTION.md     image credits (generated)
Makefile           the commands above
fixtures/          shared test data for the Python/JavaScript parity test
docs/              this file and the screenshots
ml/poy_ml/         the pipeline: data_sources, landmarks, features, train, evaluate, export
ml/data/           manifest.csv (committed); downloaded images (not committed)
ml/reports/        evaluation results
web/src/           camera, handTracker, features, classifier, game (rules), ui
web/public/models/ model.json and hand_landmarker.task
```

## Main files to read first

| To understand... | Read |
|---|---|
| the game rules | `web/src/game.js` (small, pure functions) |
| the feature contract | `ml/poy_ml/features.py` and `web/src/features.js` |
| how training works | `ml/poy_ml/train.py` |
| the model file | `ml/poy_ml/export.py` and `web/src/classifier.js` |
| how the page is wired | `web/src/ui.js` |
