# Poy: Webcam Rock-Paper-Scissors

Play rock-paper-scissors against the computer using your own hand and a webcam.

> **This is my first machine learning project.** I'm building it from scratch as part of my project-based learning, so expect rough edges. Feedback is welcome.

"Poy" comes from *jack-en-poy*, the Filipino name for the game.

## How it works

1. **OpenCV** reads frames from the webcam.
2. **MediaPipe Hand Landmarker** finds 21 points on the hand.
3. The points are normalized so hand position and size don't matter.
4. A **scikit-learn** random forest, trained on my own recorded samples, labels the hand as rock, paper or scissors.
5. The game picks a random move for the computer, decides the winner and keeps score.

## Requirements

- Python 3.10 or newer
- A webcam
- MediaPipe's `hand_landmarker.task` model file, saved in the project folder (see the MediaPipe Hand Landmarker guide for the download link)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install opencv-python mediapipe scikit-learn pandas joblib
```

## Usage

Run these from the project folder, in order:

| Step | Command | What it does |
| --- | --- | --- |
| 1 | `python hand_test.py` | Checks the camera and shows 21 dots on your hand (`q` to quit) |
| 2 | `python collect_data.py` | Records gesture samples: `r` rock, `p` paper, `s` scissors (`q` to quit) |
| 3 | `python train.py` | Trains the model and prints accuracy |
| 4 | `python play.py` | Plays the game: `SPACE` plays a round, `q` quits |

Record about 200 samples of each gesture, from different angles and distances, using the same hand you'll play with.

## Project files

```
common.py          shared helpers (hand detection, features)
hand_test.py       step 1
collect_data.py    step 2
train.py           step 3
play.py            step 4
data/              recorded samples (created by step 2)
models/            trained model (created by step 3)
```

## Limitations

- The model learns only my hand, so other hands or very different lighting may be misread until more samples are added.
- One hand and three gestures only.
- Test accuracy can look better than real play, because neighboring camera frames look alike. Judge it by playing.

## Privacy

Video is processed on your computer and is never saved or uploaded.
