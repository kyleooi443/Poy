.PHONY: setup data train test serve build fixtures format

PYTHON ?= python3.12
VENV := ml/.venv
PY := $(VENV)/bin/python
HAND_MODEL := web/public/models/hand_landmarker.task
HAND_MODEL_URL := https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
# make data ALLOW_CC_BY=1  -> also keep CC BY images (credited in ATTRIBUTION.md)
ACQUIRE_FLAGS := $(if $(ALLOW_CC_BY),--allow-cc-by,)

setup: $(HAND_MODEL)
	@if command -v uv >/dev/null; then uv venv -q --allow-existing -p 3.12 $(VENV) && uv pip install -q -p $(PY) -r ml/requirements.txt; \
	else $(PYTHON) -m venv $(VENV) && $(PY) -m pip install -q -r ml/requirements.txt; fi
	cd web && npm ci --no-audit --no-fund

$(HAND_MODEL):
	curl -fSL -o $@ $(HAND_MODEL_URL)

data:
	cd ml && ../$(PY) -m poy_ml.acquire $(ACQUIRE_FLAGS)
	cd ml && ../$(PY) -m poy_ml.landmarks
	cd ml && ../$(PY) -m poy_ml.attribution

train:
	cd ml && ../$(PY) -m poy_ml.train

fixtures:
	cd ml && ../$(PY) -m poy_ml.parity

test:
	cd ml && .venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/python -m pytest -q
	cd web && npm run lint && npm test

format:
	cd ml && .venv/bin/ruff format . && .venv/bin/ruff check --fix .
	cd web && npm run format

serve:
	cd web && npm run dev

build:
	cd web && npm run build
