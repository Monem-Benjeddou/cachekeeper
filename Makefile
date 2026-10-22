VENV ?= .venv
PY := $(VENV)/bin/python
CORE := '*/cachekeeper/core/*'

.PHONY: setup check perf live

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install -e ".[dev]"
	$(VENV)/bin/pre-commit install

check:
	$(PY) -m ruff format --check .
	$(PY) -m ruff check .
	$(PY) -m mypy
	$(PY) -m coverage run -m pytest -q
	$(PY) -m coverage report --fail-under=85
	$(PY) -m coverage report --include=$(CORE) --fail-under=95

perf:
	$(PY) -m pytest -q -m perf

live:
	CACHEKEEPER_LIVE=1 $(PY) -m pytest -q -m live tests/live
