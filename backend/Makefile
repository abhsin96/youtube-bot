PYTHON := venv/bin/python
RUFF   := venv/bin/ruff
BLACK  := venv/bin/black
PYTEST := venv/bin/pytest

.PHONY: lint fmt check test run run\:prod setup

lint:
	$(RUFF) check src tests

fmt:
	$(BLACK) src tests

check:
	$(RUFF) check src tests
	$(BLACK) --check src tests

test:
	$(PYTEST)

run:
	$(PYTHON) -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload

run\:prod:
	$(PYTHON) -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 2

setup:
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt
