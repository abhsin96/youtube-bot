PYTHON := venv/bin/python
RUFF   := venv/bin/ruff
BLACK  := venv/bin/black
PYTEST := venv/bin/pytest

.PHONY: lint fmt check test

lint:
	$(RUFF) check src tests

fmt:
	$(BLACK) src tests

check:
	$(RUFF) check src tests
	$(BLACK) --check src tests

test:
	$(PYTEST)
