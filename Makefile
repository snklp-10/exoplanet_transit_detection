PY ?= python

.PHONY: help install check test lint format

help:
	@echo "make install   - editable install with dev extras"
	@echo "make check     - verify environment (add NETWORK=1 to test MAST/NASA reachability)"
	@echo "make test      - run unit tests"
	@echo "make lint      - ruff lint"
	@echo "make format    - ruff format + import sort"

install:
	$(PY) -m pip install -e ".[dev,boost,detrend,track]"

check:
	$(PY) scripts/check_env.py $(if $(NETWORK),--network,)

test:
	$(PY) -m pytest

lint:
	ruff check src tests scripts

format:
	ruff check --fix src tests scripts
	ruff format src tests scripts
