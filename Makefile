.PHONY: venv install lint format test

venv:
	uv venv .venv --python 3.10

install:
	uv pip install -e ".[dev]"

lint:
	uv run --no-project ruff check lib/ tests/ nodes.py

format:
	uv run --no-project ruff format lib/ tests/ nodes.py

test:
	uv run --no-project pytest tests/ -v
