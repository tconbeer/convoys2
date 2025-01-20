.PHONY: lint docs check

check:
	uv run ruff format .
	uv run ruff check --fix .
	uv run mypy
	uv run pytest

lint:
	uv run ruff format .
	uv run ruff check --fix .
	uv run mypy

docs:
	uv run --only-group docs sphinx-build -M html docs/ docs/_build/ -W
