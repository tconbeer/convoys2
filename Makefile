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
	uv run --group docs --all-packages sphinx-build -M html docs/ docs/_build/ -W
