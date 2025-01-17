.PHONY: lint docs

lint:
	uv run ruff format .
	uv run ruff check --fix .
	uv run mypy

docs:
	uv run --only-group docs sphinx-build -M html docs/ docs/_build/ -W
