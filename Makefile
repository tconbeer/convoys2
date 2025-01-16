.PHONY: lint

lint:
	ruff format .
	ruff check --fix .
	mypy