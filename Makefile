.PHONY: install test lint example plan registry

install:
	python -m pip install -e .[dev]

test:
	pytest -q

lint:
	ruff check src tests scripts

example:
	python -m tsyparty example --out outputs/sample

plan:
	python -m tsyparty show-plan

registry:
	python -m tsyparty registry --public-only
