.PHONY: install test lint example plan registry

install:
	python -m pip install -e .[dev]

test:
	pytest -q

lint:
	ruff check src tests scripts

example:
	python -m tsyparty.cli example --out outputs/sample

plan:
	python -m tsyparty.cli show-plan

registry:
	python -m tsyparty.cli registry --public-only
