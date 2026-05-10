.PHONY: install test lint check api web gate-example inspect-v5

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

lint:
	python -m ruff check .

check: lint test

api:
	python -m uvicorn quant_lab.api.main:app --host 127.0.0.1 --port 8027

web:
	qlab-web --host 127.0.0.1 --port 8501 --lake-root /var/lib/quant-lab/lake

gate-example:
	qlab gate-example

inspect-v5:
	qlab inspect-v5 tests/fixtures/v5_reports
