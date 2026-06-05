.PHONY: test lint typecheck clean install dev

install:
	pip install -e "."

dev:
	pip install -e ".[dev]"

test:
	pytest -v --cov=src/engram --cov-report=term

lint:
	ruff check src/

typecheck:
	mypy src/engram/

format:
	ruff format src/

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

all: lint typecheck test
