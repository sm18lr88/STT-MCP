.PHONY: test-fast test lint coverage build pmat quality hooks

test-fast:
	uv run pytest -q --ignore tests/test_mcp_acceptance.py --ignore tests/test_media_integration.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run basedpyright
	uv run ty check src tests
	uv run mypy

coverage:
	uv run coverage run --source=stt_mcp -m pytest
	uv run coverage report --show-missing --fail-under=80

build:
	uv build

pmat:
	pmat analyze complexity --path src --fail-on-violation
	pmat analyze tdg
	pmat analyze dead-code --path src --exclude "tests/**" --max-depth 4
	pmat repo-score

quality: lint test coverage build pmat

hooks:
	uv run pre-commit install
