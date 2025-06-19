# Makefile for easy development workflows.
# See development.md for docs.
# Note GitHub Actions call uv directly, not this Makefile.

.DEFAULT_GOAL := default

.PHONY: default install lint test test-full upgrade build clean agent-rules

default: agent-rules install lint test 

install:
	uv sync --all-extras --dev

lint:
	uv run python devtools/lint.py

test:
	uv run pytest

test-full:
	ENABLE_TESTS_ONLINE=1 ENABLE_TESTS_INTEGRATION=1 uv run pytest

run: install lint test
	uv run kash

upgrade:
	uv sync --upgrade --all-extras --dev

build:
	uv build

clean:
	-rm -rf dist/
	-rm -rf *.egg-info/
	-rm -rf .pytest_cache/
	-rm -rf .mypy_cache/
	-rm -rf .venv/
	-find . -type d -name "__pycache__" -exec rm -rf {} +

# Rebuild the docx template from source.
# Template source was created from Google docs then unzipped and formatted.
docx_template:
	cd ./template_src/docx_template && zip -r ../../src/kash/kits/docs/doc_formats/templates/docx_template.docx .
