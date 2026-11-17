.PHONY: help setup setup-minimal setup-gpu prepare-monoted notebooks test lint

help:
	@printf '%s\n' \
		' setup-minimal   Install dependencies for the MonoTED preparation script' \
		' setup           Install CPU, notebook, research, and development dependencies' \
		' setup-gpu       Install the CUDA 12.1 PyTorch variant' \
		' prepare-monoted Run the MonoTED preparation command (pass ARGS="...")' \
		' notebooks       Start JupyterLab with research dependencies' \
		' test            Run the project test suite' \
		' lint            Run Ruff checks'

setup-minimal:
	uv sync --locked --no-dev

setup:
	uv sync --locked --extra cpu --extra notebooks --extra research --group dev

setup-gpu:
	uv sync --locked --extra gpu --extra notebooks --extra research --group dev

prepare-monoted:
	uv run --locked --no-dev prepare-monoted $(ARGS)

notebooks:
	uv run --locked --no-dev --extra cpu --extra notebooks --extra research jupyter lab --ServerApp.root_dir=notebooks

test:
	uv run --locked --extra cpu --extra notebooks --extra research --group dev pytest -q

lint:
	uv run --locked --extra cpu --extra notebooks --extra research --group dev ruff check scripts/prepare_monoted.py src tests
