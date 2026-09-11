PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
PY := $(BIN)/python
PIP := $(BIN)/pip
INSTALL_STAMP := $(VENV)/.installed

.DEFAULT_GOAL := help
.PHONY: help setup export-deps run test clean-venv

help:
	@echo "make setup       Create .venv and install GUI, analysis and report dependencies"
	@echo "make export-deps Install or update PPTX/PDF report dependencies"
	@echo "make run ARGS='--inspect /path/to/scan.hdr'"
	@echo "make test        Run the test suite using .venv"
	@echo "make clean-venv  Remove only this project's virtual environment"

$(INSTALL_STAMP): pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade setuptools wheel
	$(PIP) install --no-build-isolation -e '.[gui,analysis,export,dev]'
	touch $(INSTALL_STAMP)

setup: $(INSTALL_STAMP)

export-deps: $(INSTALL_STAMP)
	$(PIP) install --no-build-isolation -e '.[export]'

run: $(INSTALL_STAMP)
	$(PY) -m muaxis $(ARGS)

test: $(INSTALL_STAMP)
	$(PY) -m unittest discover -s tests -v

clean-venv:
	rm -rf $(VENV)
