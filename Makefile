.PHONY: clean clean-test clean-pyc clean-build docs help camera-test
.DEFAULT_GOAL := help

PYTHON_VERSION := 3.5

TENSORFLOW_VERSION := v2.0.0-beta0
DOCKER_TAG := 2.0.0b0-py3
TMP_DIR := .tmp
DIST_DIR := .dist
ENV_DIR := .env
MODEL := mobilenet_v2

REGION=us-central1
NOW=$(shell date +"%Y%m%d_%H%M%S")
GCS_BUCKET=gs://data-literate/public-datasets/flower-images
OUTPUT_DIR=${GCS_BUCKET}/flowers-${NOW}
SCALE_TIER=basic

WORKSPACE = $(shell echo $$PWD)

###
# Consumer targets
###

usb-accel-install:
	ansible-playbook playbooks/install-usb-accel.yml --extra-vars "@.env/example-vars.json" -i .env/example-inventory.ini

usb-accel-demo:
	echo "Running Edge TPU Demo from https://coral.withgoogle.com/docs/accelerator/get-started/#run-a-model-on-the-edge-tpu"
	ansible-playbook playbooks/demo-usb-accel.yml --extra-vars "@.env/example-vars.json" -i .env/example-inventory.ini
tflite:
	python -m "models.mobilenet_v2.py"

camera-test:
	python -m "detector.camera_test"

cp-tflite-lib:
	ansible-playbook playbooks/rpi-tensorflow-lite.yml --extra-vars "@.env/example-vars.json" -i .env/example-inventory.ini
	
tflite-lib:
	TENSORFLOW_VERSION=${TENSORFLOW_VERSION} ./tools/build-tflite-lib

rpi-install:
	ansible-playbook playbooks/bootstrap-rpi.yml --extra-vars "@.env/example-vars.json" -i .env/example-inventory.ini

###
# Maintainer targets
###

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint: ## check style with flake8
	flake8 rpi_vision tests

test: ## run tests quickly with the default Python
	py.test

test-all: ## run tests on every Python version with tox
	tox

coverage: ## check code coverage quickly with the default Python
	coverage run --source rpi_vision -m pytest
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/rpi_vision.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ rpi_vision
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs ## compile the docs watching for changes
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: dist ## package and upload a release
	twine upload dist/*

dist: clean ## builds source and wheel package
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean ## install the package to the active Python's site-packages
	python setup.py install
