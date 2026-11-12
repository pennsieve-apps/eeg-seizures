.PHONY: help test fixture run run-demo clean

SERVICE_NAME ?= eeg-seizures
PYTHON ?= python3
IMAGE ?= eeg-seizures:dev
DEMO ?= ../workflow-scalp-eeg/data/siena/BIDS_Siena

.DEFAULT: help

help:
	@echo "Make Help for $(SERVICE_NAME)"
	@echo ""
	@echo "make test     - pytest: units, and the synthetic fixture through the processor (no Docker)"
	@echo "make fixture  - write the synthetic fixture to data/fixture/ (eeg/ port + truth.json)"
	@echo "make run      - build the image and run it in Docker over data/fixture/, writing to data/output/"
	@echo "make run-demo - run the image over a BIDS-EEG dataset or eeg-prep output (DEMO=$(DEMO)) into data/output-demo/"
	@echo "make clean    - remove data/output*/ and data/fixture/"
	@echo ""
	@echo "PYTHON=$(PYTHON) (set PYTHON=.venv/bin/python after pip install -r requirements-dev.txt)"

test:
	$(PYTHON) -m pytest -q tests

fixture:
	$(PYTHON) -m tests.make_fixture data/fixture

run: clean fixture
	mkdir -p data/output
	docker compose -f docker-compose.yml down --remove-orphans
	docker compose -f docker-compose.yml build
	docker compose -f docker-compose.yml up --exit-code-from processor
	docker compose -f docker-compose.yml down --remove-orphans
	@echo "--- outputs:"
	@find data/output -type f | sort

run-demo:
	docker build --platform linux/amd64 -t $(IMAGE) .
	rm -rf data/output-demo && mkdir -p data/output-demo
	docker run --rm --platform linux/amd64 --cpus 1 --memory 2g --env-file dev.env \
		-v "$(abspath $(DEMO))":/data/input:ro -v "$(CURDIR)/data/output-demo":/data/output $(IMAGE)

clean:
	rm -rf data/output data/output-demo data/output-lambda data/fixture
