PYTHON ?= python
OUTPUT ?= results/run

.PHONY: all test serve

all:
	$(PYTHON) -m reciprec.run --archive "$(ARCHIVE)" --output "$(OUTPUT)"

test:
	$(PYTHON) -m pytest

serve:
	docker compose up --build -d --wait
