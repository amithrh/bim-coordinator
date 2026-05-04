.PHONY: all validate build svg fragments backend frontend clean

PY := .venv/bin/python
TEMPLATES := $(wildcard data/templates/*/*.json)
IFCS := $(patsubst data/templates/%/%.json,data/ifc_samples/%.ifc,$(TEMPLATES))

all: validate build svg fragments

validate:
	@for t in $(TEMPLATES); do $(PY) scripts/validate_template.py $$t || exit 1; done

build:
	$(PY) scripts/build_all.py

svg:
	$(PY) scripts/render_svg.py

fragments:
	$(PY) scripts/convert_to_fragments.py

backend:
	cd backend && ../.venv/bin/uvicorn main:app --reload --port 8000

frontend:
	cd frontend && npm run dev

clean:
	rm -rf data/ifc_samples/* data/svg_plans/* data/fragments/*
