# BIM Coordinator — Phase 1 (Walking Skeleton)

Voice-driven AI architect for the Nemetschek ecosystem. Phase 1 is a static demo: text prompt → 4 floor plan cards → detail view (2D + 3D) → adjustment sliders.

See `Day-Zero-Build-Plan-v3.2.docx` for the full spec.

## Quick start

```bash
# 1. Activate venv (already created)
source .venv/bin/activate

# 2. Build all templates → IFC + SVG + .frag
make all

# 3. Backend
make backend     # http://localhost:8000

# 4. Frontend (separate terminal)
make frontend    # http://localhost:3000
```

## Repo layout

See §4 of the build plan.
