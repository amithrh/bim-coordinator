"""Knowledge-distillation data generator for architect→TowerSpec mapping.

Uses Codex CLI as the teacher to generate (architect_name, spec_json) pairs
that we'll later use to fine-tune Llama 3.2 3B. After fine-tuning, Llama
will know architects on its own — no Codex needed at runtime.

Output: training/data/architect_specs.jsonl
  Each line: {"architect": "Sou Fujimoto", "spec": {...}, "rationale": "..."}

Usage:
  python training/scripts/generate_architect_data.py
  python training/scripts/generate_architect_data.py --architects MAD MVRDV "Tadao Ando"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


# Curated list of architects spanning a range of styles, eras, regions.
# Aiming for ~100 names so the model sees enough examples per pattern.
ARCHITECTS = [
    # Modernist masters
    "Le Corbusier", "Mies van der Rohe", "Frank Lloyd Wright",
    "Walter Gropius", "Alvar Aalto", "Oscar Niemeyer", "Louis Kahn",
    # Late-modernist / postmodern
    "I.M. Pei", "Philip Johnson", "Kenzo Tange", "James Stirling",
    "Robert Venturi", "Aldo Rossi", "Charles Moore",
    # Deconstructivists / contemporary form-finders
    "Frank Gehry", "Zaha Hadid", "Daniel Libeskind", "Peter Eisenman",
    "Coop Himmelb(l)au", "Wolf D. Prix",
    # High-tech
    "Norman Foster", "Richard Rogers", "Renzo Piano",
    "Nicholas Grimshaw", "Santiago Calatrava",
    # Regionalist / minimalist
    "Tadao Ando", "Glenn Murcutt", "Peter Zumthor", "Alvaro Siza",
    "Eduardo Souto de Moura", "Alberto Campo Baeza",
    # Japanese masters / contemporaries
    "Kenzo Tange", "Kazuo Shinohara", "Toyo Ito", "SANAA",
    "Kazuyo Sejima", "Sou Fujimoto", "Kengo Kuma", "Shigeru Ban",
    "Junya Ishigami",
    # Programmatic / OMA-influenced
    "Rem Koolhaas", "Bjarke Ingels", "BIG", "MVRDV", "OMA",
    "MAD architects", "WORK Architecture",
    # Stars of the 2010s-20s
    "Jeanne Gang", "Studio Gang", "SHoP Architects",
    "Diller Scofidio + Renfro", "Snøhetta", "Herzog & de Meuron",
    "Jacques Herzog", "Pierre de Meuron",
    # Asian contemporary
    "Wang Shu", "Riken Yamamoto", "Toshiko Mori",
    "Liu Jiakun", "Zhang Ke",
    # Indian / South Asian
    "Charles Correa", "B.V. Doshi", "Raj Rewal", "Anupama Kundoo",
    "Bijoy Jain", "Studio Mumbai",
    # African / Latin American / Middle Eastern
    "David Adjaye", "Francis Kéré", "Lina Bo Bardi",
    "Lacaton & Vassal", "Solano Benitez",
    "Marwa Al-Sabouni", "Sahel Al Hiyari",
    # Australian / Pacific
    "Peter Stutchbury", "Sean Godsell", "Kerstin Thompson",
    # Younger / emerging
    "Junya Ishigami", "Smiljan Radić", "Sou Fujimoto",
    "Selgas Cano", "Anne Lacaton", "Kjetil Trædal Thorsen",
    # Big-name firms (often mentioned by firm name not principal)
    "Foster + Partners", "BIG-Bjarke Ingels Group",
    "Skidmore Owings & Merrill", "SOM", "KPF",
    "Pelli Clarke Pelli", "Adrian Smith + Gordon Gill",
    "Gensler", "Aedas", "RTKL",
    # Historical reference points
    "Antoni Gaudí", "Louis Sullivan", "Erich Mendelsohn",
    "Karl Friedrich Schinkel",
]


def generate_one(architect: str, model: str = "gpt-5.4-mini") -> dict | None:
    """Use Codex CLI to generate a TowerSpec for one architect.

    Returns the spec dict, or None if Codex failed.
    """
    sys.path.insert(0, str(REPO))
    from backend.app.codex_client import interpret_architect

    print(f"  → {architect}", flush=True)
    t0 = time.time()
    interp = interpret_architect(architect, model=model, timeout=120, use_cache=True)
    dt = time.time() - t0

    if interp.backend == "fallback" or not interp.spec:
        print(f"    ❌ {dt:.1f}s — {interp.error}", flush=True)
        return None
    print(f"    ✅ {dt:.1f}s ({interp.backend}) — {interp.rationale[:80]}", flush=True)
    return {
        "architect": architect,
        "spec": interp.spec,
        "rationale": interp.rationale,
        "backend": interp.backend,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architects", nargs="*", default=None,
                        help="Override default list (for testing/extending)")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--out", default=str(REPO / "training" / "data" / "architect_specs.jsonl"))
    args = parser.parse_args()

    architects = args.architects if args.architects else ARCHITECTS
    # Dedup while preserving order
    seen = set()
    unique = []
    for a in architects:
        if a not in seen:
            seen.add(a)
            unique.append(a)
    architects = unique

    print(f"Generating spec data for {len(architects)} architects via {args.model}...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Append mode so we can resume across multiple runs
    existing = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                    existing.add(d["architect"])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Found {len(existing)} existing entries; will skip them.")

    n_ok = 0
    n_skip = 0
    n_err = 0
    with out_path.open("a") as f:
        for architect in architects:
            if architect in existing:
                n_skip += 1
                continue
            entry = generate_one(architect, model=args.model)
            if entry is None:
                n_err += 1
                continue
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            n_ok += 1

    print()
    print(f"Done. New: {n_ok}  Skipped (already cached): {n_skip}  Errors: {n_err}")
    print(f"Output: {out_path}")
    print(f"Total entries in file: {len(existing) + n_ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
