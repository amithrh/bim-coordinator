"""Convert (architect, spec) JSONL into ChatML training examples for mlx-lm.

Input:  training/data/architect_specs.jsonl
Output: training/data/processed/train_architect.jsonl
        training/data/processed/valid_architect.jsonl
        training/data/processed/test_architect.jsonl

Each example follows the same ChatML format the existing fine-tune pipeline
uses, with system + user + assistant roles. The model learns:
  Brief mentioning architect X → JSON spec for X.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


SYSTEM_PROMPT = (
    "You are an architectural BIM assistant. Given a design brief that "
    "names a famous architect, output a JSON object describing the geometric "
    "signature of a residential tower inspired by their work. Output only "
    "valid JSON — no prose, no markdown fences."
)


# Brief templates so the model sees varied phrasings per architect
BRIEF_TEMPLATES = [
    "Design me a {n}-story residential tower inspired by {architect}.",
    "I want a {n}-story tower in the style of {architect}.",
    "Generate a {n}-story residential tower inspired by {architect}'s work.",
    "Output the geometric signature for a tower inspired by {architect}.",
    "{n}-story residential tower, {architect} style.",
    "Tower in {city}, {architect}-inspired, {n} floors.",
    "Apartment building inspired by {architect} — {n} levels.",
]

CITIES = [
    "Dubai", "London", "New York", "Tokyo", "Singapore", "Mumbai",
    "Berlin", "Paris", "Sydney", "Shanghai", "Toronto", "Stockholm",
]


def make_examples_for_architect(architect: str, spec: dict, rationale: str,
                                 n_briefs: int = 4, rng: random.Random | None = None) -> list[dict]:
    """Create N training examples for one architect with varied briefs."""
    rng = rng or random.Random()
    out = []
    for _ in range(n_briefs):
        template = rng.choice(BRIEF_TEMPLATES)
        n_floors = rng.choice([12, 15, 18, 20, 25, 30])
        city = rng.choice(CITIES)
        brief = template.format(architect=architect, n=n_floors, city=city)
        # Assistant response = JSON spec + the rationale
        assistant_payload = {**spec, "rationale": rationale}
        out.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": brief},
                {"role": "assistant",
                 "content": json.dumps(assistant_payload, ensure_ascii=False)},
            ],
            "meta": {"architect": architect, "n_floors": n_floors,
                      "city": city},
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="training/data/architect_specs.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        default="training/data/processed",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-briefs-per-architect", type=int, default=4)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--valid-fraction", type=float, default=0.10)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent.parent
    in_path = repo / args.input
    if not in_path.exists():
        print(f"ERROR: input file does not exist: {in_path}", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    architects: list[tuple[str, dict, str]] = []
    with in_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
                architects.append((d["architect"], d["spec"], d.get("rationale", "")))
            except (json.JSONDecodeError, KeyError):
                continue

    if not architects:
        print(f"ERROR: no entries in {in_path}", file=sys.stderr)
        return 2

    print(f"Loaded {len(architects)} architects from {in_path}")

    # Architect-level stratified split (same architect's briefs all in same split)
    rng.shuffle(architects)
    n_test = max(1, int(len(architects) * args.test_fraction))
    n_valid = max(1, int(len(architects) * args.valid_fraction))
    test_set = architects[:n_test]
    valid_set = architects[n_test:n_test + n_valid]
    train_set = architects[n_test + n_valid:]

    print(f"Splits: train={len(train_set)}, valid={len(valid_set)}, test={len(test_set)} "
          f"(at architect level)")

    # Generate ChatML examples
    train_examples = []
    valid_examples = []
    test_examples = []
    for arch, spec, rationale in train_set:
        train_examples.extend(make_examples_for_architect(
            arch, spec, rationale, args.n_briefs_per_architect, rng))
    for arch, spec, rationale in valid_set:
        valid_examples.extend(make_examples_for_architect(
            arch, spec, rationale, max(1, args.n_briefs_per_architect // 2), rng))
    for arch, spec, rationale in test_set:
        test_examples.extend(make_examples_for_architect(
            arch, spec, rationale, max(1, args.n_briefs_per_architect // 2), rng))

    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(rows: list[dict], name: str, strip_meta: bool = True) -> None:
        p = out_dir / name
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                if strip_meta:
                    f.write(json.dumps({"messages": r["messages"]}, ensure_ascii=False) + "\n")
                else:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {p.name}: {len(rows)} examples")

    write(train_examples, "train_architect.jsonl")
    write(valid_examples, "valid_architect.jsonl")
    write(test_examples, "test_architect.jsonl", strip_meta=False)

    print()
    print(f"Total: train {len(train_examples)} + valid {len(valid_examples)} + test {len(test_examples)} "
          f"= {len(train_examples) + len(valid_examples) + len(test_examples)} examples")
    print(f"Held-out test architects: {[a for a, _, _ in test_set]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
