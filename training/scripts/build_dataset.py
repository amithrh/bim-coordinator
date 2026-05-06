"""Pipeline that produces the full training/val/test dataset.

Outputs (in training/data/):
  raw/briefs.jsonl          - all generated briefs with target template ids
  raw/examples.jsonl        - full (prompt, response, metadata) examples
  processed/train.jsonl     - 80% in ChatML format, ready for mlx-lm
  processed/valid.jsonl     - 10% validation
  processed/test.jsonl      - 10% held-out test (with ground truth)

The ChatML format used:
  {"messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]}

This matches the format mlx-lm expects when --data points at a directory
containing train.jsonl/valid.jsonl/test.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from brief_generator import briefs_for_template
from candidate_selector import CandidatePool
from response_generator import build_training_example
from templates_index import load_all


SYSTEM_PROMPT = (
    "You are an expert architectural consultant for the BIM Coordinator. "
    "You help users find floor plan templates that match their lifestyle, "
    "budget, and location preferences from a curated library of 500 real-world "
    "templates spanning 74 countries. "
    "Always ground your recommendations in the specific metadata of the candidate "
    "templates provided. Pick the top 4 best fits, with the strongest match first, "
    "and explain each choice in 2-4 sentences using real architectural reasoning "
    "(layout, ceiling height, room flow, cultural conventions, persona fit). "
    "Be honest about trade-offs."
)


def build_dataset(
    n_per_template: int = 10,
    test_fraction: float = 0.10,
    val_fraction: float = 0.10,
    seed: int = 42,
    output_dir: str = "training/data",
) -> dict[str, Any]:
    """Build the full dataset.

    Returns metadata about the build (counts, splits, etc.).
    """
    rng = random.Random(seed)

    # Load templates
    templates = load_all()
    pool = CandidatePool(templates)
    print(f"Loaded {len(templates)} templates")

    # Generate briefs + examples for every template
    examples: list[dict[str, Any]] = []
    raw_briefs: list[dict[str, Any]] = []
    persona_counter: Counter[str] = Counter()
    style_counter: Counter[str] = Counter()

    for t in templates:
        briefs = briefs_for_template(t, rng, n=n_per_template)
        for b in briefs:
            cands = pool.select(t, rng=rng)
            ex = build_training_example(b, t, cands, rng)
            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": ex.user_prompt},
                    {"role": "assistant", "content": ex.assistant_response},
                ],
                "meta": {
                    "target_template_id": ex.target_template_id,
                    "candidate_ids": ex.candidate_ids,
                    "persona": ex.persona,
                    "brief_text": b.text,
                    "brief_dimensions": b.style_dimensions,
                },
            })
            raw_briefs.append({
                "brief": b.text,
                "target_template_id": b.target_template_id,
                "persona": b.persona,
                "dimensions": b.style_dimensions,
            })
            persona_counter[b.persona] += 1
            for d in b.style_dimensions:
                style_counter[d] += 1

    print(f"Generated {len(examples)} examples")

    # Shuffle and split (split is at the EXAMPLE level, but we ensure no template
    # leaks across train/test by stratifying — OR we accept some leakage because
    # each brief is unique).
    # We want the test set to contain templates the model COULD see during
    # training (just different briefs), so we'll split at example level.
    rng2 = random.Random(seed + 1)
    rng2.shuffle(examples)

    n_test = int(len(examples) * test_fraction)
    n_val = int(len(examples) * val_fraction)
    n_train = len(examples) - n_test - n_val

    train = examples[:n_train]
    val = examples[n_train:n_train + n_val]
    test = examples[n_train + n_val:]

    # Write files
    out = Path(output_dir)
    raw_dir = out / "raw"
    proc_dir = out / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(raw_dir / "briefs.jsonl", raw_briefs)
    write_jsonl(raw_dir / "examples.jsonl", examples)

    # Strip the meta from train/valid (mlx-lm wants only messages); keep meta in test
    def messages_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"messages": r["messages"]} for r in rows]

    write_jsonl(proc_dir / "train.jsonl", messages_only(train))
    write_jsonl(proc_dir / "valid.jsonl", messages_only(val))
    write_jsonl(proc_dir / "test.jsonl", test)  # keep meta for ground-truth eval

    # Sanity checks
    summary = {
        "total_examples": len(examples),
        "train": n_train,
        "valid": n_val,
        "test": n_test,
        "personas": dict(persona_counter.most_common()),
        "dimensions": dict(style_counter.most_common()),
        "n_templates": len(templates),
        "n_per_template": n_per_template,
    }
    with (out / "stats.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Build summary ===")
    print(f"  total examples:  {summary['total_examples']}")
    print(f"  train / val / test: {summary['train']} / {summary['valid']} / {summary['test']}")
    print(f"  personas (top 5): {dict(list(summary['personas'].items())[:5])}")
    print(f"  output: {out}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-template", type=int, default=10,
                        help="Briefs to generate per template (10 -> 5000 examples)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="training/data")
    args = parser.parse_args()
    build_dataset(
        n_per_template=args.n_per_template,
        seed=args.seed,
        output_dir=args.output_dir,
    )
