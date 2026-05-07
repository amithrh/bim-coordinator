"""Evaluate the architect-adapter Llama on held-out architects.

For each held-out architect:
  - Ask the model for a TowerSpec (using the same prompt used in training)
  - Parse the JSON
  - Score:
      * format_valid: did it parse as JSON?
      * schema_valid: does it have all required keys with valid types?
      * value_in_bounds: are values within sane ranges?
      * spec_matches_teacher: does it match what Codex produced for this architect?
        (we have ground truth in test_architect.jsonl)
      * builds_validated_tower: can we build a 35/35 PASS IFC from this spec?

Usage:
  python training/scripts/eval_architect_adapter.py --adapter <path>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def parse_json_obj(text: str) -> dict | None:
    """Extract the first JSON object from text. Handles markdown fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
    return None


REQUIRED_KEYS = {
    "setback_pattern", "n_setbacks", "setback_amount_m",
    "n_amenity_floors", "units_per_typical_floor",
    "typical_unit_area_sqm", "floor_height_mm", "footprint_aspect",
}
ALLOWED_PATTERNS = {"none", "stepped", "pyramid", "inverse_taper", "mid_setback"}


def score_spec(spec: dict | None, ground_truth: dict | None) -> dict:
    """Score a single model output against expected ranges + ground truth."""
    out = {
        "format_valid": False,
        "schema_valid": False,
        "value_in_bounds": False,
        "matches_teacher_pattern": False,
        "matches_teacher_n_amenity": False,
    }
    if spec is None or not isinstance(spec, dict):
        return out
    out["format_valid"] = True
    if not REQUIRED_KEYS.issubset(spec.keys()):
        return out
    out["schema_valid"] = True

    try:
        ok = True
        if spec["setback_pattern"] not in ALLOWED_PATTERNS: ok = False
        if not (0 <= int(spec["n_setbacks"]) <= 12): ok = False
        if not (0.0 <= float(spec["setback_amount_m"]) <= 5.0): ok = False
        if not (1 <= int(spec["n_amenity_floors"]) <= 3): ok = False
        if not (2 <= int(spec["units_per_typical_floor"]) <= 8): ok = False
        if not (40 <= float(spec["typical_unit_area_sqm"]) <= 200): ok = False
        if not (2700 <= int(spec["floor_height_mm"]) <= 4000): ok = False
        if not (0.8 <= float(spec["footprint_aspect"]) <= 2.5): ok = False
        out["value_in_bounds"] = ok
    except (TypeError, ValueError, KeyError):
        return out

    if ground_truth:
        out["matches_teacher_pattern"] = (
            spec.get("setback_pattern") == ground_truth.get("setback_pattern")
        )
        out["matches_teacher_n_amenity"] = (
            spec.get("n_amenity_floors") == ground_truth.get("n_amenity_floors")
        )

    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter dir")
    parser.add_argument("--base-model", default="mlx-community/Llama-3.2-3B-Instruct-4bit")
    parser.add_argument("--test-file", default="training/data_architect/processed/test.jsonl")
    parser.add_argument("--build-ifc", action="store_true",
                        help="Also build + verify IFC for each generated spec")
    args = parser.parse_args()

    print(f"Loading model + adapter: {args.adapter}")
    from mlx_lm import load, generate

    model, tokenizer = load(args.base_model, adapter_path=args.adapter)
    print("Loaded.\n")

    # Load test set
    test_path = REPO / args.test_file
    examples = []
    with test_path.open() as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Test set: {len(examples)} examples ({len(set(e['meta']['architect'] for e in examples))} architects)")
    print()

    results: list[dict] = []
    n_format = n_schema = n_bounds = n_pattern_match = n_amenity_match = 0
    n_builds = 0

    for i, ex in enumerate(examples, 1):
        msgs = [m for m in ex["messages"] if m["role"] != "assistant"]
        prompt = tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False
        )
        # Ground truth JSON in the test example
        gt_str = ex["messages"][-1]["content"]
        gt = parse_json_obj(gt_str)

        t0 = time.time()
        out = generate(model, tokenizer, prompt=prompt, max_tokens=400, verbose=False)
        dt = time.time() - t0

        spec = parse_json_obj(out)
        s = score_spec(spec, gt)

        if s["format_valid"]: n_format += 1
        if s["schema_valid"]: n_schema += 1
        if s["value_in_bounds"]: n_bounds += 1
        if s["matches_teacher_pattern"]: n_pattern_match += 1
        if s["matches_teacher_n_amenity"]: n_amenity_match += 1

        arch = ex["meta"]["architect"]
        flag = "✅" if s["value_in_bounds"] else "🟡" if s["schema_valid"] else "❌"
        marker = ""
        if s["matches_teacher_pattern"]:
            marker += "🎯"
        if args.build_ifc and s["value_in_bounds"]:
            try:
                # Quick IFC build check
                import subprocess
                from backend.app.tower_generator import TowerSpec, ArchitectProfile, generate_tower
                from scripts.validate_template import validate_dict
                prof = ArchitectProfile(
                    setback_pattern=spec["setback_pattern"],
                    setback_n=spec["n_setbacks"],
                    setback_amount_m=spec["setback_amount_m"],
                    n_amenity_floors=spec["n_amenity_floors"],
                    sky_lobby_relative=spec.get("sky_lobby_relative", 0.5),
                    floor_height_mm=spec["floor_height_mm"],
                    typical_unit_area_sqm=spec["typical_unit_area_sqm"],
                    units_per_typical_floor=spec["units_per_typical_floor"],
                    footprint_aspect=spec["footprint_aspect"],
                )
                ts = TowerSpec(
                    n_floors=20, country="Test", city="Test",
                    floor_height_mm=prof.floor_height_mm,
                    typical_unit_area_sqm=prof.typical_unit_area_sqm,
                    units_per_typical_floor=prof.units_per_typical_floor,
                    setback_top_n=prof.setback_n if prof.setback_pattern in ("stepped", "pyramid") else 0,
                    setback_amount_m=prof.setback_amount_m if prof.setback_pattern in ("stepped", "pyramid") else 0,
                    profile=prof,
                )
                template = generate_tower(ts)
                if not validate_dict(template):
                    n_builds += 1
                    marker += "🏗"
            except Exception:
                pass

        print(f"  [{i:2d}/{len(examples)}] {flag}{marker:<3s} {dt:.1f}s {arch:25s} "
              f"pattern={spec.get('setback_pattern','?'):14s} "
              f"(gt={gt.get('setback_pattern','?') if gt else '?'})")

    n = len(examples)
    print()
    print("=" * 60)
    print(f"  format_valid:           {n_format}/{n}  ({100*n_format/n:.0f}%)")
    print(f"  schema_valid:           {n_schema}/{n}  ({100*n_schema/n:.0f}%)")
    print(f"  value_in_bounds:        {n_bounds}/{n}  ({100*n_bounds/n:.0f}%)")
    print(f"  matches teacher pattern:{n_pattern_match}/{n}  ({100*n_pattern_match/n:.0f}%)")
    print(f"  matches teacher amenity:{n_amenity_match}/{n}  ({100*n_amenity_match/n:.0f}%)")
    if args.build_ifc:
        print(f"  builds validated IFC:   {n_builds}/{n}  ({100*n_builds/n:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
