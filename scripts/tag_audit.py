#!/usr/bin/env python3
"""Audit all templates against the controlled tag vocabulary.

Reports:
- Templates missing facets that are required (region, country, size_band, typology, etc.)
- Tags used in templates that are NOT in the controlled vocabulary (typos / drift)
- Coverage stats per facet (how many templates per typology, era, etc.)
- Tag inventory for retrieval / mix-and-match planning

Usage:
    python scripts/tag_audit.py
    python scripts/tag_audit.py --strict   # exit 1 if any template has unknown tag
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOCAB_PATH = ROOT / "data" / "tag_vocabulary.json"
TEMPLATE_DIRS = [ROOT / "data" / "templates" / "europe", ROOT / "data" / "templates" / "india", ROOT / "data" / "templates" / "global"]


def load_vocab():
    return json.loads(VOCAB_PATH.read_text())


def collect_templates():
    out = []
    for d in TEMPLATE_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                t = json.loads(f.read_text())
                out.append((f, t))
            except Exception as e:
                print(f"ERROR reading {f.name}: {e}")
    return out


def audit(strict: bool = False) -> int:
    vocab = load_vocab()
    facets = vocab["facets"]
    templates = collect_templates()
    issues: list[str] = []
    facet_counts: dict[str, Counter] = defaultdict(Counter)
    untagged_per_facet: dict[str, list[str]] = defaultdict(list)
    typology_to_templates: dict[str, list[str]] = defaultdict(list)

    for path, t in templates:
        meta = t.get("metadata", {})
        tags = set(meta.get("tags", []))
        name = path.name

        # Required facets
        if "region" not in meta:
            issues.append(f"{name}: missing metadata.region")
        else:
            facet_counts["region"][meta["region"]] += 1
            if meta["region"] not in facets["region"]["values"]:
                issues.append(f"{name}: region '{meta['region']}' not in vocab")

        if "country" not in meta:
            issues.append(f"{name}: missing metadata.country")
        else:
            facet_counts["country"][meta["country"]] += 1

        # Size band — derive from size_label if not explicit
        size_label = meta.get("size_label", "")
        size_band = meta.get("size_band")
        if not size_band:
            mapping = facets["size_band"].get("mapping_de" if meta.get("region") == "europe" else "mapping_in", {})
            size_band = mapping.get(size_label.lower())
        if size_band:
            facet_counts["size_band"][size_band] += 1
        else:
            untagged_per_facet["size_band"].append(name)

        # Typology — pull from tags
        typology_tags_in_template = tags & set(facets["typology"]["values"])
        if typology_tags_in_template:
            for tg in typology_tags_in_template:
                facet_counts["typology"][tg] += 1
                typology_to_templates[tg].append(name)
        else:
            untagged_per_facet["typology"].append(name)

        # Era
        era_tags_in_template = tags & set(facets["era"]["values"])
        for tg in era_tags_in_template:
            facet_counts["era"][tg] += 1

        # Features
        feat_tags = tags & set(facets["features"]["values"])
        for tg in feat_tags:
            facet_counts["features"][tg] += 1

        # Suitable_for from explicit field
        for s in meta.get("suitable_for", []):
            facet_counts["suitable_for"][s] += 1

    # ---- Report ----
    print("=" * 60)
    print(f"Tag Audit Report  ({len(templates)} templates)")
    print("=" * 60)

    print("\n## Region coverage")
    for k, v in facet_counts["region"].most_common():
        print(f"  {k}: {v}")

    print("\n## Country coverage")
    for k, v in facet_counts["country"].most_common():
        print(f"  {k}: {v}")

    print("\n## Size band coverage")
    for k, v in facet_counts["size_band"].most_common():
        print(f"  {k}: {v}")

    print("\n## Typology coverage")
    for k, v in facet_counts["typology"].most_common():
        print(f"  {k}: {v}")

    print("\n## Era coverage")
    for k, v in facet_counts["era"].most_common():
        print(f"  {k}: {v}")

    print("\n## Top features")
    for k, v in facet_counts["features"].most_common(20):
        print(f"  {k}: {v}")

    if untagged_per_facet["typology"]:
        print(f"\n## Templates WITHOUT a typology tag ({len(untagged_per_facet['typology'])})")
        for n in untagged_per_facet["typology"][:30]:
            print(f"  {n}")
        if len(untagged_per_facet["typology"]) > 30:
            print(f"  ... and {len(untagged_per_facet['typology']) - 30} more")

    if issues:
        print(f"\n## Issues ({len(issues)})")
        for i in issues[:30]:
            print(f"  {i}")

    print("\n" + "=" * 60)
    print(f"Total: {len(templates)} | Issues: {len(issues)} | Untagged typology: {len(untagged_per_facet['typology'])}")

    if strict and (issues or untagged_per_facet["typology"]):
        return 1
    return 0


if __name__ == "__main__":
    strict = "--strict" in sys.argv
    sys.exit(audit(strict=strict))
