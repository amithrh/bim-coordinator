"""Pick K candidate templates for a given target.

The candidate set is what the model sees in its prompt. For training:
  - 1 target (the ground-truth right answer)
  - 9 distractors (mix of difficulty levels)

Mix design (out of 9 distractors):
  - 2 'near' (same country and size_band, different city/style)
  - 2 'sibling_size' (same country, adjacent size_band)
  - 2 'far_country' (same size_band, different country/region)
  - 2 'persona_overlap' (shares a persona but different size or country)
  - 1 'random' (very different - acts as easy negative)

We shuffle the final list before returning so the target is not at a fixed position.
"""

from __future__ import annotations

import random
from collections import defaultdict

from templates_index import Template


_SIZE_ORDER = ["studio", "1bed", "2bed", "3bed", "4bed", "4bed+", "5plus_bed"]


def _adjacent_sizes(size_band: str) -> list[str]:
    """Return the size bands one step up and one step down."""
    if size_band not in _SIZE_ORDER:
        return []
    idx = _SIZE_ORDER.index(size_band)
    out = []
    if idx > 0:
        out.append(_SIZE_ORDER[idx - 1])
    if idx < len(_SIZE_ORDER) - 1:
        out.append(_SIZE_ORDER[idx + 1])
    return out


class CandidatePool:
    """Indexed view of the template library used to draw distractors quickly."""

    def __init__(self, templates: list[Template]) -> None:
        self.templates = templates
        self.by_id = {t.id: t for t in templates}

        self.by_country: dict[str, list[Template]] = defaultdict(list)
        self.by_size_band: dict[str, list[Template]] = defaultdict(list)
        self.by_country_size: dict[tuple[str, str], list[Template]] = defaultdict(list)
        self.by_persona: dict[str, list[Template]] = defaultdict(list)

        for t in templates:
            self.by_country[t.country].append(t)
            self.by_size_band[t.size_band].append(t)
            self.by_country_size[(t.country, t.size_band)].append(t)
            for p in t.suitable_for:
                self.by_persona[p].append(t)

    # ------------------------------------------------------------------ pickers
    def _pick_excluding(
        self,
        pool: list[Template],
        exclude: set[str],
        n: int,
        rng: random.Random,
    ) -> list[Template]:
        candidates = [t for t in pool if t.id not in exclude]
        if not candidates:
            return []
        if len(candidates) <= n:
            return list(candidates)
        return rng.sample(candidates, n)

    def near(
        self, target: Template, n: int, exclude: set[str], rng: random.Random
    ) -> list[Template]:
        """Same country + same size_band, different city/style."""
        pool = self.by_country_size.get((target.country, target.size_band), [])
        return self._pick_excluding(pool, exclude | {target.id}, n, rng)

    def sibling_size(
        self, target: Template, n: int, exclude: set[str], rng: random.Random
    ) -> list[Template]:
        """Same country, adjacent size_band."""
        siblings = []
        for sb in _adjacent_sizes(target.size_band):
            siblings.extend(self.by_country_size.get((target.country, sb), []))
        return self._pick_excluding(siblings, exclude | {target.id}, n, rng)

    def far_country(
        self, target: Template, n: int, exclude: set[str], rng: random.Random
    ) -> list[Template]:
        """Same size_band, different country (and ideally different region)."""
        pool = [
            t for t in self.by_size_band.get(target.size_band, [])
            if t.country != target.country
        ]
        return self._pick_excluding(pool, exclude | {target.id}, n, rng)

    def persona_overlap(
        self, target: Template, n: int, exclude: set[str], rng: random.Random
    ) -> list[Template]:
        """Shares a persona, but different size_band or country."""
        if not target.suitable_for:
            return []
        pool = []
        for p in target.suitable_for:
            for t in self.by_persona.get(p, []):
                if t.country != target.country or t.size_band != target.size_band:
                    pool.append(t)
        # de-dup
        seen = set()
        unique = []
        for t in pool:
            if t.id not in seen:
                seen.add(t.id)
                unique.append(t)
        return self._pick_excluding(unique, exclude | {target.id}, n, rng)

    def random_far(
        self, target: Template, n: int, exclude: set[str], rng: random.Random
    ) -> list[Template]:
        """Very different template, used as easy negative."""
        pool = [
            t for t in self.templates
            if t.country != target.country and t.size_band != target.size_band
        ]
        return self._pick_excluding(pool, exclude | {target.id}, n, rng)

    # ----------------------------------------------------------- main entrypoint
    def select(
        self,
        target: Template,
        n_distractors: int = 9,
        rng: random.Random | None = None,
    ) -> list[Template]:
        """Return [target + 9 distractors] shuffled."""
        rng = rng or random.Random()
        chosen: list[Template] = [target]
        seen: set[str] = {target.id}

        # Quotas (try to fill in order; if a bucket is empty, slack rolls forward)
        plan = [
            ("near", 2),
            ("sibling_size", 2),
            ("far_country", 2),
            ("persona_overlap", 2),
            ("random_far", 1),
        ]

        for bucket_name, quota in plan:
            picker = getattr(self, bucket_name)
            picks = picker(target, quota, seen, rng)
            for p in picks:
                if p.id not in seen:
                    chosen.append(p)
                    seen.add(p.id)

        # Top up if we fell short (e.g. tiny country with no near matches)
        while len(chosen) < n_distractors + 1:
            extra = self.random_far(target, n_distractors + 1 - len(chosen), seen, rng)
            if not extra:
                # last resort - any other template
                remaining = [t for t in self.templates if t.id not in seen]
                if not remaining:
                    break
                extra = rng.sample(remaining, min(n_distractors + 1 - len(chosen), len(remaining)))
            for p in extra:
                if p.id not in seen and len(chosen) < n_distractors + 1:
                    chosen.append(p)
                    seen.add(p.id)

        rng.shuffle(chosen)
        return chosen[: n_distractors + 1]


if __name__ == "__main__":
    from templates_index import load_all

    templates = load_all()
    pool = CandidatePool(templates)
    rng = random.Random(7)
    samples = rng.sample(templates, 3)

    for t in samples:
        print(f"\n{'='*70}\nTARGET: {t.id} ({t.short_locale}, {t.size_band})")
        cands = pool.select(t, rng=rng)
        for i, c in enumerate(cands, 1):
            marker = " <-- TARGET" if c.id == t.id else ""
            print(f"  [{i}] {c.id} ({c.short_locale}, {c.size_band}, {c.style[:40]}){marker}")
