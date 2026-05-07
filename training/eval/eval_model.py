"""Evaluation harness for the BIM Coordinator SLM.

Runs a model over the held-out test set and scores it on:

  - top4_accuracy:  did the response include the ground-truth template id in its top-4 picks?
  - format_valid:   can we parse the response into ranked picks?
  - faithfulness:   do all referenced template ids exist in the candidate list?
  - mention_target: did the response even mention the target template id (top-K=10)?
  - latency_p50/95: wall-clock time per generation
  - response_len:   length sanity check

Supports three model backends:
  - mlx     (local MLX model dir or HF id)
  - openai  (OpenAI-compatible HTTP endpoint, e.g. Cloud Run, Ollama, vLLM)
  - claude  (Anthropic API direct - used for baseline comparison)

Output: prints scorecard, writes JSON results.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Patterns: structured `**N. Title** (\`id\`)`, backticked, or bare id.
TEMPLATE_ID_RE = re.compile(r"`((?:eu|gl|in)_[a-z0-9_]+)`")
NUMBERED_PICK_RE = re.compile(
    r"\*\*\s*(\d+)\.\s*[^*]+\*\*\s*\(`((?:eu|gl|in)_[a-z0-9_]+)`\)",
    re.MULTILINE,
)
# Bare template id (matches eu_de_xxx, gl_jp_xxx, in_2bhk_xxx with prefix discipline).
# Char-after-prefix can be letter OR digit (e.g. in_2bhk_trichy).
BARE_ID_RE = re.compile(r"\b((?:eu|gl|in)_[a-z0-9][a-z0-9_]+)\b")


def _dedup(ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tid in ids:
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


def parse_picks(response: str) -> list[str]:
    r"""Extract the model's top picks in rank order.

    Tries:
      1. Structured pattern ``**1. Title** (\`id\`)``
      2. Backticked id ``\`id\```
      3. Bare template id (eu_/gl_/in_ prefix)

    Dedupes while preserving order.
    """
    structured = NUMBERED_PICK_RE.findall(response)
    if structured:
        ranked = sorted(structured, key=lambda x: int(x[0]))
        return _dedup([tid for _, tid in ranked])

    ids = TEMPLATE_ID_RE.findall(response)
    if ids:
        return _dedup(ids)

    bare = BARE_ID_RE.findall(response)
    return _dedup(bare)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Per-example evaluation row."""

    target_template_id: str
    candidate_ids: list[str]
    persona: str
    response: str
    picks: list[str]
    latency_s: float
    error: str | None = None

    @property
    def top4_correct(self) -> bool:
        return self.target_template_id in self.picks[:4]

    @property
    def top1_correct(self) -> bool:
        return bool(self.picks) and self.picks[0] == self.target_template_id

    @property
    def mention_target(self) -> bool:
        return self.target_template_id in self.picks

    @property
    def format_valid(self) -> bool:
        return len(self.picks) >= 4

    @property
    def faithful(self) -> bool:
        """All referenced template ids must exist in the candidates."""
        cset = set(self.candidate_ids)
        return all(tid in cset for tid in self.picks)


def score_results(results: list[EvalResult]) -> dict[str, Any]:
    """Compute aggregate scorecard."""
    valid = [r for r in results if r.error is None]
    if not valid:
        return {"error": "no successful generations"}

    n = len(valid)
    metrics: dict[str, Any] = {
        "n_examples": n,
        "n_errors": len(results) - n,
        "top1_acc": sum(r.top1_correct for r in valid) / n,
        "top4_acc": sum(r.top4_correct for r in valid) / n,
        "mention_acc": sum(r.mention_target for r in valid) / n,
        "format_valid": sum(r.format_valid for r in valid) / n,
        "faithful": sum(r.faithful for r in valid) / n,
        "latency_p50_s": statistics.median(r.latency_s for r in valid),
        "latency_p95_s": _percentile([r.latency_s for r in valid], 95),
        "avg_response_chars": statistics.mean(len(r.response) for r in valid),
    }

    # Slice by persona too — interesting for diagnosing weakness
    by_persona: dict[str, list[EvalResult]] = defaultdict(list)
    for r in valid:
        by_persona[r.persona].append(r)
    metrics["per_persona_top4"] = {
        p: sum(x.top4_correct for x in items) / len(items)
        for p, items in by_persona.items()
    }
    return metrics


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    k = (len(sv) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sv) - 1)
    if f == c:
        return sv[int(k)]
    return sv[f] + (sv[c] - sv[f]) * (k - f)


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

GenerateFn = Callable[[list[dict[str, str]]], str]
"""A backend takes a chat-formatted message list and returns the assistant text."""


def make_mlx_backend(
    model_path: str,
    max_tokens: int = 768,
    adapter_path: str | None = None,
) -> GenerateFn:
    """Load an MLX model and return a generate function.

    If adapter_path is provided, the LoRA adapter is loaded on top of the base.
    """
    from mlx_lm import generate, load

    if adapter_path:
        model, tokenizer = load(model_path, adapter_path=adapter_path)
    else:
        model, tokenizer = load(model_path)

    def fn(messages: list[dict[str, str]]) -> str:
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        return out

    return fn


def make_openai_backend(
    base_url: str,
    model: str,
    api_key: str = "anything",
    max_tokens: int = 768,
) -> GenerateFn:
    """Hit an OpenAI-compatible endpoint (Ollama, vLLM, Cloud Run)."""
    import urllib.request
    import urllib.error

    def fn(messages: list[dict[str, str]]) -> str:
        body = json.dumps({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            base_url.rstrip("/") + "/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    return fn


def make_claude_backend(
    model: str = "claude-3-5-haiku-latest",
    max_tokens: int = 768,
) -> GenerateFn:
    """Use Anthropic API for baseline comparison."""
    try:
        import anthropic
    except ImportError:
        raise SystemExit("Run: pip install anthropic")

    client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY

    def fn(messages: list[dict[str, str]]) -> str:
        # Anthropic API: separate system + user/assistant
        system = ""
        chat = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                chat.append({"role": m["role"], "content": m["content"]})
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=chat,
        )
        return resp.content[0].text

    return fn


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_eval(
    backend: GenerateFn,
    test_path: str,
    limit: int | None = None,
    out_path: str | None = None,
) -> dict[str, Any]:
    """Run the backend over the test set and score."""
    rows = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]

    print(f"Evaluating {len(rows)} examples...")
    results: list[EvalResult] = []
    for i, row in enumerate(rows, 1):
        meta = row["meta"]
        # When evaluating, we strip the assistant message — the model has to produce it
        eval_messages = [m for m in row["messages"] if m["role"] != "assistant"]

        t0 = time.time()
        try:
            response = backend(eval_messages)
            err = None
        except Exception as e:
            response = ""
            err = repr(e)
        dt = time.time() - t0

        picks = parse_picks(response) if response else []
        result = EvalResult(
            target_template_id=meta["target_template_id"],
            candidate_ids=meta["candidate_ids"],
            persona=meta["persona"],
            response=response,
            picks=picks,
            latency_s=dt,
            error=err,
        )
        results.append(result)

        # Live progress
        if i % 10 == 0 or i == len(rows):
            recent = results[-min(20, len(results)) :]
            recent_acc = sum(r.top4_correct for r in recent if r.error is None) / max(
                1, sum(1 for r in recent if r.error is None)
            )
            print(
                f"  [{i:4d}/{len(rows)}] "
                f"recent_top4_acc={recent_acc:.2%} avg_lat={dt:.2f}s",
                file=sys.stderr,
            )

    scorecard = score_results(results)

    # Print summary
    print("\n" + "=" * 60)
    print("SCORECARD")
    print("=" * 60)
    for k, v in scorecard.items():
        if isinstance(v, dict):
            print(f"\n  {k}:")
            for sk, sv in sorted(v.items(), key=lambda x: -x[1] if isinstance(x[1], float) else 0):
                if isinstance(sv, float):
                    print(f"    {sk:30s} {sv:.2%}")
                else:
                    print(f"    {sk:30s} {sv}")
        elif isinstance(v, float):
            if "acc" in k or "valid" in k or "faithful" in k or "mention" in k:
                print(f"  {k:25s} {v:.2%}")
            else:
                print(f"  {k:25s} {v:.3f}")
        else:
            print(f"  {k:25s} {v}")

    if out_path:
        out = {
            "scorecard": scorecard,
            "samples": [
                {
                    "target": r.target_template_id,
                    "persona": r.persona,
                    "picks": r.picks,
                    "top4_correct": r.top4_correct,
                    "faithful": r.faithful,
                    "latency_s": round(r.latency_s, 3),
                    "response_excerpt": r.response[:300],
                    "error": r.error,
                }
                for r in results
            ],
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {out_path}")

    return scorecard


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["mlx", "openai", "claude"], required=True)
    parser.add_argument("--model", required=True, help="MLX model path/id or model name")
    parser.add_argument("--adapter-path", help="LoRA adapter path (mlx backend only)")
    parser.add_argument("--base-url", help="For openai backend")
    parser.add_argument("--api-key", default="anything", help="For openai backend")
    parser.add_argument("--test", default="training/data/processed/test.jsonl")
    parser.add_argument("--limit", type=int, help="Eval only first N examples")
    parser.add_argument("--out", help="Write per-example results to JSON")
    parser.add_argument("--max-tokens", type=int, default=768)
    args = parser.parse_args()

    if args.backend == "mlx":
        backend = make_mlx_backend(
            args.model,
            max_tokens=args.max_tokens,
            adapter_path=args.adapter_path,
        )
    elif args.backend == "openai":
        if not args.base_url:
            sys.exit("--base-url required for openai backend")
        backend = make_openai_backend(args.base_url, args.model, args.api_key, max_tokens=args.max_tokens)
    elif args.backend == "claude":
        backend = make_claude_backend(args.model, max_tokens=args.max_tokens)

    run_eval(backend, args.test, limit=args.limit, out_path=args.out)
