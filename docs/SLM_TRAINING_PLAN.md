# BIM Coordinator SLM Training Plan

**Goal:** Fine-tune a small language model on the 500-template floor plan library to power architectural brief consultation. Demo-ready by Tuesday.

**Strategy principle:** Quality > speed. Safe choices throughout. Hedge with multiple model sizes. Rigorous evaluation at every gate.

---

## 1. Architecture

Two-stage retrieval + reasoning pipeline:

```
User brief
    │
    ▼
Stage 1: Sentence-Transformer (MiniLM-L6-v2)
  - Embed brief, cosine search over 500 templates
  - Returns top-10 candidates (~10ms, runs on CPU)
    │
    ▼
Stage 2: Fine-tuned Gemma 4 (E4B or E2B, decided Day 2)
  - Reads brief + top-10 candidates
  - Selects best 4 with architectural reasoning
  - Hosted on GCP Cloud Run + L4 GPU
    │
    ▼
Frontend (Windows laptop demo)
  - 4 floor plan cards
  - AI explanation streaming inline
  - Click card -> 2D + 3D detail view
```

**Why two stages:** Retrieval is fast and the LLM is bad at it. Reasoning is what the LLM is for. Splitting them gives <3s total response and keeps each component good at its job.

---

## 2. Locked decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model family | Gemma 4 (Apache 2.0) | Latest, fully open, Google-supported on Cloud Run |
| Model variants | **E2B AND E4B** (hedge) | Train both, pick winner on Day 2 evaluation |
| Training method | LoRA, rank 32, alpha 64 | Fits on M4 Max 64GB, fast, mergeable |
| Training framework | mlx-lm (Apple Silicon native) | 3-4x faster than PyTorch MPS on M4 |
| Quantization | Q4_K_M GGUF for inference | 4x size reduction, <3% quality loss |
| Vocab pruning | None (safe choice) | Avoid risk of breaking multilingual tokens |
| Hosting | GCP Cloud Run + L4 GPU + Ollama | Google's official Gemma 4 path, scale-to-zero |
| Hosting backup | HuggingFace Inference Endpoints | One-click fallback if Cloud Run fails |
| Local fallback | Ollama on Mac Studio | Ultimate emergency fallback for demo |
| Monitoring | Weights & Biases | Industry standard, free for individual |
| Eval framework | 6-metric scorecard, 3-way A/B | See section 5 |

---

## 3. Day-by-day plan

### Day 1 (Today, Thursday) - Foundation + Training Data

**Morning (3h)**
- [ ] Install MLX, mlx-lm, mlx-examples
- [ ] Verify Gemma 4 E2B and E4B downloadable
- [ ] Sanity-check both with stock prompt
- [ ] GCP project check / quota request for L4 GPU
- [ ] Set up Weights & Biases project

**Afternoon (4h)**
- [ ] Build training data generator script
- [ ] Generate **batch 1: 50 examples** (1 per ~10 templates)
- [ ] Quality review batch 1, iterate on prompts
- [ ] Generate **batch 2: 500 examples** (basic brief→match)

**Night (unattended, 6h)**
- [ ] Generate **full 8000 examples** across 6 categories:
  - Brief → match + reasoning (3000)
  - Comparison reasoning (1000)
  - Local knowledge Q&A (800)
  - Constraint resolution (800)
  - Modification intent (600)
  - Architectural reasoning (1300)
- [ ] Hold out 500 examples as test set with ground truth

**Day 1 Checkpoint:** ~8000 training + 500 test examples in JSONL.

---

### Day 2 (Friday) - Fine-tune + Evaluate

**Morning (1h)**
- [ ] Validate training data (format, length, dedup)
- [ ] Build evaluation harness (eval_finetune.py)

**Morning (start training, runs unattended)**
- [ ] Kick off **E2B LoRA fine-tune**: 3 epochs, batch 4, lr 1e-4
- [ ] Kick off **E4B LoRA fine-tune** (sequential or after E2B done)

**During training (3-4h)**
- [ ] Build GCP Cloud Run skeleton
- [ ] Test deployment with **stock Gemma 4 E4B**
- [ ] Run baseline eval: stock E2B, stock E4B, Claude Haiku
- [ ] Verify endpoint reachable from Windows laptop

**Evening (2h)**
- [ ] Fine-tunes complete -> automated eval on val set
- [ ] Manual spot-check 30 representative outputs
- [ ] Generate scorecard

**Day 2 Decision Gate:**
- Is fine-tuned E2B or E4B meaningfully better than stock?
- Does either pass the quality bar (top-4 acc >=85%, faithfulness >=90%)?
- **Yes** -> proceed to Day 3 with winner
- **No** -> diagnose problem categories, regenerate that data, retrain overnight

---

### Day 3 (Saturday) - Deploy + Integrate

**Morning (2h)**
- [ ] Merge LoRA adapter into base model (winner)
- [ ] Convert to GGUF Q4_K_M
- [ ] Verify quantized model passes same eval (no regression)
- [ ] Push GGUF to GCS bucket

**Afternoon (4h)**
- [ ] Update Cloud Run service to use fine-tuned GGUF
- [ ] Frontend integration:
  - Stage 1 retrieval (MiniLM via ONNX in browser, or small API)
  - Stage 2 LLM call to Cloud Run
  - Streaming response display
  - Fallback to Claude API on endpoint failure

**Evening (2h)**
- [ ] End-to-end testing from Windows laptop
- [ ] Latency profiling (target: <3s p95)

**Day 3 Checkpoint:** Full pipeline live, reachable from Windows.

---

### Day 4 (Sunday) - Polish + Demo Script

**Morning (3h)**
- [ ] Write demo script with **5-7 curated briefs** known to work well
- [ ] Test each brief, iterate response if weak

**Afternoon (3h)**
- [ ] Visual polish: loading states, transitions, error UI
- [ ] Demo theatrics: pre-warm button, "thinking..." animation

**Evening (2h)**
- [ ] **First dress rehearsal** on Windows laptop
- [ ] Note all issues

**Day 4 Checkpoint:** Demo polished, dress rehearsal complete.

---

### Day 5 (Monday) - Freeze + Final Rehearsals

**Morning (2h)**
- [ ] Fix issues from dress rehearsal
- [ ] Code freeze at noon

**Afternoon (2h)**
- [ ] Two more dress rehearsals
- [ ] Verify Cloud Run endpoint warm
- [ ] Snapshot Cloud Run config + GGUF locally

**Evening (1h)**
- [ ] Pre-demo checklist
- [ ] Mac Studio set up as emergency local fallback

---

### Tuesday - Demo
- T-30min: Wake Cloud Run endpoint (`min_instances=1`)
- T-5min: Send warm-up query
- Demo: Live

---

## 4. Training data structure

### Format: ChatML JSONL

```json
{"messages": [
  {"role": "system", "content": "You are a BIM architectural consultant..."},
  {"role": "user", "content": "<brief>\n<10 candidate templates>"},
  {"role": "assistant", "content": "<reasoning + top-4 selection>"}
]}
```

### Six categories (~8000 total)

1. **Brief -> match + reasoning** (3000) - Vary user personas, phrasings, constraints
2. **Comparison reasoning** (1000) - "A vs B for use case X"
3. **Local knowledge Q&A** (800) - "What's typical for Paris haussmann?"
4. **Constraint resolution** (800) - "2-bed under 60 sqm in Europe?"
5. **Modification intent** (600) - "Make the bedroom bigger"
6. **Architectural reasoning** (1300) - "Why is this layout good for this user?"

### Generation method
Use Claude 3.5 Sonnet API. Each template generates ~16 examples across categories. Vary temperature 0.3-0.9 for diversity.

### Held-out test set (500 examples)
- Generated separately, never seen during training
- Each example has ground-truth template ID for top-K accuracy measurement

---

## 5. Evaluation framework

### 6-metric scorecard

| Metric | What it measures | How |
|--------|------------------|-----|
| Top-4 accuracy | Right template in top 4? | Programmatic, ground truth |
| Format validity | Valid JSON output? | JSON parser |
| Faithfulness | No invented facts? | Extract claims, verify against templates |
| Reasoning score | Explanation quality (1-5) | Claude-as-judge on 100 samples |
| Latency p50/p95 | Response speed | Wall-clock |
| Demo prompts | Works on 7 demo briefs? | Manual review |

### Three-way comparison (always)

| System | Top-4 | Format | Faith | Reason | Latency |
|--------|-------|--------|-------|--------|---------|
| Stock Gemma 4 E2B | ? | ? | ? | ?/5 | ? |
| Stock Gemma 4 E4B | ? | ? | ? | ?/5 | ? |
| Fine-tuned E2B | ? | ? | ? | ?/5 | ? |
| Fine-tuned E4B | ? | ? | ? | ?/5 | ? |
| Claude 3.5 Haiku | ? | ? | ? | ?/5 | ? |

This becomes a slide in the demo: "Our fine-tuned model beats stock Gemma 4 by X% on this domain."

### Quality gates

| Top-4 Acc | Faithfulness | Action |
|-----------|--------------|--------|
| >=85% | >=90% | Ship it |
| 70-85% | >=90% | Acceptable, polish demo prompts |
| <70% OR <80% faithful | - | Diagnose, regenerate, retrain |

---

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Fine-tune quality bad | Medium | High | Day 2 gate; regenerate problem categories |
| Cloud Run cold start | High | Medium | min_instances=1 demo day; warm-up query |
| GPU quota denied | Low | High | Request Day 1 morning; HF backup |
| Model says wrong thing on stage | Medium | High | Curated briefs only; pre-test all |
| Network failure | Low | Catastrophic | Mac Studio local fallback |
| Training data poor | Medium | High | Day 1 batch quality check before scaling |
| Frontend bugs | High | Medium | Day 4 dress rehearsal |

---

## 7. Fallback ladder

```
Best:       MiniLM + fine-tuned Gemma 4 on Cloud Run
              v (if Cloud Run fails)
Good:       MiniLM + fine-tuned Gemma 4 served from Mac Studio (tunnel)
              v (if model quality poor)
Acceptable: MiniLM + Claude API with template context
              v (if no internet)
Last resort: MiniLM only, top-4 with stock descriptions
```

Demo always works. Story changes, demo doesn't break.

---

## 8. Cost estimate

| Item | Cost |
|------|------|
| Claude API (training data generation) | ~$40 |
| GCP Cloud Run testing Days 2-5 | ~$15 |
| GCP Cloud Run demo day (warm 8h L4) | ~$8 |
| Buffer / contingency | ~$15 |
| **Total** | **~$75** |

---

## 9. Success criteria

Demo Tuesday is successful if:
- Pipeline responds in <3s on at least 5/7 demo briefs
- Reasoning is correct and faithful (no hallucinated facts)
- Endpoint stays available for demo duration
- Story slide shows fine-tuned model beating stock Gemma 4 measurably

Stretch goals (if time permits):
- Streaming token-by-token output for "thinking" effect
- Multi-turn conversation ("what if I want a bigger kitchen?")
- Image input (multimodal — leverage Gemma 4 capability)
