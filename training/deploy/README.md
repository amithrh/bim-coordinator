# Deployment Guide

End-to-end pipeline from a trained LoRA adapter to a public Cloud Run endpoint.

## Pipeline overview

```
LoRA adapter (training/checkpoints/<run>/)
  │
  │ scripts/merge_and_export.py
  ▼
Fused MLX model (training/exports/<run>/fused/)
  │
  ▼ (mlx_lm fuse --de-quantize -> bf16 safetensors)
HuggingFace format (training/exports/<run>/hf/)
  │
  ▼ (llama.cpp convert_hf_to_gguf.py + llama-quantize)
GGUF Q4_K_M (training/exports/<run>/gguf/<run>.Q4_K_M.gguf)
  │
  ▼ (cloud_run/deploy.sh)
Cloud Run + L4 GPU + Ollama (https://bim-slm-...run.app)
```

## Step 1: Merge LoRA + export

```bash
source .venv/bin/activate
python training/scripts/merge_and_export.py \
  --adapter training/checkpoints/gemma4-e4b-lora-r32-iters1500-XXXXX
```

Produces:
- `training/exports/<run>/fused/` - MLX merged model
- `training/exports/<run>/hf/` - HF safetensors
- `training/exports/<run>/gguf/<run>.Q4_K_M.gguf` - quantized GGUF

If llama.cpp isn't installed:
```bash
git clone https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp && cmake -B build && cmake --build build -j
```

## Step 2: Local sanity check with Ollama

```bash
# Build the local Ollama model from the GGUF
cp training/exports/<run>/gguf/<run>.Q4_K_M.gguf training/deploy/cloud_run/model.gguf
cd training/deploy/cloud_run
ollama create bim-slm -f Modelfile

# Test it
echo '{"model":"bim-slm","prompt":"BRIEF: 2-bed in Athens for a couple\n...","stream":false}' | \
  ollama generate
```

## Step 3: Deploy to Cloud Run

```bash
export PROJECT_ID=your-gcp-project
export GGUF_PATH=training/exports/<run>/gguf/<run>.Q4_K_M.gguf

# One-time GCP setup
gcloud auth login
gcloud config set project $PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# Build + deploy
bash training/deploy/cloud_run/deploy.sh
```

The script:
1. Stages Dockerfile + Modelfile + GGUF in a temp build context
2. Builds with `gcloud builds submit` (Cloud Build does the docker build)
3. Deploys to Cloud Run with NVIDIA L4 GPU, 16Gi RAM, scale-to-zero

## Cost expectations

L4 GPU on Cloud Run: ~$0.60/hr while serving requests.
Demo day: warm 30 min before, demo for 1 hr -> **~$1**.

To keep warm during testing days, set `--min-instances 1` (charges per hour even idle).

## Verify the endpoint

```bash
URL=$(gcloud run services describe bim-slm --region us-central1 --format='value(status.url)')

# Quick test
curl -X POST "$URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "bim-slm",
    "messages": [
      {"role": "user", "content": "BRIEF: 2-bed in Athens for a couple\n\nCANDIDATES: ..."}
    ]
  }'
```

## Run eval against the deployed endpoint

```bash
python training/eval/eval_model.py \
  --backend openai \
  --model bim-slm \
  --base-url $URL \
  --test training/data/processed/test.jsonl \
  --limit 50 \
  --out training/eval/cloud_run_results.json
```

## Backup: HuggingFace Inference Endpoint

If Cloud Run has issues:

1. Push fused model to HF Hub:
   ```bash
   hf upload --private myuser/bim-slm-private training/exports/<run>/hf/
   ```
2. Go to https://endpoints.huggingface.co/ -> New Endpoint -> select your repo
3. Pick T4 GPU, $0.60/hr
4. Copy endpoint URL, point demo at it

## Local fallback (Mac Studio)

If both cloud paths fail:
```bash
ollama serve
ollama create bim-slm -f training/deploy/cloud_run/Modelfile
# Expose via ngrok or Tailscale
ngrok http 11434
```
Demo points at the ngrok URL. Mac Studio runs the model.
