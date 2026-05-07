#!/usr/bin/env bash
# Autonomous overnight orchestrator for SLM training.
#
# Runs end-to-end:
#   1. Real LoRA fine-tune on Llama 3.2 3B (1500 iters, ~1.5h)
#   2. Eval fine-tuned Llama 3.2 3B vs stock baseline
#   3. If Gemma 4 E2B downloaded -> baseline + fine-tune Gemma E2B (~2-3h)
#   4. If Gemma 4 E4B downloaded -> baseline + fine-tune Gemma E4B (~3-5h)
#   5. Final 3-way (or N-way) scorecard comparison
#   6. Quantize the WINNER to GGUF Q4_K_M ready for deployment
#   7. Write MORNING_STATUS.md
#
# Idempotent — each step writes a marker file, skipped if already done.
# Tolerates failures: each step continues to next on error, logged in status.

set -uo pipefail

REPO_ROOT="/Users/amitmishra/worksppace-central/demo-bimplus/bim-coordinator"
cd "$REPO_ROOT"
source .venv/bin/activate

LOG_DIR="$REPO_ROOT/training/overnight_logs"
mkdir -p "$LOG_DIR"
STATUS_FILE="$REPO_ROOT/MORNING_STATUS.md"
ORCH_LOG="$LOG_DIR/orchestrator.log"

log() {
  local msg="[$(date '+%H:%M:%S')] $*"
  echo "$msg" | tee -a "$ORCH_LOG"
}

write_status() {
  python "$REPO_ROOT/training/scripts/write_status.py" 2>>"$ORCH_LOG" || true
}

# -----------------------------------------------------------------------------
# Step helpers
# -----------------------------------------------------------------------------

run_step() {
  local name="$1"; shift
  local marker="$LOG_DIR/.${name}.done"
  if [ -f "$marker" ]; then
    log "[skip] $name (marker exists: $marker)"
    return 0
  fi
  log "=== START $name ==="
  if "$@"; then
    log "=== DONE $name ==="
    touch "$marker"
    write_status
    return 0
  else
    log "!!! FAILED $name (exit $?) — continuing"
    return 1
  fi
}

# -----------------------------------------------------------------------------
# Training step
# -----------------------------------------------------------------------------

train_lora() {
  local model_short="$1"   # LLAMA32 | GEMMA_E2B | GEMMA_E4B
  local model_path="$2"
  local iters="$3"
  local batch="$4"
  local rank="$5"
  local lora_layers="$6"

  local lower_short=$(echo "$model_short" | tr '[:upper:]' '[:lower:]')
  local run_name="${lower_short}-real-r${rank}-iters${iters}-$(date +%s)"
  local adapter_path="$REPO_ROOT/training/checkpoints/$run_name"
  local config_path="$adapter_path/config.yaml"
  mkdir -p "$adapter_path"

  python -c "
import sys
sys.path.insert(0, 'training/scripts')
from train_lora import _write_lora_config
from pathlib import Path
_write_lora_config(
    Path('$config_path'),
    model='$model_path',
    data_dir=Path('$REPO_ROOT/training/data/processed'),
    adapter_path=Path('$adapter_path'),
    iters=$iters,
    batch_size=$batch,
    lr=1e-4,
    lora_layers=$lora_layers,
    rank=$rank,
    alpha=$((rank*2)).0,
    val_batches=10,
    save_every=200,
    seed=42,
)
"

  local log_file="$LOG_DIR/train_${model_short}.log"
  log "Training $model_short (iters=$iters, batch=$batch, rank=$rank)"
  log "Log: $log_file"
  log "Adapter: $adapter_path"

  # Run with caffeinate to prevent system sleep
  caffeinate -is python -m mlx_lm lora --config "$config_path" 2>&1 | tee "$log_file"
  local rc=${PIPESTATUS[0]}

  if [ "$rc" -ne 0 ]; then
    log "Training $model_short failed with exit $rc"
    return $rc
  fi

  # Save which adapter was just produced (for downstream eval)
  echo "$adapter_path" > "$LOG_DIR/.last_adapter_${model_short}.path"
  return 0
}

# -----------------------------------------------------------------------------
# Eval step
# -----------------------------------------------------------------------------

run_eval() {
  local label="$1"
  local model_path="$2"
  local adapter_arg="$3"   # empty string for stock, "--adapter-path X" for fine-tuned
  local out_file="$REPO_ROOT/training/eval/${label}.json"

  log "Eval: $label (full 600 test)"
  python "$REPO_ROOT/training/eval/eval_model.py" \
    --backend mlx \
    --model "$model_path" \
    --test "$REPO_ROOT/training/data/processed/test.jsonl" \
    --out "$out_file" \
    $adapter_arg \
    2>&1 | tee "$LOG_DIR/eval_${label}.log"
  return ${PIPESTATUS[0]}
}

# -----------------------------------------------------------------------------
# Gemma availability checks: prefer bf16 (best quality), fall back to 4-bit.
# -----------------------------------------------------------------------------

gemma_bf16_ready() {
  local size="$1"  # e2b or e4b
  local need
  if [ "$size" = "e2b" ]; then need=3; else need=4; fi
  local count=$(ls ~/.cache/huggingface/hub/models--mlx-community--gemma-4-${size}-it-bf16/snapshots/*/model-*.safetensors 2>/dev/null | wc -l | tr -d ' ')
  [ "$count" -ge "$need" ]
}

gemma_4bit_ready() {
  local size="$1"
  # 4-bit models are typically a single .safetensors file
  ls ~/.cache/huggingface/hub/models--mlx-community--gemma-4-${size}-it-4bit/snapshots/*/model.safetensors 2>/dev/null | head -1 | grep -q safetensors
}

# Returns the best available model path, or empty if none ready
gemma_best_path() {
  local size="$1"
  if gemma_bf16_ready "$size"; then
    echo "mlx-community/gemma-4-${size}-it-bf16"
    return 0
  fi
  if gemma_4bit_ready "$size"; then
    echo "mlx-community/gemma-4-${size}-it-4bit"
    return 0
  fi
  echo ""
  return 1
}

# Wait up to N seconds for ANY Gemma version to be ready
wait_for_gemma() {
  local size="$1"
  local timeout="${2:-1800}"   # default 30 min
  local elapsed=0
  while [ "$elapsed" -lt "$timeout" ]; do
    if [ -n "$(gemma_best_path $size)" ]; then
      return 0
    fi
    sleep 60
    elapsed=$((elapsed + 60))
  done
  return 1
}

# -----------------------------------------------------------------------------
# Main flow
# -----------------------------------------------------------------------------

log "============================================"
log "OVERNIGHT ORCHESTRATOR START"
log "============================================"
write_status

# Step 1: Real Llama 3.2 3B fine-tune (always - this is the safety net)
run_step "train_llama32" \
  train_lora "LLAMA32" "mlx-community/Llama-3.2-3B-Instruct-4bit" 1500 2 32 16

# Step 2: Eval fine-tuned Llama 3.2 3B
if [ -f "$LOG_DIR/.last_adapter_LLAMA32.path" ]; then
  llama_adapter=$(cat "$LOG_DIR/.last_adapter_LLAMA32.path")
  run_step "eval_llama32_finetuned" \
    run_eval "finetuned_llama3.2-3b" \
      "mlx-community/Llama-3.2-3B-Instruct-4bit" \
      "--adapter-path $llama_adapter"
fi

# Step 3: Gemma E2B — wait up to 30 min for ANY version (bf16 or 4-bit)
log "Checking Gemma E2B availability (wait up to 30 min for download)..."
if wait_for_gemma "e2b" 1800; then
  e2b_path=$(gemma_best_path "e2b")
  log "Gemma E2B available at: $e2b_path"
  run_step "eval_gemma_e2b_stock" \
    run_eval "baseline_gemma-e2b" "$e2b_path" ""

  run_step "train_gemma_e2b" \
    train_lora "GEMMA_E2B" "$e2b_path" 1500 2 32 16

  if [ -f "$LOG_DIR/.last_adapter_GEMMA_E2B.path" ]; then
    e2b_adapter=$(cat "$LOG_DIR/.last_adapter_GEMMA_E2B.path")
    run_step "eval_gemma_e2b_finetuned" \
      run_eval "finetuned_gemma-e2b" "$e2b_path" \
        "--adapter-path $e2b_adapter"
  fi
else
  log "[skip] Gemma E2B not downloaded after 30 min — sticking with Llama"
fi

# Step 4: Gemma E4B — wait up to 30 min, prefer bf16, fall back to 4-bit
log "Checking Gemma E4B availability (wait up to 30 min for download)..."
if wait_for_gemma "e4b" 1800; then
  e4b_path=$(gemma_best_path "e4b")
  log "Gemma E4B available at: $e4b_path"
  run_step "eval_gemma_e4b_stock" \
    run_eval "baseline_gemma-e4b" "$e4b_path" ""

  # E4B is 2x bigger - use batch_size=1 for memory safety
  run_step "train_gemma_e4b" \
    train_lora "GEMMA_E4B" "$e4b_path" 1500 1 32 12

  if [ -f "$LOG_DIR/.last_adapter_GEMMA_E4B.path" ]; then
    e4b_adapter=$(cat "$LOG_DIR/.last_adapter_GEMMA_E4B.path")
    run_step "eval_gemma_e4b_finetuned" \
      run_eval "finetuned_gemma-e4b" "$e4b_path" \
        "--adapter-path $e4b_adapter"
  fi
else
  log "[skip] Gemma E4B not downloaded after 30 min"
fi

# Step 5: Pick winner and quantize
log "============================================"
log "ORCHESTRATOR COMPLETE — see MORNING_STATUS.md"
log "============================================"
write_status
