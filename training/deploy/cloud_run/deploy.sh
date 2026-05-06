#!/usr/bin/env bash
# Build, push, and deploy the BIM Coordinator SLM to Cloud Run with L4 GPU.
#
# Required env vars:
#   PROJECT_ID       - GCP project id
#   REGION           - default: us-central1 (L4 availability)
#   SERVICE_NAME     - default: bim-slm
#   GGUF_PATH        - path to the .gguf file built locally
#
# Prereqs:
#   gcloud auth login
#   gcloud config set project $PROJECT_ID
#   gcloud services enable run.googleapis.com cloudbuild.googleapis.com

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?must set PROJECT_ID}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-bim-slm}"
GGUF_PATH="${GGUF_PATH:?must set GGUF_PATH to your .gguf file}"

if [ ! -f "$GGUF_PATH" ]; then
  echo "GGUF file not found: $GGUF_PATH" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Stage build context (Dockerfile + Modelfile + GGUF) in a temp dir so we
# don't pollute the source tree with a 2GB binary.
BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

cp "$SCRIPT_DIR/Dockerfile" "$BUILD_DIR/Dockerfile"
cp "$SCRIPT_DIR/Modelfile" "$BUILD_DIR/Modelfile"
cp "$GGUF_PATH" "$BUILD_DIR/model.gguf"

IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:$(date +%Y%m%d-%H%M%S)"
LATEST="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"

echo "=== Building image: $IMAGE ==="
gcloud builds submit --tag "$IMAGE" "$BUILD_DIR"
gcloud container images add-tag "$IMAGE" "$LATEST" -q

echo
echo "=== Deploying to Cloud Run: $SERVICE_NAME ==="
gcloud run deploy "$SERVICE_NAME" \
  --image "$LATEST" \
  --region "$REGION" \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --no-cpu-throttling \
  --memory 16Gi \
  --cpu 4 \
  --max-instances 1 \
  --min-instances 0 \
  --timeout 300 \
  --concurrency 4 \
  --port 8080 \
  --allow-unauthenticated

URL="$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format='value(status.url)')"

echo
echo "=== Deployed ==="
echo "URL: $URL"
echo
echo "Quick test:"
echo "  curl -X POST $URL/api/generate -d '{\"model\":\"bim-slm\",\"prompt\":\"hi\"}' --no-buffer"
echo
echo "Production OpenAI-compatible endpoint:"
echo "  $URL/v1/chat/completions"
echo
echo "Demo day: warm the endpoint 5 min before by sending a small request."
