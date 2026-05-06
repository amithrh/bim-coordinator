# Cloud Run + L4 GPU + Ollama serving the BIM Coordinator fine-tuned model.
#
# Build:
#   gcloud builds submit --tag gcr.io/$PROJECT/bim-slm:latest training/deploy/cloud_run
#
# Deploy:
#   gcloud run deploy bim-slm \
#     --image gcr.io/$PROJECT/bim-slm:latest \
#     --gpu 1 --gpu-type nvidia-l4 --no-cpu-throttling \
#     --memory 16Gi --cpu 4 \
#     --max-instances 1 --min-instances 0 \
#     --timeout 300 --concurrency 4 \
#     --region us-central1
#
# The image expects the GGUF file to be present in /models/ at runtime.
# We mount it from a GCS bucket via gcsfuse (sidecar) or bake it into the
# image. For the demo we bake it in for reliability.

FROM ollama/ollama:latest

# Switch to root so we can prepare files
USER root

# Pre-create dirs
RUN mkdir -p /models /root/.ollama

# Copy the GGUF model and Modelfile that we built locally
COPY model.gguf /models/model.gguf
COPY Modelfile /models/Modelfile

# Pre-pull the model into Ollama's cache so cold starts are fast
RUN ollama serve & \
    sleep 3 && \
    ollama create bim-slm -f /models/Modelfile && \
    pkill ollama

# Listen on Cloud Run's port
ENV OLLAMA_HOST=0.0.0.0:8080
ENV OLLAMA_KEEP_ALIVE=24h
EXPOSE 8080

# Cloud Run health check needs HTTP on $PORT; Ollama responds on /api/tags
ENTRYPOINT ["/bin/ollama"]
CMD ["serve"]
