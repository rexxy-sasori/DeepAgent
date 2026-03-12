# DeepAgent Dockerfile
# Optimized for Kubernetes deployment
# Uses SGLang base image for better compatibility

# Build argument for base image
ARG BASE_IMAGE=harbor.xa.xshixun.com:7443/hanfeigeng/lmsysorg/sglang:kv-cache-logging-dev-otel-0.8

# ============================================
# Stage: Runtime
# ============================================
FROM ${BASE_IMAGE} as runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Install any additional system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglib2.0-0 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy application code
COPY . .

# Install DeepAgent-specific dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/outputs /app/logs

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data'); \
               nltk.download('averaged_perceptron_tagger', download_dir='/app/nltk_data'); \
               nltk.download('wordnet', download_dir='/app/nltk_data')"

ENV NLTK_DATA=/app/nltk_data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command (overridden in K8s)
CMD ["python", "src/run_deep_agent.py", "--config_path", "./config/base_config.yaml", "--help"]
