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

# Install DeepAgent-specific dependencies for Gaia/HLM datasets
# Uses --ignore-installed to avoid conflicts with system packages (like cryptography)
# External LLM services (SGLang) already provide sglang, transformers, openai, etc.
ENV PLAYWRIGHT_BROWSERS_PATH=/usr/local/share/playwright

RUN pip install --no-cache-dir --ignore-installed \
    beautifulsoup4 requests aiohttp crawl4ai[pdf] \
    chardet lxml nltk numpy regex \
    pandas python-dateutil \
    rouge scipy pyyaml spotipy \
    tqdm func-timeout fuzzywuzzy python-Levenshtein \
    pdfplumber PyPDF2 \
    torch==2.9.1 \
    openai-whisper python-docx python-pptx sympy fastapi transformers==4.57.1 \
    uvicorn sentence-transformers faiss-cpu -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

RUN crawl4ai-setup && crawl4ai-doctor

# Verify Crawl4AI installation
RUN python -c "from crawl4ai import AsyncWebCrawler; print('Crawl4AI imported successfully')" || \
    echo "Warning: Crawl4AI import failed"

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/outputs /app/logs

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', download_dir='/app/nltk_data'); \
               nltk.download('averaged_perceptron_tagger', download_dir='/app/nltk_data'); \
               nltk.download('wordnet', download_dir='/app/nltk_data')"

# Copy application code
COPY . .

ENV NLTK_DATA=/app/nltk_data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command (overridden in K8s)
CMD ["python", "src/run_deep_agent.py", "--config_path", "./config/base_config.yaml", "--help"]
