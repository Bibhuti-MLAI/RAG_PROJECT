# Legal Document Assistant - container image
# Build & run with:  docker compose up --build -d
FROM python:3.11-slim-bookworm

# System packages:
#   tesseract-ocr (+ language packs)  -> OCR for uploaded images
#   curl                              -> used by the health check
# Package sources are switched to HTTPS first: some networks (e.g. corporate
# proxies) block plain-HTTP downloads from deb.debian.org with "403 Forbidden".
RUN sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-fra \
        tesseract-ocr-deu \
        tesseract-ocr-spa \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only build of PyTorch first. sentence-transformers would
# otherwise pull the GPU build, which adds several GB to the image.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code. Secrets (my.env) are NOT copied into the image; they are
# passed in at run time by docker compose (env_file).
COPY api.py simple_rag.py index.html ./
COPY static/ ./static/

# Run as a normal user instead of root, and give it only the folders it writes to.
RUN useradd --create-home appuser \
    && mkdir -p legal_documents legal_vectordb \
    && chown -R appuser:appuser /app
USER appuser

# Download the embedding model at build time so the container starts quickly.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

ENV PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/test || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
