FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/Mohan-mlk/ad-performance-mlops"

# libgomp is LightGBM's OpenMP runtime -- the image builds without it and then
# fails at import time, which is a miserable way to find out.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY data/reference/ ./data/reference/
COPY data/processed/ ./data/processed/
COPY scripts/start-api.sh scripts/start-dashboard.sh ./scripts/

RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://localhost:%s/ready' % os.getenv('PORT','8000'))"

CMD ["uvicorn", "adperf.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
