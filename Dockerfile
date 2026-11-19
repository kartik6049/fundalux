# Fundalux web app: one container, CPU only, no PyTorch.
# Build after exporting the models to ONNX:  python -m fundalux.export models
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FUNDALUX_MODELS=/app/models FUNDALUX_REVIEW_DIR=/app/review PORT=8000
WORKDIR /app

COPY requirements-server.txt .
RUN pip install -r requirements-server.txt

COPY fundalux/ fundalux/
COPY server/ server/
COPY samples/ samples/
COPY models/config.json models/metrics.json models/sites.json models/
COPY models/onnx/ models/onnx/
COPY models/reports/ models/reports/
COPY models/figures/ models/figures/
COPY review/ review/

# Run as an unprivileged user. It may write only models/sites.json (site calibration) and review/
# (clinician answers). Mount volumes on both to keep them across redeploys.
RUN useradd --create-home fundalux && chown fundalux /app/models /app/models/sites.json && chown -R fundalux /app/review
USER fundalux

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=20s --start-period=60s --retries=3 CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/health')"
CMD ["sh", "-c", "uvicorn server.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
