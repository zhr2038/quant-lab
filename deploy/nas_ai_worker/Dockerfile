FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    WORKER_DATA_DIR=/data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin quantai

WORKDIR /app
COPY deploy/nas_ai_worker/requirements.txt /app/requirements.txt
RUN python -m pip install --no-cache-dir --requirement /app/requirements.txt

COPY src/quant_lab /app/src/quant_lab
COPY deploy/nas_ai_worker/worker.py /app/worker.py

RUN mkdir -p /data && chown -R quantai:quantai /app /data
USER quantai

ENTRYPOINT ["python", "/app/worker.py"]
