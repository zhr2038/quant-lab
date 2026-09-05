FROM python:3.11-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3
ARG CODE_REVISION
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client rsync \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY deploy/locks/nas-worker.txt /tmp/worker-lock.txt
COPY src ./src
RUN pip install --no-cache-dir -r /tmp/worker-lock.txt \
    && pip install --no-cache-dir --no-deps --no-build-isolation .
ENV QUANT_LAB_CODE_REVISION=$CODE_REVISION PYTHONUNBUFFERED=1 POLARS_MAX_THREADS=2
LABEL org.opencontainers.image.revision=$CODE_REVISION
ENTRYPOINT ["python", "-m", "quant_lab.decision.jobs", "nas-worker"]
