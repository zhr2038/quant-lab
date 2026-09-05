FROM python:3.11-slim-bookworm
ARG CODE_REVISION
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client rsync \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
ENV QUANT_LAB_CODE_REVISION=$CODE_REVISION PYTHONUNBUFFERED=1 POLARS_MAX_THREADS=2
LABEL org.opencontainers.image.revision=$CODE_REVISION
ENTRYPOINT ["python", "-m", "quant_lab.decision.jobs", "nas-worker"]
