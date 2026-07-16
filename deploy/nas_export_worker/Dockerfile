FROM python:3.12-slim

ARG BUILD_GIT_COMMIT=unknown
ARG BUILD_TIMESTAMP=unknown
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POLARS_MAX_THREADS=3 \
    BUILD_GIT_COMMIT=${BUILD_GIT_COMMIT} \
    BUILD_TIMESTAMP=${BUILD_TIMESTAMP} \
    EXPORT_SCHEMA_VERSION=quant_lab.expert_pack.v1

RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10002 quant-export \
    && useradd --uid 10002 --gid 10002 --no-create-home --home-dir /app quant-export

WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir .

USER 10002:10002
ENTRYPOINT ["python", "-m", "quant_lab.export_worker.runner"]
