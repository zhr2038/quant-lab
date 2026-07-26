FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POLARS_MAX_THREADS=3 \
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

ARG BUILD_GIT_COMMIT
ARG BUILD_TIMESTAMP=unknown
RUN test -n "$BUILD_GIT_COMMIT" \
    && test "${#BUILD_GIT_COMMIT}" = 40 \
    && printf '%s\n' "$BUILD_GIT_COMMIT" > /app/BUILD_GIT_COMMIT \
    && chmod 0444 /app/BUILD_GIT_COMMIT
ENV BUILD_GIT_COMMIT=${BUILD_GIT_COMMIT} \
    BUILD_TIMESTAMP=${BUILD_TIMESTAMP}

USER 10002:10002
ENTRYPOINT ["python", "-m", "quant_lab.export_worker.runner"]
