FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN groupadd --gid 10003 quant-download \
    && useradd --uid 10003 --gid 10003 --no-create-home --home-dir /app quant-download
WORKDIR /app
COPY pyproject.toml README.md /app/
COPY src /app/src
RUN pip install --no-cache-dir .
USER 10003:10003
CMD ["uvicorn", "quant_lab.export_download.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
