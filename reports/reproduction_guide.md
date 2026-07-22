# Reproduction Guide

From the repository root:

```powershell
$env:PYTHONPATH='src'
python tools/export_factor_research_schemas.py
python tools/export_factor_discovery_v2_artifacts.py
python -m ruff check .
python -m pytest -q
python -m compileall -q src deploy tools
git diff --check
npm --prefix frontend-bigscreen run build
docker compose -f deploy/nas_research_worker/docker-compose.yml config
```

Create a signed task only in a controlled shadow environment with aligned full
commits and valid signing paths:

```bash
qlab request-factor-research \
  --lake-root /var/lib/quant-lab/lake \
  --queue-root /var/lib/quant-lab/research_queue \
  --key-id "$QUANT_LAB_RESEARCH_TASK_KEY_ID" \
  --quant-lab-commit "$(git -C /opt/quant-lab rev-parse HEAD)" \
  --date auto --start-date auto --end-date auto --max-history-days 730
```

Use the existing worker in run-once mode for shadow validation, then run the
cloud importer first with its validate-only option. Confirm the generation
pointer, dataset generation metadata, row counts, zero live effect, and unchanged
V5 state before a formal import.
