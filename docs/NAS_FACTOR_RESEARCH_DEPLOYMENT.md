# NAS Factor Research Deployment

## Cloud

Install the factor request service and timer alongside the existing Research
Plane units. Keep the timer disabled initially. Set the new environment switch
to `0` until code, worker image, and full signed commit are aligned.

## NAS

Rebuild the existing `quant-research-worker`; do not create a new worker. The
container keeps 3 CPUs, 8 GiB memory, PID limit 256, read-only root, no privilege
escalation, no exchange credentials, and the shared heavy-job lock.

## Acceptance

Run two tasks in controlled run-once mode. For each task verify signature,
snapshot digest, registry/ledger digest, exact output set, all-PASS anti-leakage,
peak RSS, idempotent import, atomic generation, no queue duplication, and zero
live effect. Only after both pass may the weekly request timer be enabled.

## Rollback

Disable the Factor Research request timer and set
`QUANT_LAB_NAS_FACTOR_RESEARCH_ENABLED=0`. Do not enable cloud fallback
automatically. Existing signed results and Gold generations remain auditable.
