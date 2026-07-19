# Current NAS Migration Inventory

Captured at `2026-07-19T04:02:00Z`. This report records observed runtime state, not README claims.

## Source identities

| Surface | Observed commit | Working tree |
| --- | --- | --- |
| GitHub `zhr2038/quant-lab` `main` | `49ad71fb9d3043e4882546fd8b8d4ff0ba93106b` | remote reference |
| qyun2 `/opt/quant-lab` | `49ad71fb9d3043e4882546fd8b8d4ff0ba93106b` | clean |
| NAS Research Worker image | `dcbfc22217d48bf80a0a2ea6e1aa55c42d047511` | running image |
| qyun V5 `/home/ubuntu/clawd/v5-prod` | `3df1c67cc44cc8be364ec5d3798ea0d3595c0abc` | clean, read-only for this task |

The qyun2/NAS commit mismatch is currently material: the latest Alpha Factory task is repeatedly rejected by the worker as `worker_code_mismatch`. It is not counted as a successful current-main run.

## Capability inventory

| Capability | State | Runtime evidence |
| --- | --- | --- |
| Expert Export | `PRODUCTION_RESEARCH_RUNNING` | `quant-export-worker` and download services running on NAS; qyun2 receipt import timer active |
| AI Research | `PRODUCTION_RESEARCH_RUNNING` | `quant-ai-worker` running; qyun2 AI task/import timers active |
| Entry Quality History | `PRODUCTION_RESEARCH_RUNNING` | latest task `entry-quality-history-8475bf67a8d388705332fbc2` completed, imported, Anti-Leakage `PASS` |
| Alpha Factory / Second Stage | `FAILED` | previous signed tasks completed at `dcbfc22`; current task `alpha-factory-2f9862f5951430b2a27724d2` loops on `worker_code_mismatch` |
| Factor Research v2 | `CODE_ONLY` | cloud Factor Factory exists, but the signed Research Plane has no `factor_research` task, snapshot, worker dispatch, result, or importer contract |

## Cloud runtime

- Root filesystem: 79 GiB total, 38 GiB used, 38 GiB free, 51% used.
- Memory: 7.4 GiB total, 6.4 GiB available.
- Lake: 13 GiB.
- Research Queue: 75 MiB.
- `quant-lab-alpha-factory-request.timer`: enabled and active.
- `quant-lab-entry-quality-history-request.timer`: enabled and active.
- `quant-lab-research-result-import.timer`: enabled and active.
- Legacy local Alpha Factory timer: disabled.
- `quant-lab-v5-research-refresh.service` still includes cloud-local `qlab build-factor-factory` with `MemoryMax=4G`, `CPUQuota=60%`, and a 45 minute timeout.

## NAS runtime

- Host: UGREEN `HR-Cloud`, x86_64, 15 GiB RAM visible to the OS.
- `/volume1`: 916 GiB total, 803 GiB free, 13% used.
- `quant-research-worker`: healthy, 3 CPUs, 8 GiB memory limit, PID limit 256, read-only root, `RUN_ONCE=false`.
- Research Worker image ID: `sha256:4fa0bb11388a22d2a2a3d27718e8c8d39d6466cda8f30dca2e5ce5e6d24119d0`.
- Research Worker idle observation: about 142 MiB.
- Shared runtime mount: `/volume1/docker/quant-runtime -> /runtime`; configured heavy-job lock defaults to `/runtime/quant-runtime/heavy-job.lock`.
- No exchange credential mount was observed on `quant-research-worker`.

## Latest successful research tasks

| Task | Input | Download | Cache | Output rows | Peak RSS | Compute | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `entry-quality-history-8475bf67a8d388705332fbc2` | 7,452,309 B | 7,429,118 B | 23,191 B | 2,755 | 245,825,536 B | 20.40 s | imported, Anti-Leakage PASS |
| `alpha-factory-cb0df230d9639799882689da` | 10,710,510 B | 10,617,295 B | 93,215 B | 173,947 | 1,003,642,880 B | 212.73 s | imported, Anti-Leakage PASS |
| `alpha-factory-2eb57de8ba8734f0cff95bb6` | 8,640,879 B | 2,651,171 B | 5,989,708 B | 179,966 | 1,022,083,072 B | 223.13 s | imported, Anti-Leakage PASS |

## Current blocker

Task `alpha-factory-2f9862f5951430b2a27724d2` was sealed by cloud commit `49ad71f...`, while the NAS worker enforces exact worker commit `dcbfc22...`. It remains retry-pending with attempt zero because rejection occurs before the claimed status increments. The v2 rollout must align cloud, worker repository, image, and signed task commit before shadow acceptance.

## Safety state

This inventory made no V5, order, position, risk-permission, canary, enforce, or capital changes. Existing NAS research outputs remain `research_only=true` with `live_order_effect=none` and no automatic promotion.
