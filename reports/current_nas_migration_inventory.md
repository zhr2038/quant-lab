# Current NAS Migration Inventory

Captured at `2026-07-19T12:48:01Z`. This is observed runtime state from Git,
SSH, systemd, Docker, queue, and Lake inspection. It does not infer deployment
from documentation.

## Source identities

| Surface | Observed commit | State |
| --- | --- | --- |
| GitHub `main` | `49ad71fb9d3043e4882546fd8b8d4ff0ba93106b` | unchanged; this refactor is not merged |
| GitHub operational runtime commit | `1bd35db43c71298e37d6ae8ef16a694a94ebaab3` | pushed on the refactor branch |
| qyun2 `/opt/quant-lab` | `1bd35db43c71298e37d6ae8ef16a694a94ebaab3` | clean shadow deployment |
| NAS Research Worker repo/image | `1bd35db43c71298e37d6ae8ef16a694a94ebaab3` | exact worker commit |
| qyun V5 | `3df1c67cc44cc8be364ec5d3798ea0d3595c0abc` | clean and read-only |

The original cloud/worker mismatch was resolved before acceptance. The two
accepted signed tasks remain bound to `8d8c884ad9fc35ff600921fc8c0a05bf732cff72`;
the later `1bd35db` deployment is an operational Web-refresh fix and is not
presented as new Factor or Alpha evidence.

## Capability inventory

| Capability | State | Runtime evidence |
| --- | --- | --- |
| Expert Export | `PRODUCTION_RESEARCH_RUNNING` | NAS export worker and download services are running and healthy |
| AI Research | `PRODUCTION_RESEARCH_RUNNING` | NAS AI worker is running |
| Entry Quality History | `PRODUCTION_RESEARCH_RUNNING` | existing request timer remains active; protocol tests remain compatible |
| Alpha Factory / Second Stage | `PRODUCTION_RESEARCH_RUNNING` | final task `alpha-factory-c3b0425791b3ce7800ed1470` imported, Anti-Leakage `PASS` |
| Factor Research v2 | `PRODUCTION_RESEARCH_RUNNING` | final task `factor-research-097a452bfeb6edfd8baf5d1e` imported, Anti-Leakage `PASS`; weekly timer active |

## Cloud runtime

- Root filesystem: 79 GiB total, 41 GiB used, 35 GiB free, 54% used.
- Memory: 7.4 GiB total, about 6.3 GiB available at capture time.
- Lake: 13,592,119,349 bytes (about 12.7 GiB).
- Research Queue: 101,766,175 bytes (about 97 MiB).
- Factor request, Alpha request, result import, Entry Quality request, API, and
  Web services/timers were active after acceptance.
- The V5 research refresh no longer calls cloud-local `build-factor-factory`.
  It is capped at 40% CPU and 1.2 GB memory.
- Its Web-derived final stage now loads 40 declared inputs instead of all 190
  registered datasets. The full service completed in 704.4 seconds with
  `Result=success`; full-chain swap peak fell by 73.15%.
- Factor snapshot request is capped at 30% CPU and 900 MB memory.
- Result import is capped at 40% CPU and 1.2 GB memory.

## NAS runtime

- Host: UGREEN `HR-Cloud`, x86_64, 15.4 GiB RAM.
- `/volume1`: about 916 GiB total, 796 GiB available, 13% used.
- Research Worker image ID:
  `sha256:28c7e80192cdbdfb8f3debb7d5496e88f60fb9250d6579f7f762a35276a149e7`.
- Research Worker: 3 CPUs, 8 GiB memory limit, PID limit 256, read-only root.
- The persistent compose container is intentionally `created`/idle; controlled
  shadow tasks used the same image through `docker compose run --rm`.
- `quant-ai-worker`, `quant-export-worker`, and export download services were
  running. No exchange credential mount was added to the Research Worker.

## Final accepted tasks

| Task | Input | Download | Cache | Output rows | Peak RSS | Compute | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `factor-research-097a452bfeb6edfd8baf5d1e` | 388,332 B | 311,072 B | 77,260 B | 22,122 | 354,426,880 B | 24.88 s | imported, Anti-Leakage PASS |
| `alpha-factory-c3b0425791b3ce7800ed1470` | 16,295,243 B | 16,220,714 B | 74,529 B | 307,138 | 1,439,264,768 B | 20.65 s | imported, Anti-Leakage PASS |

The Alpha task binds Factor generation
`factor-research-097a452bfeb6edfd8baf5d1e` and digest
`82655622935aa4fcfc6e153162ff4c4edb7450b41f098b8cfdf8043eb5511f52`.

## Current research truth

- Four independent hypotheses are registered; two are active and two are
  `DATA_BLOCKED` because real derivatives or microstructure inputs are absent.
- The current generation contains eight confirmatory trials. All eight are
  `REJECTED_DATA_QUALITY`; FDR passes, `SIGNAL_VALID`, and `PAPER_CANDIDATE` are
  all zero.
- Point-in-time cost coverage is 74.7907%, below the 80% gate. The available
  historical universe context is also materially shorter than the requested
  two-year window.
- Factor attribution and portfolio tables contain eight current rows each and
  have zero null primary keys.
- Alpha output contains 165 rows: 65 `KEEP_SHADOW`, 78 `KILL`, and 22
  `RESEARCH`. Every row has zero live notional and requires manual live approval.

## Safety state

No V5 file or runtime state was changed. Both generations are
`research_only=true`, `live_order_effect=none`, and
`automatic_promotion=false`. No Paper ACK/Tracker, risk permission, canary,
enforce, capital, position, or exchange-order path was changed.
