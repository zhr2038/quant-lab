# Remaining Risks

1. The current Factor generation is correctly `DATA_QUALITY_BLOCKED`: all eight
   trials are `REJECTED_DATA_QUALITY`, point-in-time cost coverage is 74.7907%,
   and available historical universe context is much shorter than the requested
   two-year research window.
2. Funding and microstructure hypotheses remain `DATA_BLOCKED`. Proxy OHLCV
   fields must not be substituted for real funding, basis, depth, spread, or
   aggressive-flow history.
3. There are zero new `SIGNAL_VALID` and zero `PAPER_CANDIDATE` factors. The
   accepted platform cannot be described as profitable or ready for Paper,
   Canary, or live use.
4. qyun2 systemd enforces 900 MB snapshot and 1.2 GB import limits, but this host
   does not retain inactive oneshot cgroup `MemoryPeak`. Exact cloud peak RSS
   needs explicit telemetry in a future operational change.
5. The V5 research refresh is now bounded and no longer runs Factor Factory, but
   the remaining multi-command refresh is still long. A live run reached
   995,586,048 bytes memory peak and 1,993,785,344 bytes swap peak while staying
   under `MemoryMax`; runtime and swap pressure need separate optimization.
6. The append-only ledger contains historical compatibility rows, including a
   cancelled pre-final task lineage. They remain for audit and are not silently
   deleted; dashboards must scope current metrics to the generation trial IDs.
7. PBO and DSR remain inconclusive with few valid variants or periods. They must
   continue to block Paper review rather than default to pass.
8. The accepted code is deployed from a review branch. GitHub `main` remains at
   `49ad71f`; production shadow evidence must not be represented as main-branch
   acceptance before review and merge.
9. A correct research platform does not guarantee profitable factors. Zero new
   valid factors remains an acceptable result.
