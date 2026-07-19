# NAS Factor Research Plane

Factor Research extends the existing signed Research Plane with the discriminated
task type `factor_research`. It reuses the same queue, Ed25519 keys, snapshots,
blob cache, lease, heartbeat, heavy-job lock, worker container, importer, snapshot
GC, and durable atomic publication machinery.

Cloud owns hypothesis approval, trial registration, immutable snapshot sealing,
task signing, result validation, publication, and generation pointers. NAS owns
pure bounded computation only. The NAS cannot write Gold, register hypotheses,
publish candidates, promote Paper, modify V5, or access exchange credentials.

The task binds full cloud/worker commit, registry and ledger digests, source input
digest, date range, hypothesis IDs, trial IDs, test count, and all safety fields.
The result must return the exact output/report set, hashes, schemas, row counts,
resource metrics, and an all-PASS anti-leakage report.

The request timer is weekly and remains disabled until two real shadow tasks pass.
NAS offline behavior is wait-and-retry; there is no automatic cloud fallback.
