# NAS Research Worker

This container pulls signed, sealed Entry Quality History, Alpha Factory,
Factor Research, Factor Factory, V5 Candidate Evidence, and Trade-Level History
snapshots from qyun2, computes research-only derived data, signs the result, and
uploads it for strict cloud validation. It has no exchange credentials, does
not mount the cloud Lake, and cannot publish Gold or influence live orders.
Factor Factory claims are skipped unless
`QUANT_RESEARCH_FACTOR_FACTORY_ENABLED=1`. V5 Candidate Evidence claims are
likewise skipped unless
`QUANT_RESEARCH_V5_CANDIDATE_EVIDENCE_ENABLED=1`; those results contain only
Candidate Label and Evidence Sample deltas, never PAPER/LIVE decisions.
Trade-Level History claims require
`QUANT_RESEARCH_TRADE_LEVEL_HISTORY_ENABLED=1` and can return only causal
Trade Opportunity Labels and Similarity Outcomes.

## Host preparation

```bash
install -d -m 2750 \
  /volume1/docker/quant-research/data/{blobs/sha256,snapshots,work,results,archive,rejected,status} \
  /volume1/docker/quant-research/secrets
chown -R 10004:10004 /volume1/docker/quant-research/data
install -d -m 2770 -o 10002 -g 10002 /volume1/docker/quant-runtime
touch /volume1/docker/quant-runtime/heavy-job.lock
chown 10002:10002 /volume1/docker/quant-runtime/heavy-job.lock
chmod 0660 /volume1/docker/quant-runtime/heavy-job.lock

# The worker reads repository identity only; it does not need the worktree.
find /volume1/docker/quant-research/repo/.git -type d \
  -exec setfacl -m u:10004:rx,d:u:10004:rx {} +
find /volume1/docker/quant-research/repo/.git -type f \
  -exec setfacl -m u:10004:r {} +
```

Install these files under `secrets/` with mode `0400` and ownership readable by
UID 10004:

* `id_ed25519`: dedicated outbound-only SSH key for the cloud `quant-research` account;
* `known_hosts`: pinned qyun2 SSH host key;
* `cloud_research_task_public_key`: cloud task-verification Ed25519 public key;
* `nas_research_signing_key`: NAS result-signing Ed25519 private key.

Never place keys in `.env`, Compose, an image layer, or Git. Copy `.env.example`
to `.env`, keep `NAS_RESEARCH_SECRETS_HOST_PATH` pointed at the host secrets
directory above, set `NAS_RESEARCH_IMAGE_GIT_COMMIT` to the exact deployed
40-character commit, and set `NAS_RESEARCH_REPOSITORY_GIT_PATH` to the `.git`
directory of that exact checked-out repository. Do not add a runtime
`QUANT_RESEARCH_WORKER_COMMIT` override. The worker compares the immutable image
commit file, both image-provided commit environment values, and repository HEAD
before it polls or claims any task. The repository ACL above is read-only for
UID 10004 and its default directory entries keep future Git objects readable;
do not grant the worker write access to the repository. Then:

```bash
docker compose build --pull
docker compose run --rm -e RUN_ONCE=true quant-research-worker
docker compose up -d
docker compose logs -f quant-research-worker
```

The default container limit is 8 GB to leave headroom, while acceptance still
requires observed peak RSS to remain below 6 GB. `heavy-job.lock` is shared with
the Export Worker, so only one NAS heavy job computes at a time. The dedicated
`10004:10004` user receives supplemental group `10002` only for this setgid
runtime directory; it receives no Docker or sudo access. A NAS outage leaves
tasks waiting; there is intentionally no automatic cloud fallback.
