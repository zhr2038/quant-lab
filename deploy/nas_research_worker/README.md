# NAS Entry Quality History Research Worker

This container pulls signed, sealed Entry Quality History snapshots from qyun2,
computes research-only derived data, signs the result, and uploads it for strict
cloud validation. It has no exchange credentials, does not mount the cloud Lake,
and cannot publish Gold or influence live orders.

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
```

Install these files under `secrets/` with mode `0400` and ownership readable by
UID 10004:

* `id_ed25519`: dedicated outbound-only SSH key for the cloud `quant-research` account;
* `known_hosts`: pinned qyun2 SSH host key;
* `cloud_research_task_public_key`: cloud task-verification Ed25519 public key;
* `nas_research_signing_key`: NAS result-signing Ed25519 private key.

Never place keys in `.env`, Compose, an image layer, or Git. Copy `.env.example`
to `.env`, set `BUILD_GIT_COMMIT` to the exact deployed 40-character commit, then:

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
