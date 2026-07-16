# NAS Export Worker

This container pulls signed tasks and sealed snapshot blobs from the cloud and generates
Expert Packs entirely on NAS. It does not call exchange APIs and has no trading secrets.

## Host preparation

```bash
install -d -m 2750 /volume/path/quant-export/data/{accepted,blobs,snapshots,work,status,audit}
install -d -m 2750 /volume/path/quant-export/secrets /volume/path/quant-runtime
chown -R 10002:10002 /volume/path/quant-export/data /volume/path/quant-runtime
printf '%s\n' '{"schema_version":"quant_lab_export_accepted_index.v1","packs":[]}' \
  > /volume/path/quant-export/data/accepted_index.json
chown 10002:10002 /volume/path/quant-export/data/accepted_index.json
chmod 0640 /volume/path/quant-export/data/accepted_index.json
```

Install the SSH key, strict `known_hosts`, NAS receipt-signing private key, and cloud task
public key under `secrets/` with mode `0400`. Do not put keys in `.env` or Git.

Copy `.env.example` to `.env`, replace placeholders, and set the exact deployed full Git
SHA in `BUILD_GIT_COMMIT`.

```bash
docker compose build --pull
docker compose up -d
docker compose logs -f quant-export-worker
```

The worker refuses a task when `expected_worker_commit` differs from its build commit.
`SNAPSHOT_FETCH_WORKERS` controls independent, resumable snapshot streams. Use `4` by
default and raise it only after checking qyun2 SSH load and NAS network stability.
