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

Copy `.env.example` to `.env` and replace the secret or host-specific placeholders.
The deployment script writes the repository's exact full Git SHA to `BUILD_GIT_COMMIT`
without exposing or replacing the other environment values.

```bash
./deploy_current_head.sh
docker compose logs -f quant-export-worker
```

Each task binds `expected_worker_commit` to the sealed snapshot's full quant-lab commit.
The worker refuses the task when its build commit differs, so cloud and NAS deployments
remain fail-closed without a separately maintained commit setting.
Always use `deploy_current_head.sh` after updating the NAS repository; manually running
`docker compose up` can leave an old image commit active and will correctly cause
`worker_code_mismatch` when a newer cloud task arrives.
`SNAPSHOT_FETCH_WORKERS` controls independent, resumable snapshot streams. Use `4` by
default and raise it only after checking qyun2 SSH load and NAS network stability.
Each stream retries interrupted transfers. `SNAPSHOT_TRANSFER_IDLE_SECONDS` closes a
connection that remains alive without writing data; verified blob-cache batches are reused.
