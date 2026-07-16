# NAS Expert Pack Download

This LAN/VPN-only service reads `accepted_index.json`. The Python process never streams
Pack bytes. After HMAC validation it sends `X-Accel-Redirect` to unprivileged Nginx.

Create:

- `secrets/nas-download.key` with at least 32 random bytes and mode `0400`;
- `secrets/download.htpasswd` with an operator account and mode `0440`;
- `.env` from `.env.example`, binding only a LAN or VPN address.

```bash
docker compose build --pull
docker compose up -d
curl http://127.0.0.1:8788/healthz
```

Do not publish the port through router NAT, public tunnels, or the qyun2 reverse proxy.
