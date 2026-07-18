# ruff: noqa: E501
"""Collect a secret-safe, read-only snapshot from the V5 production host."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

REMOTE_SOURCE = r'''
import hashlib, json, os, re, sqlite3, subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/ubuntu/clawd/v5-prod")
SENSITIVE = re.compile(r"(?:api[_-]?key|secret|pass(?:word|phrase)?|token|private[_-]?key|database[_-]?(?:url|password)|dsn)", re.I)
STATE = re.compile(r"(?:account|alpha|candidate|gate|risk|permission|weight|regime|hmm|rss|universe|position|optimizer|decision|signal|status)", re.I)
SKIP = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache"}
HISTORY_PARTS = {"archive", "archives", "backup", "backups", "bundles", "exports", "output", "transfer", "tests", "testdata", "fixtures"}

def run(args, cwd=None, timeout=30):
    try:
        p = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr[-2000:]}
    except Exception as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}

def sha_bytes(data): return hashlib.sha256(data).hexdigest()
def sha_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def redact(value, key=""):
    if SENSITIVE.search(str(key)):
        raw = "" if value is None else str(value)
        return {"present": value not in (None, ""), "sha256": sha_bytes(raw.encode()) if raw else ""}
    if isinstance(value, dict): return {str(k): redact(v, str(k)) for k,v in value.items()}
    if isinstance(value, list): return [redact(v, key) for v in value[:200]]
    if isinstance(value, str) and (SENSITIVE.search(value) or ("BEGIN " in value and "PRIVATE KEY" in value)):
        return {"present": bool(value), "sha256": sha_bytes(value.encode()), "redacted_embedded_sensitive_text": True}
    if isinstance(value, str) and len(value) > 4000: return {"sha256": sha_bytes(value.encode()), "length": len(value)}
    return value

git_head = run(["git","rev-parse","HEAD"], REPO)["stdout"].strip()
git_branch = run(["git","branch","--show-current"], REPO)["stdout"].strip()
git_status = run(["git","status","--porcelain=v1","--untracked-files=all"], REPO)["stdout"].splitlines()
tracked = run(["git","ls-files"], REPO)["stdout"].splitlines()
tracked_set=set(tracked)
untracked = run(["git","ls-files","--others","--exclude-standard"], REPO)["stdout"].splitlines()

unit_names=[]
unit_hashes=[]
unit_runtime=[]
environment_presence={}
for scope, prefix in (("user", ["systemctl","--user"]),("system", ["systemctl"])):
    listing=run(prefix+["list-unit-files","--no-legend","--no-pager"])
    for line in listing["stdout"].splitlines():
        name=line.split()[0] if line.split() else ""
        if name and re.search(r"(?:v5|quant)", name, re.I): unit_names.append((scope,name))
for scope,name in sorted(set(unit_names)):
    prefix=["systemctl","--user"] if scope=="user" else ["systemctl"]
    cat=run(prefix+["cat",name])
    unit_hashes.append({"scope":scope,"unit":name,"sha256":sha_bytes(cat["stdout"].encode()),"readable":cat["returncode"]==0})
    show=run(prefix+["show",name,"-p","Id","-p","ActiveState","-p","SubState","-p","UnitFileState","-p","MainPID","-p","ExecMainStartTimestamp","-p","FragmentPath","-p","EnvironmentFiles"])
    props={}
    for line in show["stdout"].splitlines():
        if "=" in line:
            k,v=line.split("=",1); props[k]=v
    pid=int(props.get("MainPID") or 0)
    if pid>0:
        try:
            raw=Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            for item in raw:
                if b"=" not in item: continue
                key,val=item.split(b"=",1)
                key=key.decode(errors="replace"); value=val.decode(errors="replace")
                environment_presence[f"{scope}:{name}:{key}"]={"present":bool(value),"sha256":sha_bytes(value.encode()) if value else "","sensitive_name":bool(SENSITIVE.search(key))}
            cmdline=[part.decode(errors="replace") for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part]
            props["runtime_argv"]=[redact(arg,"runtime_arg" if not SENSITIVE.search(arg) else "secret") for arg in cmdline]
        except Exception as exc: props["process_snapshot_error"]=str(exc)
    unit_runtime.append({"scope":scope,"unit":name,**props})

config_hashes=[]
state_candidates=[]
db_schemas=[]
for path in REPO.rglob("*"):
    try:
        rel=path.relative_to(REPO)
        if any(part in SKIP for part in rel.parts): continue
        if any(part.lower() in HISTORY_PARTS for part in rel.parts[:-1]): continue
        if not path.is_file(): continue
        name=path.name.lower()
        if str(rel) in tracked_set and (path.suffix.lower() in {".yaml",".yml",".toml",".ini",".cfg",".service",".timer"} or name.startswith(".env") or "config" in name):
            config_hashes.append({"path":str(rel),"sha256":sha_file(path),"size":path.stat().st_size})
        state_scope = len(rel.parts) <= 3 and (len(rel.parts) == 1 or rel.parts[0].lower() in {"reports","runtime","state","config"})
        if state_scope and path.suffix.lower() in {".json"} and STATE.search(name) and path.stat().st_size <= 2_000_000:
            try:
                obj=json.loads(path.read_text(errors="replace"))
                selected={}
                def walk(current,prefix=""):
                    if isinstance(current,dict):
                        for k,v in current.items():
                            key=f"{prefix}.{k}" if prefix else str(k)
                            if STATE.search(str(k)) and not isinstance(v,(dict,list)): selected[key]=redact(v,str(k))
                            walk(v,key)
                    elif isinstance(current,list):
                        for i,v in enumerate(current[:100]): walk(v,f"{prefix}[{i}]")
                walk(obj)
                state_candidates.append({"path":str(rel),"sha256":sha_file(path),"mtime":datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),"selected_values":selected,"document":redact(obj)})
            except Exception: pass
        if state_scope and "pre-" not in name and path.suffix.lower() in {".db",".sqlite",".sqlite3"} and path.stat().st_size <= 5_000_000_000:
            entry={"path":str(rel),"sha256":sha_file(path),"size":path.stat().st_size,"mtime":datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()}
            try:
                con=sqlite3.connect(f"file:{path}?mode=ro",uri=True,timeout=2)
                entry["user_version"]=con.execute("pragma user_version").fetchone()[0]
                tables=[r[0] for r in con.execute("select name from sqlite_master where type='table' order by name")]
                entry["tables"]=tables
                schemas=[]
                state_rows={}
                for table in tables:
                    sql=con.execute("select sql from sqlite_master where type='table' and name=?",(table,)).fetchone()[0] or ""
                    schemas.append(table+":"+sql)
                    if STATE.search(table) and not SENSITIVE.search(table):
                        try:
                            cols=[r[1] for r in con.execute(f'pragma table_info("{table}")')]
                            rows=con.execute(f'select * from "{table}" limit 50').fetchall()
                            state_rows[table]=[
                                {col:redact(value,"secret_payload" if isinstance(value,str) and SENSITIVE.search(value) else col) for col,value in zip(cols,row)} for row in rows
                            ]
                        except Exception as exc: state_rows[table]={"error":str(exc)}
                entry["schema_sha256"]=sha_bytes("\n".join(schemas).encode())
                entry["state_rows"]=state_rows
                if "alembic_version" in tables:
                    entry["alembic_version"]=con.execute("select version_num from alembic_version").fetchall()
                con.close()
            except Exception as exc: entry["error"]=str(exc)
            db_schemas.append(entry)
    except Exception:
        continue

state_candidates=sorted(state_candidates,key=lambda row:row["mtime"],reverse=True)[:300]
db_schemas=sorted(db_schemas,key=lambda row:row.get("mtime",""),reverse=True)[:30]

docker_rows=[]
docker=run(["docker","ps","--no-trunc","--format","{{json .}}"])
if docker["returncode"]==0:
    for line in docker["stdout"].splitlines():
        try:
            row=json.loads(line); cid=row.get("ID")
            inspect=run(["docker","inspect",cid])
            detail=json.loads(inspect["stdout"])[0] if inspect["returncode"]==0 else {}
            env={}
            for item in ((detail.get("Config") or {}).get("Env") or []):
                if "=" in item:
                    k,v=item.split("=",1); env[k]={"present":bool(v),"sha256":sha_bytes(v.encode()) if v else "","sensitive_name":bool(SENSITIVE.search(k))}
            docker_rows.append({"container_id":cid,"name":row.get("Names"),"image":row.get("Image"),"image_id":detail.get("Image"),"status":row.get("Status"),"command_sha256":sha_bytes(str((detail.get("Config") or {}).get("Cmd") or []).encode()),"environment":env})
        except Exception: pass

raw_diff=run(["git","diff","HEAD","--no-ext-diff","--no-color"],REPO,timeout=60)["stdout"]
known_values=[]
for meta in environment_presence.values(): pass
# Redact assignment-like sensitive lines and private-key material.  Environment
# raw values never entered the snapshot object in the first place.
safe_lines=[]
for line in raw_diff.splitlines():
    if "BEGIN " in line and "PRIVATE KEY" in line:
        safe_lines.append("<REDACTED_PRIVATE_KEY_LINE>"); continue
    if SENSITIVE.search(line) and ("=" in line or ":" in line):
        prefix=line[:1] if line[:1] in "+- " else ""
        safe_lines.append(prefix+"<REDACTED_SENSITIVE_LINE sha256="+sha_bytes(line.encode())+">")
    else: safe_lines.append(line)
safe_diff="\n".join(safe_lines)+("\n" if raw_diff else "")

payload={
 "captured_at":datetime.now(timezone.utc).isoformat(),"host":"qyun.hrhome.top","repo_path":str(REPO),
 "git":{"head":git_head,"branch":git_branch,"status_porcelain":git_status,"tracked_files":tracked,"untracked_files":untracked,"diff_sha256":sha_bytes(safe_diff.encode())},
 "systemd":{"unit_hashes":unit_hashes,"runtime":unit_runtime},"containers":docker_rows,
 "config_hashes":config_hashes,"database_schemas":db_schemas,"state_documents":state_candidates,
 "environment_variable_presence":environment_presence,
 "collection_semantics":{"read_only":True,"production_mutations":0,"orders_submitted":0,"secret_values_stored":False},
 "v5_git_diff_patch":safe_diff,
}
print(json.dumps(payload,ensure_ascii=False,default=str))
'''


def _write_lines(path: Path, rows: list[dict], keys: list[str]) -> None:
    lines = ["\t".join(str(row.get(key, "")) for key in keys) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    if not os.environ.get("SSHPASS"):
        raise SystemExit("SSHPASS must be provided in the process environment")
    root = Path(os.environ.get("AUDIT_ROOT", "/home/hr/quant-alpha-audit-v2"))
    manifests = root / "manifests"
    artifacts = root / "artifacts"
    manifests.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    command = [
        "sshpass",
        "-e",
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
        "ubuntu@qyun.hrhome.top",
        "python3",
        "-",
    ]
    completed = subprocess.run(
        command,
        input=REMOTE_SOURCE,
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"remote snapshot failed: {completed.stderr[-2000:]}")
    payload = json.loads(completed.stdout)
    patch = payload.pop("v5_git_diff_patch")
    serialized = json.dumps(payload, ensure_ascii=False)
    forbidden = [
        os.environ.get("SSHPASS", ""),
        "-----BEGIN PRIVATE KEY-----",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
    ]
    for value in forbidden:
        if value and (value in serialized or value in patch):
            raise SystemExit("secret leak scan failed; refusing to write snapshot")
    (manifests / "v5_runtime_snapshot.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (manifests / "v5_git_diff.patch").write_text(patch, encoding="utf-8")
    unit_hashes = payload["systemd"]["unit_hashes"]
    _write_lines(
        manifests / "systemd_units_sha256.txt",
        unit_hashes,
        ["scope", "unit", "sha256", "readable"],
    )
    _write_lines(
        manifests / "container_image_digests.txt",
        payload["containers"],
        ["container_id", "name", "image", "image_id", "status"],
    )
    _write_lines(
        manifests / "config_sha256.txt",
        payload["config_hashes"],
        ["path", "sha256", "size"],
    )
    (manifests / "environment_variable_presence.json").write_text(
        json.dumps(payload["environment_variable_presence"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    missing: list[str] = []
    if payload["git"]["status_porcelain"]:
        missing.append("uncommitted and/or untracked files are identified but untracked contents are not embedded")
    if not payload["database_schemas"]:
        missing.append("no runtime database schema could be read")
    if not payload["state_documents"]:
        missing.append("no dynamic state JSON documents could be read")
    missing.extend(
        [
            "per-decision sentiment inputs are not durably snapshotted for the complete historical interval",
            "per-decision dynamic IC weights and regime/HMM/RSS state are not durably versioned as one atomic generation",
            "per-decision optimizer inputs, pre-trade positions, target weights, and executable universe are not bound to one immutable receipt",
            "environment secrets are intentionally represented by hashes only and cannot recreate exchange connectivity",
        ]
    )
    replayability = "PARTIALLY_REPLAYABLE"
    status = {
        "captured_at": payload["captured_at"],
        "replayability": replayability,
        "deployment_readiness": "FAIL",
        "git_head": payload["git"]["head"],
        "git_dirty": bool(payload["git"]["status_porcelain"]),
        "systemd_units_captured": len(unit_hashes),
        "containers_captured": len(payload["containers"]),
        "config_hashes_captured": len(payload["config_hashes"]),
        "database_files_captured": len(payload["database_schemas"]),
        "state_documents_captured": len(payload["state_documents"]),
        "missing_state": missing,
        "production_mutations": 0,
        "orders_submitted": 0,
        "reason": "The present runtime is substantially fingerprinted, but the historical per-decision state chain needed for exact V5 replay does not exist as immutable atomic snapshots.",
    }
    (artifacts / "v5_replayability_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
