"""Verify an immutable release before a service starts. Never opens trading stores."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def verify(root: Path, manifest: dict, *, dependencies=False, runtime=False):
    root = root.resolve()
    if (
        manifest.get("schema_version") != "review.release.v1"
        or len(manifest.get("code_revision", "")) != 40
    ):
        raise ValueError("release version manifest missing or invalid")
    errors = []
    for relative, expected in manifest["files"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(relative + ":missing_or_outside_release")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(relative + ":hash_mismatch")
    if dependencies:
        for package, expected in manifest["dependencies"].items():
            try:
                actual = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                actual = None
            if actual != expected:
                errors.append("dependency:" + package + ":version_mismatch")
    if runtime:
        for relative, required in manifest.get("required_runtime_values", {}).items():
            try:
                value = json.loads((root / relative).read_text(encoding="utf-8"))
                if any(value.get(key) != expected for key, expected in required.items()):
                    errors.append(relative + ":runtime_override_mismatch")
            except (OSError, ValueError):
                errors.append(relative + ":runtime_override_unobservable")
    if errors:
        raise ValueError("release verification failed: " + ";".join(errors))
    return {
        "verified": True,
        "code_revision": manifest["code_revision"],
        "file_count": len(manifest["files"]),
        "dependencies_verified": dependencies,
        "runtime_overrides_verified": runtime,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--dependencies", action="store_true")
    parser.add_argument("--runtime", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    manifest = json.loads((root / "release-manifest.json").read_text(encoding="utf-8"))
    print(json.dumps(verify(root, manifest, dependencies=args.dependencies, runtime=args.runtime)))


if __name__ == "__main__":
    main()
