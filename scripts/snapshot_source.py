"""在启动前记录已提交的训练代码版本及每个源文件的 SHA-256。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def git(root, *arguments):
    return subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("source_manifest.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    blocked = [Path("/Users/mayiding/Library/Mobile Documents"), Path("/Users/mayiding/Library/CloudStorage"),
               Path("/Users/mayiding/Library/Application Support/CloudDocs")]
    if any(root == path or path in root.parents for path in blocked):
        raise RuntimeError("Protected working directory")
    if git(root, "status", "--porcelain", "--untracked-files=normal").strip():
        raise RuntimeError("Commit training source before creating its provenance snapshot")
    files = {}
    for name in git(root, "ls-files", "-z").decode().split("\0"):
        if not name:
            continue
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root) or any(path == base or base in path.parents for base in blocked):
            raise RuntimeError("Source file resolves outside the authorized project")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"upstream_repository": "https://github.com/THU-KEG/MAVEN-ERE",
                "upstream_commit": "ac81a9711a69f43f55bfbc50b3bb573fd11c64b0",
                "code_commit": git(root, "rev-parse", "HEAD").decode().strip(),
                "created_at": datetime.now(timezone.utc).isoformat(), "files": files}
    args.output.resolve().write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Snapshotted {len(files)} source files at {manifest['code_commit']}")


if __name__ == "__main__":
    main()
