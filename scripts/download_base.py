"""固定 RoBERTa-base 的 Hub commit，下载单份 safetensors 权重。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="FacebookAI/roberta-base")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", type=Path, default=Path("base_model"))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    info = HfApi().model_info(args.repo_id, revision=args.revision)
    snapshot_download(
        repo_id=args.repo_id, revision=info.sha, local_dir=output,
        allow_patterns=["config.json", "model.safetensors", "vocab.json", "merges.txt", "tokenizer.json",
                        "tokenizer_config.json", "special_tokens_map.json", "README.md"],
    )
    files = {}
    for path in sorted(output.iterdir()):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            files[path.name] = {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}
    assert "model.safetensors" in files and "config.json" in files
    report = {"repo_id": args.repo_id, "revision": info.sha, "files": files,
              "downloaded_at": datetime.now(timezone.utc).isoformat()}
    (output / "base_model_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
