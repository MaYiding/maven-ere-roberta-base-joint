"""下载官方 MAVEN-ERE v1.0，校验文档划分并记录文件哈希。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

TSINGHUA = "https://cloud.tsinghua.edu.cn/f/a7d1db6c44ea458bb6f0/?dl=1"
GOOGLE_ID = "1fxomYO6zPl5DDrDr_HeWFK14s8BpW1z-"
EXPECTED = {"train": 2913, "valid": 710, "test": 857}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/MAVEN_ERE"))
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    target = args.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    archive = args.archive.resolve() if args.archive else target.parent / "MAVEN_ERE.zip"
    source = "existing archive"
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        partial = archive.with_suffix(".partial")
        print("Downloading official MAVEN-ERE v1.0", flush=True)
        try:
            with urllib.request.urlopen(TSINGHUA, timeout=120) as response, partial.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            if not zipfile.is_zipfile(partial):
                raise ValueError("Tsinghua endpoint returned a non-ZIP response")
            source = TSINGHUA
        except Exception as error:
            print(f"Primary endpoint unavailable ({type(error).__name__}); using official Google Drive mirror", flush=True)
            partial.unlink(missing_ok=True)
            subprocess.run([sys.executable, "-m", "gdown", GOOGLE_ID, "-O", str(partial)], check=True)
            if not zipfile.is_zipfile(partial):
                raise ValueError("Official mirror did not return a ZIP archive")
            source = f"https://drive.google.com/file/d/{GOOGLE_ID}/view"
        partial.replace(archive)

    # 只提取约定的三个 JSONL 文件，不执行或展开压缩包中的其他内容。
    with zipfile.ZipFile(archive) as bundle:
        for split in EXPECTED:
            name = f"{split}.jsonl"
            matches = [item for item in bundle.infolist() if Path(item.filename).name == name and not item.is_dir()]
            if len(matches) != 1:
                raise ValueError(f"Expected one {name}, found {len(matches)}")
            destination = target / name
            if not destination.exists():
                with bundle.open(matches[0]) as source_stream, destination.open("wb") as output:
                    shutil.copyfileobj(source_stream, output)

    manifest = {
        "dataset": "MAVEN-ERE", "version": "1.0", "download_source": source,
        "archive_sha256": sha256(archive), "created_at": datetime.now(timezone.utc).isoformat(),
        "test_labels_public": False, "splits": {},
    }
    ids: dict[str, set[str]] = {}
    for split, expected in EXPECTED.items():
        path = target / f"{split}.jsonl"
        ids[split] = set()
        mentions = 0
        relation_counts: dict[str, int] = {}
        count = 0
        with path.open() as stream:
            for line in stream:
                doc = json.loads(line)
                count += 1
                if doc["id"] in ids[split]:
                    raise ValueError(f"Duplicate document ID in {split}")
                ids[split].add(doc["id"])
                if split != "test":
                    assert all(key in doc for key in ("events", "temporal_relations", "causal_relations", "subevent_relations"))
                    mentions += sum(len(event["mention"]) for event in doc["events"])
                    for task in ("temporal", "causal"):
                        relation_counts[task] = relation_counts.get(task, 0) + sum(len(pairs) for pairs in doc[f"{task}_relations"].values())
                    relation_counts["subevent"] = relation_counts.get("subevent", 0) + len(doc["subevent_relations"])
                else:
                    assert "event_mentions" in doc and "events" not in doc
                    mentions += len(doc["event_mentions"])
        if count != expected:
            raise ValueError(f"Unexpected {split} count: {count}, expected {expected}")
        manifest["splits"][split] = {"file": path.name, "sha256": sha256(path), "documents": count,
                                      "mentions_or_test_candidates": mentions, "relation_counts": relation_counts}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if ids[left] & ids[right]:
            raise ValueError(f"Document overlap between {left} and {right}")
    manifest["document_splits_disjoint"] = True
    (target / "data_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
