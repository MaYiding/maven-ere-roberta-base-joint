"""训练结束后生成 HF 模型目录和代码归档；此脚本不进行任何上传。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def load_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run_dir.resolve()
    summary = load_json(run / "training_summary.json")
    if summary["status"] not in {"completed", "early_stopped"} or summary.get("smoke"):
        raise RuntimeError("Only a successfully completed full training run can be packaged")
    best = (run / "best_joint").resolve(strict=True)
    if not best.is_relative_to(run):
        raise RuntimeError("Best checkpoint must belong to this training run")
    metrics = load_json(best / "dev_metrics.json")
    provenance = load_json(root / "source_manifest.json")
    recipe = load_json(run / "training_config.json")
    data = load_json(root / "data/MAVEN_ERE/data_manifest.json")
    base = load_json(root / "base_model/base_model_manifest.json")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing release: {output}")
    output.mkdir(parents=True)
    hf = output / "huggingface"
    hf.mkdir()
    # HF 导出仅包含模型、tokenizer 和评测汇总，完整语料及预测留在训练目录。
    allowed = {".safetensors", ".json", ".py", ".txt"}
    for path in best.iterdir():
        if path.is_file() and path.suffix in allowed:
            shutil.copy2(path, hf / path.name)
    for name in ("LICENSE", "CITATION.cff", "UPSTREAM_NOTICE.md", "requirements-lock.txt"):
        shutil.copy2(root / name, hf / name)
    shutil.copy2(root / "notices/ROBERTA-MIT-LICENSE", hf / "ROBERTA-MIT-LICENSE")
    for key in ("base_model", "data_dir", "output_dir", "cache_dir", "resume", "prediction_file", "checkpoint"):
        recipe.pop(key, None)
    # 配方仅保留可复现信息，不公开内部机器绝对路径。
    recipe["base_model"] = {"repo_id": base["repo_id"], "revision": base["revision"]}
    recipe["dataset"] = {"name": data["dataset"], "version": data["version"], "splits": data["splits"]}
    recipe["code_commit"] = provenance["code_commit"]
    recipe["upstream_commit"] = provenance["upstream_commit"]
    (hf / "training_recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    shutil.copy2(root / "source_manifest.json", hf / "source_manifest.json")
    scores = metrics["official_protocol"]
    rows = [f"| {task} | {scores[task][task + '_precision']:.2f} | {scores[task][task + '_recall']:.2f} | {scores[task][task + '_f1']:.2f} |"
            for task in ("temporal", "causal", "subevent")]
    card = f'''---
language:
- en
library_name: transformers
base_model: FacebookAI/roberta-base
tags:
- event-relation-extraction
- maven-ere
- custom_code
- roberta
license: gpl-3.0
---

# MAVEN-ERE Joint RoBERTa-base

This is a four-task joint model adapted from THU-KEG/MAVEN-ERE. It requires
pre-identified event mentions and temporal expressions; it does not detect all
events from raw text. It predicts coreference, temporal, causal and subevent
relations within English documents. Subevent pairs are [parent, child].

Training uses seed {recipe['seed']}, microbatch {recipe['batch_size']}, gradient
accumulation {recipe['grad_accum']}, {'BF16' if recipe['bf16'] else 'FP32'}, a {recipe['max_length']}-token maximum chunk length and
one NVIDIA GeForce RTX 4090 D. The completed run performed
{summary['update_step']} optimizer updates across {summary['epoch']} completed epochs
with early stopping permitted. See training_recipe.json for exact settings,
data hashes, loss reductions, base revision and code provenance.

## Validation results

These are **validation**, not test, results from one shared checkpoint. The model
was selected by the upstream evaluator's overall F1 on the 710-document valid
split. The training labels symmetrize SIMULTANEOUS/BEGINS-ON, while the official
evaluation uses the original annotated directions.

| Task | Precision | Recall | F1 |
|---|---:|---:|---:|
{chr(10).join(rows)}

Coreference mean F1 (B3, CEAF-e, MUC, BLANC): {scores['coreference_mean_f1']:.2f}.
Overall F1: {scores['overall_f1']:.2f}. Detailed per-metric values are in
dev_metrics.json. Test labels are hidden; **no test score is claimed**.

## Loading and prediction

Install the training repository requirements, then use its train_joint.py
--predict-only entry point for the complete tokenizer/span alignment and JSONL
decoding pipeline. The model itself can be loaded with AutoModel.from_pretrained
and trust_remote_code=True after reviewing the included custom source. It needs
input_ids, attention_mask, event_spans, splits and events_idx, not just raw text.

## Implementation changes and limitations

The scorer hidden size is 200, matching upstream code (the paper appendix states
150). This implementation fixes missing first-mention dummy labels, stable
coreference marginal loss, optimizer-step scheduling and complete resume state.
The nominal effective batch is {recipe['batch_size'] * recipe['grad_accum']} documents; the {recipe['batch_size']}-document microbatch differs
from the upstream README's 8-document microbatch and changes relation-pair
weighting. This is a documented adaptation, not an exact paper replication.
The data is English Wikipedia and covers within-document relations. Predicted
causal links reflect the text and are not independently verified causal facts.

## Attribution

The derivative code follows GPL-3.0; retain LICENSE and UPSTREAM_NOTICE.md.
The RoBERTa base model is credited separately under its MIT notice. Full corpus
text is not redistributed here. Cite MAVEN-ERE (Wang et al., EMNLP 2022), MAVEN
(Wang et al., EMNLP 2020), and RoBERTa (Liu et al., 2019).
'''
    (hf / "README.md").write_text(card)
    code_archive = output / "training-code.tar.gz"
    with tarfile.open(code_archive, "w:gz") as bundle:
        for relative, expected in provenance["files"].items():
            path = (root / relative).resolve(strict=True)
            if not path.is_relative_to(root) or digest(path) != expected:
                raise RuntimeError(f"Training source changed since launch: {relative}")
            bundle.add(path, arcname=relative, recursive=False)
        bundle.add(root / "source_manifest.json", arcname="source_manifest.json", recursive=False)
    files = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(output))] = {"sha256": digest(path), "bytes": path.stat().st_size}
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "training_status": summary["status"],
                "code_commit": provenance["code_commit"], "validation_only": True,
                "files": files, "uploaded": False}
    (output / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"release": str(output), "files": len(files), "uploaded": False}, indent=2))


if __name__ == "__main__":
    main()
