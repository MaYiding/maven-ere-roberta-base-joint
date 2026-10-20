#!/usr/bin/env python3
"""Reproducible four-task MAVEN-ERE training with official-protocol dev scoring.

This trains on supplied event mentions (not end-to-end event detection). The
coreference loss is a mean over mentions, then a mean over documents, matching
the upstream reduction while fixing missing dummy targets. Relation losses are
means over each microbatch's non-ignored ordered pairs. Consequently changing
microbatch size with the same nominal effective batch changes pair weighting.
Checkpoints are saved only at optimizer update boundaries, including mid-epoch.
"""

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoConfig, AutoModel, RobertaTokenizer
from transformers import get_linear_schedule_with_warmup

from joint.src.data import (
    CAUSALREL2ID, ID2CAUSALREL, ID2SUBEVENTREL, ID2TEMPREL,
    SUBEVENTREL2ID, TEMPREL2ID, collator, myDataset,
)
from joint.src.utils import fill_expand, get_predicted_clusters
from maven_ere import MavenEreConfig, MavenEreModel


LOGGER = logging.getLogger("maven_ere.train")
LOSS_WEIGHTS = {"coreference": 0.4, "temporal": 2.0, "causal": 4.0, "subevent": 4.0}
PIPELINE_VERSION = "maven-ere-joint-v2-dummy-fix-padding-trim"
TASK_MAPS = {"temporal": TEMPREL2ID, "causal": CAUSALREL2ID, "subevent": SUBEVENTREL2ID}
REVERSE_MAPS = {"temporal": ID2TEMPREL, "causal": ID2CAUSALREL, "subevent": ID2SUBEVENTREL}
STOP_REQUESTED = False
SELECTION_TASKS = ("coreference", "temporal", "causal", "subevent")
COREFERENCE_METRICS = ("muc", "b_cubed", "ceaf", "blanc")
NEW_SIGNATURE_DEFAULTS = {
    "scorer_hidden_size": 200, "scorer_num_layers": 3, "positive_class_weight": 1.0,
    "export_task_bests": False, "early_stopping_monitor": "overall",
    "early_stopping_patience": 10, "eval_every_updates": 0,
    "eval_epoch_end": True, "coreference_selection": "mean",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def load_checkpoint(path):
    """Training states contain Python/NumPy RNG objects; only load trusted files."""
    return torch.load(path, map_location="cpu", weights_only=False)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # Keep the attention implementation fixed across resume and HF roundtrips.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def validate_splits(data_dir):
    """Validate document disjointness; test is read only for IDs, never scored."""
    split_ids, digests = {}, {}
    for split in ("train", "valid", "test"):
        path = Path(data_dir) / f"{split}.jsonl"
        if not path.exists():
            if split != "test":
                raise FileNotFoundError(path)
            continue
        ids = set()
        with open(path, encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["id"] in ids:
                    raise ValueError(f"Duplicate document ID in {split}:{line_number}")
                ids.add(record["id"])
                if split in ("train", "valid") and "events" not in record:
                    raise ValueError(f"{split}:{line_number} has no gold events/relations")
        if not ids:
            raise ValueError(f"Empty {split} split")
        for previous, previous_ids in split_ids.items():
            if ids & previous_ids:
                raise ValueError(f"Document overlap between {previous} and {split}")
        split_ids[split] = ids
        digests[split] = {"sha256": sha256_file(path), "documents": len(ids)}
    return digests


def cached_dataset(tokenizer, data_dir, split, max_length, cache_dir, split_digest):
    vocab_hash = hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()
    identity = {"pipeline": PIPELINE_VERSION, "split": split, "data_sha256": split_digest,
                "vocab_sha256": vocab_hash, "max_length": max_length,
                "tokenizer_class": type(tokenizer).__name__,
                "tokenizer_init": tokenizer.init_kwargs,
                "special_tokens": tokenizer.special_tokens_map}
    cache_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    cache_path = Path(cache_dir) / f"{split}-{cache_hash[:20]}.pt"
    if cache_path.exists():
        LOGGER.info("Loading %s tokenization cache %s", split, cache_path)
        payload = load_checkpoint(cache_path)
        if payload["identity"] != identity:
            raise ValueError(f"Tokenization cache identity mismatch: {cache_path}")
        dataset = myDataset.__new__(myDataset)
        dataset.tokenized_samples = payload["samples"]
    else:
        LOGGER.info("Tokenizing %s (cache %s)", split, cache_path)
        dataset = myDataset(tokenizer, str(data_dir), split, max_length=max_length)
        atomic_torch_save(cache_path, {"identity": identity, "samples": dataset.tokenized_samples})
        # Original document JSON is not needed in training workers.
        del dataset.examples
        del dataset.tokenizer
    if not len(dataset):
        raise ValueError(f"No nonempty {split} examples")
    return dataset, cache_hash


def choose_smoke_indices(dataset, limit=64):
    """Cover long documents and actual positive labels without synthetic labels."""
    selected = []
    def add(index):
        if index not in selected:
            selected.append(index)
    add(max(range(len(dataset)), key=lambda i: dataset[i]["input_ids"].shape[0]))
    add(max(range(len(dataset)), key=lambda i: sum(map(len, dataset[i]["event_spans"]))))
    for task, label_map in TASK_MAPS.items():
        none_id = label_map["NONE"]
        index = next((i for i in range(len(dataset))
                      if bool(((dataset[i][task + "_labels"] >= 0) &
                               (dataset[i][task + "_labels"] != none_id)).any())), None)
        if index is None:
            raise ValueError(f"Smoke data has no positive {task} labels")
        add(index)
    coref = next((i for i in range(len(dataset)) if any(len(c) > 1 for c in dataset[i]["coref_labels"])), None)
    if coref is None:
        raise ValueError("Smoke data has no non-singleton coreference cluster")
    add(coref)
    for index in range(len(dataset)):
        if len(selected) >= min(limit, len(dataset)):
            break
        add(index)
    return selected


def coreference_loss(log_probs, clusters):
    """Marginal NLL: mean per mention, then mean per document (upstream reduction)."""
    losses = []
    for probs, labels in zip(log_probs, clusters):
        if probs.numel() == 0:
            continue
        mask = fill_expand(labels).to(device=probs.device, dtype=torch.bool)
        if probs.shape != mask.shape:
            raise ValueError("Coreference gold/model shape mismatch")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Every mention needs at least one valid gold antecedent")
        losses.append(-torch.logsumexp(probs.float().masked_fill(~mask, -torch.inf), dim=1).mean())
    if losses:
        return torch.stack(losses).mean()
    # Allow a document with TIMEX but no event mentions, with a differentiable zero.
    return sum((p.sum() for p in log_probs), torch.tensor(0.0, device=log_probs[0].device)) * 0


def relation_loss(logits, labels, positive_class_weight=1.0, none_id=0):
    if logits.shape[:-1] != labels.shape:
        raise ValueError(f"Relation gold/model shape mismatch: {logits.shape}, {labels.shape}")
    valid = labels != -100
    if bool(valid.any()):
        weights = None
        if positive_class_weight != 1.0:
            weights = torch.full((logits.shape[-1],), positive_class_weight,
                                 device=logits.device, dtype=torch.float32)
            weights[none_id] = 1.0
        return F.cross_entropy(logits[valid].float(), labels[valid], weight=weights)
    return logits.sum() * 0.0


def joint_loss(output, batch, positive_class_weight=1.0):
    losses = {"coreference": coreference_loss(output.coref_log_probs, batch["coref_labels"])}
    for task in TASK_MAPS:
        losses[task] = relation_loss(getattr(output, task + "_logits"), batch[task + "_labels"],
                                     positive_class_weight, TASK_MAPS[task]["NONE"])
    for task, value in losses.items():
        if not bool(torch.isfinite(value)):
            raise FloatingPointError(f"Non-finite {task} loss")
    total = sum(LOSS_WEIGHTS[task] * loss for task, loss in losses.items())
    return total, losses


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def autocast_context(args, device):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.bf16 and device.type == "cuda" else contextlib.nullcontext()


def preserve_padding_collator(data):
    """Keep original chunk shapes when replaying reference predictions."""
    batch = collator(data)
    batch["input_ids"] = torch.cat([item["input_ids"] for item in data])
    batch["attention_mask"] = torch.cat([item["attention_mask"] for item in data])
    return batch


def make_loader(dataset, order, batch_size, num_workers, seed, start_batch=0, pin_memory=False,
                preserve_padding=False):
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    # Iterator/worker startup must not consume the model's global dropout RNG.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_sampler=batches[start_batch:], num_workers=num_workers,
                      collate_fn=preserve_padding_collator if preserve_padding else collator,
                      pin_memory=pin_memory, generator=generator)


def load_official_evaluator():
    path = Path(__file__).resolve().parent / "evaluate.py"
    spec = importlib.util.spec_from_file_location("maven_ere_official_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_gold(path):
    with open(path, encoding="utf-8") as stream:
        return {record["id"]: record for record in map(json.loads, filter(str.strip, stream))}


def document_mentions(record):
    events = ([m for event in record["events"] for m in event["mention"]]
              if "events" in record else record["event_mentions"])
    combined = sorted(events + record["TIMEX"], key=lambda m: (m["sent_id"], m["offset"][0]))
    events_only = [m for m in combined if not m["id"].startswith("TIME")]
    return combined, events_only


def counts_prf(counts):
    tp, fp, fn = counts
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision * 100, "recall": recall * 100, "f1": f1 * 100}


@torch.inference_mode()
def evaluate_dev(model, dataset, gold, args, device, score=True):
    model.eval()
    loader = make_loader(dataset, list(range(len(dataset))), args.eval_batch_size or args.batch_size,
                         args.num_workers, args.seed + 100000, pin_memory=device.type == "cuda",
                         preserve_padding=getattr(args, "reference_padding", False))
    predictions = {}
    augmented_counts = {task: [0, 0, 0] for task in TASK_MAPS}
    for original_batch in loader:
        batch = move_batch(original_batch, device)
        with autocast_context(args, device):
            output = model(**batch)
            if score:
                joint_loss(output, batch, getattr(args, "positive_class_weight", 1.0))
        logits = {task: getattr(output, task + "_logits").argmax(-1).cpu() for task in TASK_MAPS}
        for task, pred in (logits.items() if score else []):
            labels = original_batch[task + "_labels"]
            valid = labels != -100
            positive_gold = valid & (labels != TASK_MAPS[task]["NONE"])
            positive_pred = valid & (pred != TASK_MAPS[task]["NONE"])
            correct = valid & (pred == labels) & positive_gold
            augmented_counts[task][0] += int(correct.sum())
            augmented_counts[task][1] += int((positive_pred & ~correct).sum())
            augmented_counts[task][2] += int((positive_gold & ~correct).sum())
        for index, doc_id in enumerate(original_batch["doc_id"]):
            combined, events = document_mentions(gold[doc_id])
            clusters, _ = get_predicted_clusters(output.coref_log_probs[index])
            if getattr(args, "reference_padding", False):
                # Stable reference serialization; cluster memberships are unchanged.
                clusters = sorted(clusters, key=lambda cluster: cluster[0])
            record = {"id": doc_id, "coreference": [[events[i]["id"] for i in c] for c in clusters],
                      "temporal_relations": {k: [] for k in TEMPREL2ID if k != "NONE"},
                      "causal_relations": {"CAUSE": [], "PRECONDITION": []}, "subevent_relations": []}
            pair_index = 0
            for left_index, left in enumerate(combined):
                for right_index, right in enumerate(combined):
                    if left_index == right_index:
                        continue
                    for task in TASK_MAPS:
                        if task != "temporal" and (left["id"].startswith("TIME") or right["id"].startswith("TIME")):
                            continue
                        label = int(logits[task][index, pair_index])
                        if label == TASK_MAPS[task]["NONE"]:
                            continue
                        pair = [left["id"], right["id"]]
                        if task == "subevent":
                            record["subevent_relations"].append(pair)  # [parent, child]
                        else:
                            record[task + "_relations"][REVERSE_MAPS[task][label]].append(pair)
                    pair_index += 1
            predictions[doc_id] = record
    if set(predictions) != set(gold):
        raise ValueError("Dev predictions do not cover exactly the gold documents")
    if not score:
        return None, predictions
    official = score_official_predictions(gold, predictions)
    metrics = {"split": "valid", "official_protocol": official,
               "training_augmented_labels_positive_micro": {task: counts_prf(value) for task, value in augmented_counts.items()},
               "temporal_protocol_note": "Training labels symmetrize SIMULTANEOUS/BEGINS-ON; official evaluator uses raw annotated directions."}
    return metrics, predictions


def score_official_predictions(gold, predictions):
    evaluator = load_official_evaluator()
    coref = {name: float(value) for name, value in evaluator.evaluate_coreference(gold, predictions).items()}
    official = {"coreference": coref}
    for task in TASK_MAPS:
        official[task] = {name: float(value) for name, value in evaluator.evaluate(gold, predictions, task).items()}
    coref_mean = sum(coref[name + "_f1"] for name in ("b_cubed", "ceaf", "muc", "blanc")) / 4
    official["coreference_mean_f1"] = coref_mean
    official["overall_f1"] = (coref_mean + sum(official[t][t + "_f1"] for t in TASK_MAPS)) / 4
    return official


def save_predictions(path, predictions):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as stream:
        for key in sorted(predictions):
            stream.write(json.dumps(predictions[key], ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def export_best(model, tokenizer, output_dir, metrics, predictions, name="best_joint",
                checkpoint_identity=None, shared_directory=None, defer_cleanup=False):
    """每个链接都指向一个完整快照；同次验证可共享目录，绝不混合各任务权重。"""
    output_dir = Path(output_dir)
    exports = output_dir / "exports"
    exports.mkdir(exist_ok=True)
    allowed = {"best_joint", *("best_" + task for task in SELECTION_TASKS)}
    if name not in allowed:
        raise ValueError(f"Unknown best-model selector: {name}")
    final = Path(shared_directory) if shared_directory else exports / ("best-" + uuid.uuid4().hex)
    identity = copy.deepcopy(checkpoint_identity or {"checkpoint_id": final.name})
    if shared_directory:
        if final.resolve().parent != exports.resolve():
            raise ValueError("Shared checkpoint must be in this run's exports directory")
        if load_json_file(final / "checkpoint_identity.json") != identity:
            raise ValueError("Shared export does not identify the same complete checkpoint")
    else:
        model.save_pretrained(final, safe_serialization=True)
        tokenizer.save_pretrained(final)
        atomic_json(final / "dev_metrics.json", metrics)
        save_predictions(final / "dev_predictions.jsonl", predictions)
        atomic_json(final / "checkpoint_identity.json", identity)
    link = output_dir / name
    if link.exists() and not link.is_symlink():
        raise ValueError(f"Refusing to overwrite a non-symlink export directory: {link}")
    temporary_link = output_dir / ("." + name + "-" + uuid.uuid4().hex)
    temporary_link.symlink_to(final.relative_to(output_dir), target_is_directory=True)
    os.replace(temporary_link, link)
    if not defer_cleanup:
        prune_exports(output_dir)
    return str(final.relative_to(output_dir))


def load_json_file(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prune_exports(output_dir):
    """仅清理由本驱动创建且已无 best 链接引用的完整快照。"""
    output_dir = Path(output_dir)
    exports = output_dir / "exports"
    if not exports.exists():
        return
    keep = {(output_dir / name).resolve() for name in
            ["best_joint", *("best_" + task for task in SELECTION_TASKS)]
            if (output_dir / name).is_symlink()}
    for directory in exports.iterdir():
        if directory.name.startswith("best-") and directory.is_dir() and not directory.is_symlink():
            if directory.resolve() not in keep:
                shutil.rmtree(directory)


def selection_scores(metrics):
    official = metrics["official_protocol"]
    return {"coreference": float(official["coreference_mean_f1"]),
            **{task: float(official[task][task + "_f1"]) for task in TASK_MAPS}}


def python_scalars(value):
    """上游 sklearn/NumPy 指标可含 np.bool/float32；持久化前去除 NumPy scalar。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: python_scalars(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [python_scalars(item) for item in value]
    return value


def restore_selection_artifacts(progress, output_dir, coreference_selection=None):
    """磁盘历史和链接回到 latest.pt 的事务边界，丢弃未提交的验证结果。"""
    output_dir = Path(output_dir)
    legacy = "selection_format_version" not in progress
    progress["selection_format_version"] = 1
    progress.setdefault("run_id", uuid.uuid4().hex)
    progress.setdefault("task_bests", {})
    progress.setdefault("validation_history", [])
    selector = coreference_selection or progress.get("coreference_selection", "mean")
    if progress.get("coreference_selection", selector) != selector:
        raise ValueError("Cannot restore with a different coreference selection rule")
    progress["coreference_selection"] = selector
    if "coreference_metric_bests" not in progress:
        # 旧 mean 模式只恢复实际保存的观察历史，不臆造缺失的过去验证结果。
        maxima = {name: 0.0 for name in COREFERENCE_METRICS}
        for history in progress["validation_history"]:
            observed = history["metrics"]["official_protocol"].get("coreference", {})
            for name in COREFERENCE_METRICS:
                maxima[name] = max(maxima[name], observed.get(name + "_f1", 0.0))
        progress["coreference_metric_bests"] = maxima
    if legacy and progress["best_score"] >= 0 and "joint_best" not in progress and (output_dir / "best_joint").is_symlink():
        directory = (output_dir / "best_joint").resolve()
        progress["joint_best"] = {
            "checkpoint_id": "legacy-" + directory.name, "epoch": None, "update_step": None,
            "score": progress["best_score"], "export_directory": str(directory.relative_to(output_dir)),
            "export_name": "best_joint", "exported": True,
        }
    entries = [progress.get("joint_best"), *progress["task_bests"].values()]
    expected_links = {entry["export_name"] for entry in entries if entry and entry.get("export_directory")}
    for name in ["best_joint", *("best_" + task for task in SELECTION_TASKS)]:
        link = output_dir / name
        if name not in expected_links and link.is_symlink():
            link.unlink()
    for entry in entries:
        if not entry or not entry.get("export_directory"):
            continue
        final = (output_dir / entry["export_directory"]).resolve(strict=True)
        if final.parent != (output_dir / "exports").resolve():
            raise ValueError("Checkpoint selection points outside this run's exports")
        identity_path = final / "checkpoint_identity.json"
        if identity_path.exists() and load_json_file(identity_path)["checkpoint_id"] != entry["checkpoint_id"]:
            raise ValueError("Restored selection and materialized checkpoint disagree")
        link = output_dir / entry["export_name"]
        if link.exists() and not link.is_symlink():
            raise ValueError(f"Refusing to replace a non-symlink export: {link}")
        temporary = output_dir / ("." + entry["export_name"] + "-" + uuid.uuid4().hex)
        temporary.symlink_to(final.relative_to(output_dir), target_is_directory=True)
        os.replace(temporary, link)
    path = output_dir / "history.jsonl"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for entry in progress["validation_history"]:
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    prune_exports(output_dir)


def record_validation(model, tokenizer, output_dir, progress, metrics, predictions, args, epoch):
    """一次验证对应一个模型身份；每个任务选择的是完整模型，不拼接编码器或头。"""
    metrics = python_scalars(metrics)
    identity = {"checkpoint_id": f"{progress['run_id']}-e{epoch + 1:03d}-u{progress['update_step']:08d}",
                "epoch": epoch + 1, "update_step": progress["update_step"]}
    overall = float(metrics["official_protocol"]["overall_f1"])
    overall_improved = bool(overall > progress["best_score"])
    scores = selection_scores(metrics)
    improved = {task: bool(scores[task] > progress["task_bests"].get(task, {}).get("score", -1.0))
                for task in SELECTION_TASKS}
    selector = getattr(args, "coreference_selection", "mean")
    if progress.get("coreference_selection", selector) != selector:
        raise ValueError("Coreference selection rule changed within one training run")
    progress["coreference_selection"] = selector
    coreference = metrics["official_protocol"]["coreference"]
    observed = {name: float(coreference[name + "_f1"]) for name in COREFERENCE_METRICS}
    maxima = progress.setdefault("coreference_metric_bests", {name: 0.0 for name in COREFERENCE_METRICS})
    if selector == "upstream-any":
        improved["coreference"] = bool(any(observed[name] > maxima[name] for name in COREFERENCE_METRICS))
    for name in COREFERENCE_METRICS:
        maxima[name] = max(maxima[name], observed[name])
    shared_directory = None
    if overall_improved:
        directory = export_best(model, tokenizer, output_dir, metrics, predictions,
                                checkpoint_identity=identity, defer_cleanup=True)
        shared_directory = Path(output_dir) / directory
        progress["best_score"] = overall
        progress["joint_best"] = {**identity, "score": overall, "export_directory": directory,
                                  "export_name": "best_joint", "exported": True,
                                  "selection_rule": "overall_f1"}
    for task in SELECTION_TASKS:
        if not improved[task]:
            continue
        directory = None
        if args.export_task_bests:
            directory = export_best(model, tokenizer, output_dir, metrics, predictions,
                                    name="best_" + task, checkpoint_identity=identity,
                                    shared_directory=shared_directory, defer_cleanup=True)
            shared_directory = Path(output_dir) / directory
        progress["task_bests"][task] = {**identity, "score": scores[task], "export_directory": directory,
                                        "export_name": "best_" + task, "exported": bool(directory),
                                        "selection_rule": selector if task == "coreference" else task + "_f1"}
        if task == "coreference":
            # score 是该快照当时的 mean；每项历次 max 属于 progress，不能充当同一模型成绩。
            progress["task_bests"][task]["coreference_metric_values"] = copy.deepcopy(observed)
    monitored_improvement = (overall_improved if args.early_stopping_monitor == "overall"
                             else any(improved.values()))
    progress["bad_evaluations"] = 0 if monitored_improvement else progress["bad_evaluations"] + 1
    history = {**identity, "metrics": copy.deepcopy(metrics), "overall_improved": overall_improved,
               "task_improvements": improved, "early_stopping_monitor": args.early_stopping_monitor,
               "coreference_selection": selector, "coreference_metric_values": observed,
               "coreference_metric_bests": copy.deepcopy(maxima),
               "bad_evaluations": progress["bad_evaluations"],
               "joint_best": copy.deepcopy(progress.get("joint_best")),
               "task_bests": copy.deepcopy(progress["task_bests"])}
    progress["validation_history"].append(history)
    with (Path(output_dir) / "history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(history, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return overall_improved


def should_evaluate(args, update_step, epoch_end, schedule_end):
    """末次强制评估；关闭 epoch-end 后只在配置的更新间隔评估。"""
    frequency = args.eval_every_updates
    return bool(schedule_end or (epoch_end and getattr(args, "eval_epoch_end", True))
                or (frequency and update_step % frequency == 0))


def combine_task_best_predictions(output_dir, progress, gold):
    """仅组合各快照的预测，产物显式标为多 checkpoint；不创建混合权重模型。"""
    output_dir = Path(output_dir)
    combined, augmented, sources = {}, {}, {}
    for task in SELECTION_TASKS:
        entry = progress["task_bests"].get(task)
        if not entry or not entry.get("export_directory"):
            raise ValueError("All four task-best checkpoints must be exported before combination")
        directory = (output_dir / entry["export_directory"]).resolve(strict=True)
        if directory.parent != (output_dir / "exports").resolve():
            raise ValueError("Task-best checkpoint is outside this run")
        if load_json_file(directory / "checkpoint_identity.json")["checkpoint_id"] != entry["checkpoint_id"]:
            raise ValueError("Task-best identity mismatch")
        predictions = read_gold(directory / "dev_predictions.jsonl")
        if set(predictions) != set(gold):
            raise ValueError("Task-best predictions must cover exactly the current validation split")
        field = "coreference" if task == "coreference" else task + "_relations"
        for doc_id, prediction in predictions.items():
            combined.setdefault(doc_id, {"id": doc_id})[field] = copy.deepcopy(prediction[field])
        metrics = load_json_file(directory / "dev_metrics.json")
        if task != "coreference":
            augmented[task] = metrics["training_augmented_labels_positive_micro"][task]
        sources[task] = copy.deepcopy(entry)
    metrics = {"split": "valid", "artifact_type": "multiple_checkpoint_task_combination",
               "is_single_model": False, "selected_checkpoints": sources,
               "official_protocol": score_official_predictions(gold, combined),
               "training_augmented_labels_positive_micro": augmented}
    prediction_path = output_dir / "task_best_combined_predictions.jsonl"
    metric_path = output_dir / "task_best_combined_metrics.json"
    save_predictions(prediction_path, combined)
    atomic_json(metric_path, metrics)
    return {"is_single_model": False, "predictions": str(prediction_path),
            "metrics": str(metric_path), "official_protocol": metrics["official_protocol"],
            "selected_checkpoints": sources}


def training_signature(args, cache_keys, train_ids):
    keys = ("seed", "max_epochs", "batch_size", "grad_accum", "bf16", "max_length", "max_steps",
            "encoder_lr", "head_lr", "warmup_updates", "clip_grad_norm", "gradient_checkpointing", "smoke")
    return {**{key: getattr(args, key) for key in keys},
            **{key: getattr(args, key, default) for key, default in NEW_SIGNATURE_DEFAULTS.items()}, "cache_keys": cache_keys,
            "train_document_ids": train_ids, "loss_weights": LOSS_WEIGHTS,
            "coreference_reduction": "mean_mentions_then_mean_documents",
            "relation_reduction": "mean_nonignored_ordered_pairs_per_microbatch"}


def resume_signature_matches(saved, current):
    # 旧 checkpoint 只有在新参数均取兼容默认值时，才能按原有语义恢复。
    normalized = {**NEW_SIGNATURE_DEFAULTS, **saved}
    return normalized == current


def build_model(args, state=None):
    if state is not None:
        model = MavenEreModel(MavenEreConfig.from_dict(state["model_config"]))
        model.load_state_dict(state["model"])
    else:
        encoder_config = AutoConfig.from_pretrained(args.base_model, local_files_only=True).to_dict()
        encoder_config["_attn_implementation"] = "eager"
        config = MavenEreConfig(encoder_config=encoder_config,
                               scorer_hidden_size=getattr(args, "scorer_hidden_size", 200),
                               scorer_num_layers=getattr(args, "scorer_num_layers", 3),
                               span_pooling="mean", dropout=0.2, loss_weights=LOSS_WEIGHTS)
        config.positive_class_weight = getattr(args, "positive_class_weight", 1.0)
        config.chunk_length = args.max_length
        model = MavenEreModel(config)
        pretrained = AutoModel.from_pretrained(args.base_model, local_files_only=True, attn_implementation="eager")
        model.encoder.load_state_dict(pretrained.state_dict())
        del pretrained
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def build_optimizers(model, args, total_updates):
    encoder_optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=args.encoder_lr,
                                         eps=1e-6, weight_decay=0.0)
    encoder_ids = {id(parameter) for parameter in model.encoder.parameters()}
    head_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    head_optimizer = torch.optim.Adam(head_parameters, lr=args.head_lr, eps=1e-8, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(encoder_optimizer, args.warmup_updates, total_updates)
    return encoder_optimizer, head_optimizer, scheduler


def checkpoint_state(model, encoder_optimizer, head_optimizer, scheduler, progress, signature):
    return {"format_version": 1, "model": model.state_dict(), "model_config": model.config.to_dict(),
            "encoder_optimizer": encoder_optimizer.state_dict(), "head_optimizer": head_optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "progress": python_scalars(copy.deepcopy(progress)), "signature": signature,
            "rng": rng_state()}


def restore_training_state(state, model, encoder_optimizer, head_optimizer, scheduler):
    model.load_state_dict(state["model"])
    encoder_optimizer.load_state_dict(state["encoder_optimizer"])
    head_optimizer.load_state_dict(state["head_optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"])
    return python_scalars(copy.deepcopy(state["progress"]))


@torch.inference_mode()
def smoke_roundtrip(model, dataset, tokenizer, args, device, output_dir):
    """Check the exported model against the same best checkpoint on real inputs."""
    exported = MavenEreModel.from_pretrained(Path(output_dir) / "best_joint", local_files_only=True).to(device)
    exported.eval()
    probe = move_batch(collator([dataset[0]]), device)
    with autocast_context(args, device):
        before = model(**probe)
        after = exported(**probe)
    for task in TASK_MAPS:
        torch.testing.assert_close(getattr(before, task + "_logits"), getattr(after, task + "_logits"), rtol=0, atol=0)
    for first, second in zip(before.coref_log_probs, after.coref_log_probs):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    reloaded_tokenizer = RobertaTokenizer.from_pretrained(Path(output_dir) / "best_joint", local_files_only=True)
    if tokenizer.get_vocab() != reloaded_tokenizer.get_vocab():
        raise AssertionError("Tokenizer vocabulary changed across HF roundtrip")
    del exported


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", "--base-model", help="Local RoBERTa-base snapshot (no implicit network downloads)")
    parser.add_argument("--data_dir", "--data-dir", required=True)
    parser.add_argument("--output_dir", "--output-dir")
    parser.add_argument("--cache_dir", "--cache-dir")
    parser.add_argument("--predict_only", "--predict-only", action="store_true")
    parser.add_argument("--reference-padding", action="store_true",
                        help="Prediction only: preserve reference padding and stable coreference cluster order")
    parser.add_argument("--checkpoint", help="Local save_pretrained export for prediction")
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--prediction_file", "--prediction-file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_epochs", "--epochs", type=int, default=100)
    parser.add_argument("--batch_size", "--bs", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--encoder_lr", type=float, default=2e-5)
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--warmup_updates", type=int, default=200)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--scorer-hidden-size", "--scorer_hidden_size", type=int, default=200)
    parser.add_argument("--scorer-num-layers", "--scorer_num_layers", type=int, choices=(2, 3), default=3)
    parser.add_argument("--positive-class-weight", "--positive_class_weight", type=float, default=1.0,
                        help="Weighted CE: NONE weight 1; each positive relation class uses this weight")
    parser.add_argument("--export-task-bests", "--export_task_bests", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip-test-predictions", "--skip_test_predictions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--eval_every_updates", type=int, default=0,
                        help="Periodic optimizer-update interval; 0 disables periodic evaluation")
    parser.add_argument("--eval-epoch-end", "--eval_epoch_end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--coreference-selection", "--coreference_selection", choices=("mean", "upstream-any"), default="mean")
    parser.add_argument("--save_every_updates", type=int, default=50)
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Number of dev evaluations; 0 disables")
    parser.add_argument("--early-stopping-monitor", "--early_stopping_monitor", choices=("overall", "any-task"), default="overall")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resume", nargs="?", const="latest", help="latest, or a trusted training checkpoint path")
    parser.add_argument("--smoke", action="store_true", help="64 real train docs, positive coverage, max 2 updates, checkpoint and HF roundtrips")
    parser.add_argument("--smoke_documents", type=int, default=64)
    args = parser.parse_args(argv)
    if args.predict_only:
        if not args.checkpoint or not args.prediction_file:
            parser.error("--predict-only requires --checkpoint and --prediction-file")
        if args.resume or args.smoke:
            parser.error("--predict-only cannot be combined with --resume or --smoke")
        args.output_dir = args.output_dir or str(Path(args.prediction_file).expanduser().resolve().parent)
    else:
        if args.reference_padding:
            parser.error("--reference-padding requires --predict-only")
        if not args.base_model or not args.output_dir:
            parser.error("Training requires --base_model and --output_dir")
    for name in ("max_epochs", "batch_size", "eval_batch_size", "grad_accum", "max_length", "save_every_updates", "smoke_documents"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max_steps must be positive")
    if args.scorer_hidden_size <= 0:
        parser.error("--scorer-hidden-size must be positive")
    if not math.isfinite(args.positive_class_weight) or args.positive_class_weight <= 0:
        parser.error("--positive-class-weight must be finite and positive")
    if args.num_workers < 0 or args.eval_every_updates < 0 or args.early_stopping_patience < 0:
        parser.error("Worker/evaluation/patience settings cannot be negative")
    if args.smoke:
        args.max_steps = min(args.max_steps or 2, 2)
    return args


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    LOGGER.warning("Signal %s: will save and stop at the next optimizer boundary", signum)


def predict_only(args, device):
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    model = MavenEreModel.from_pretrained(checkpoint, local_files_only=True).to(device)
    tokenizer = RobertaTokenizer.from_pretrained(checkpoint, local_files_only=True, add_prefix_space=True)
    max_length = getattr(model.config, "chunk_length", args.max_length)
    source = Path(args.data_dir) / (args.split + ".jsonl")
    dataset, _ = cached_dataset(tokenizer, args.data_dir, args.split, max_length,
                                args.cache_dir or Path(args.output_dir) / "cache", sha256_file(source))
    records = read_gold(source)
    _, predictions = evaluate_dev(model, dataset, records, args, device, score=False)
    destination = Path(args.prediction_file).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_predictions(destination, predictions)
    LOGGER.info("Predicted %s %s documents to %s; no scores calculated", len(predictions), args.split, destination)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(output_dir / "train.log")])
    if not args.predict_only and (output_dir / "latest.pt").exists() and not args.resume:
        raise ValueError("output_dir already has latest.pt; use --resume or a fresh output_dir")
    if not args.predict_only and not args.resume and any(
        (output_dir / name).exists() or (output_dir / name).is_symlink()
        for name in ["best_joint", *("best_" + task for task in SELECTION_TASKS)]
    ):
        raise ValueError("output_dir already has selected checkpoints; use a fresh output_dir")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    if device.type == "cuda" and args.bf16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested but unsupported on this GPU")
    if args.predict_only:
        predict_only(args, device)
        return
    set_seed(args.seed)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    atomic_json(output_dir / "progress.json", {"status": "validating_data", "seed": args.seed})
    split_manifest = validate_splits(args.data_dir)
    atomic_json(output_dir / "split_manifest.json", split_manifest)
    tokenizer = RobertaTokenizer.from_pretrained(args.base_model, local_files_only=True, add_prefix_space=True)
    cache_dir = Path(args.cache_dir or output_dir / "cache").expanduser().resolve()
    datasets, cache_keys = {}, {}
    for split in ("train", "valid"):
        atomic_json(output_dir / "progress.json", {"status": "loading_tokenization", "split": split, "seed": args.seed})
        datasets[split], cache_keys[split] = cached_dataset(tokenizer, args.data_dir, split, args.max_length,
                                                           cache_dir, split_manifest[split]["sha256"])
    if args.smoke:
        smoke_indices = choose_smoke_indices(datasets["train"], args.smoke_documents)
        datasets["train"] = Subset(datasets["train"], smoke_indices)
        # Dev uses actual labels for a small disjoint set; never call it a test score.
        datasets["valid"] = Subset(datasets["valid"], list(range(min(16, len(datasets["valid"])))))
    gold = read_gold(Path(args.data_dir) / "valid.jsonl")
    valid_ids = {datasets["valid"][i]["doc_id"] for i in range(len(datasets["valid"]))}
    gold = {key: value for key, value in gold.items() if key in valid_ids}
    train_ids = [datasets["train"][i]["doc_id"] for i in range(len(datasets["train"]))]
    signature = training_signature(args, cache_keys, train_ids)
    batch_count = math.ceil(len(datasets["train"]) / args.batch_size)
    updates_per_epoch = math.ceil(batch_count / args.grad_accum)
    total_updates = updates_per_epoch * args.max_epochs
    if args.max_steps:
        total_updates = min(total_updates, args.max_steps)
    state = None
    if args.resume:
        checkpoint_path = output_dir / "latest.pt" if args.resume == "latest" else Path(args.resume)
        state = load_checkpoint(checkpoint_path)
        if not resume_signature_matches(state["signature"], signature):
            raise ValueError("Resume training signature differs (data, schedule, precision, batch, seed, etc.)")
    atomic_json(output_dir / "progress.json", {"status": "loading_model", "resume": bool(state), "seed": args.seed})
    model = build_model(args, state).to(device)
    encoder_optimizer, head_optimizer, scheduler = build_optimizers(model, args, total_updates)
    progress = {"epoch": 0, "next_batch": 0, "order": [], "update_step": 0, "micro_step": 0,
                "best_score": -1.0, "bad_evaluations": 0, "last_eval_update": -1,
                "elapsed_seconds": 0.0, "last_metrics": None}
    if state is not None:
        progress = restore_training_state(state, model, encoder_optimizer, head_optimizer, scheduler)
        del state
    restore_selection_artifacts(progress, output_dir, args.coreference_selection)
    atomic_json(output_dir / "training_config.json", {**vars(args), "signature": signature,
                                                       "total_updates": total_updates, "updates_per_epoch": updates_per_epoch})
    LOGGER.info("Training %s docs; microbatch=%s accumulation=%s nominal_effective_docs=%s updates=%s",
                len(train_ids), args.batch_size, args.grad_accum, args.batch_size * args.grad_accum, total_updates)
    LOGGER.info("Coreference reduction: mean_mentions_then_mean_documents; relation reduction: mean_pairs_per_microbatch")
    initial_elapsed = progress["elapsed_seconds"]
    start_time = time.monotonic()
    last_losses = {}
    stopped_early = bool(args.early_stopping_patience and progress["bad_evaluations"] >= args.early_stopping_patience)
    smoke_resumed = False
    smoke_positive_counts = {task: 0 for task in LOSS_WEIGHTS}
    for optimizer in (encoder_optimizer, head_optimizer):
        optimizer.zero_grad(set_to_none=True)
    while not stopped_early and progress["epoch"] < args.max_epochs and progress["update_step"] < total_updates:
        epoch = progress["epoch"]
        if not progress["order"]:
            generator = torch.Generator().manual_seed(args.seed + epoch)
            progress["order"] = torch.randperm(len(datasets["train"]), generator=generator).tolist()
        loader = make_loader(datasets["train"], progress["order"], args.batch_size, args.num_workers,
                             args.seed + epoch + 500000, progress["next_batch"], pin_memory=device.type == "cuda")
        first_batch = progress["next_batch"]
        for offset, original_batch in enumerate(loader):
            batch_index = first_batch + offset
            model.train()
            batch = move_batch(original_batch, device)
            group_start = (batch_index // args.grad_accum) * args.grad_accum
            group_size = min(args.grad_accum, batch_count - group_start)
            with autocast_context(args, device):
                output = model(**batch)
                loss, task_losses = joint_loss(output, batch, args.positive_class_weight)
            if args.smoke:
                for task, label_map in TASK_MAPS.items():
                    labels = batch[task + "_labels"]
                    positive = (labels >= 0) & (labels != label_map["NONE"])
                    if bool(positive.any()):
                        positive_loss = F.cross_entropy(getattr(output, task + "_logits").detach()[positive].float(), labels[positive])
                        if not bool(torch.isfinite(positive_loss)):
                            raise FloatingPointError(f"Non-finite smoke positive-only {task} loss")
                        smoke_positive_counts[task] += int(positive.sum())
                smoke_positive_counts["coreference"] += sum(len(cluster) - 1 for clusters in batch["coref_labels"] for cluster in clusters)
            # 没有事件且不足两个 TIMEX 的文档没有任何可监督位置。
            # 这样的微批只贡献零损失，不对无梯度的空张量调用 backward。
            if loss.requires_grad:
                (loss / group_size).backward()
            last_losses = {key: float(value.detach()) for key, value in task_losses.items()}
            progress["micro_step"] += 1
            boundary = (batch_index + 1) % args.grad_accum == 0 or batch_index + 1 == batch_count
            if not boundary:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm, error_if_nonfinite=True)
            encoder_optimizer.step()
            head_optimizer.step()
            scheduler.step()
            for optimizer in (encoder_optimizer, head_optimizer):
                optimizer.zero_grad(set_to_none=True)
            progress["update_step"] += 1
            progress["next_batch"] = batch_index + 1
            epoch_end = progress["next_batch"] == batch_count
            schedule_end = progress["update_step"] >= total_updates
            evaluate_now = should_evaluate(args, progress["update_step"], epoch_end, schedule_end)
            if evaluate_now and progress["last_eval_update"] != progress["update_step"]:
                metrics, predictions = evaluate_dev(model, datasets["valid"], gold, args, device)
                progress["last_metrics"] = metrics
                progress["last_eval_update"] = progress["update_step"]
                atomic_json(output_dir / "dev_metrics.json", metrics)
                save_predictions(output_dir / "dev_predictions.jsonl", predictions)
                score = metrics["official_protocol"]["overall_f1"]
                overall_improved = record_validation(
                    model, tokenizer, output_dir, progress, metrics, predictions, args, epoch)
                if overall_improved:
                    if args.smoke:
                        smoke_roundtrip(model, datasets["valid"], tokenizer, args, device, output_dir)
                        LOGGER.info("SMOKE: HF model/tokenizer roundtrip passed")
                LOGGER.info("Dev epoch=%s update=%s official_overall_f1=%.4f best=%.4f", epoch + 1, progress["update_step"], score, progress["best_score"])
                if args.early_stopping_patience and progress["bad_evaluations"] >= args.early_stopping_patience:
                    stopped_early = True
            if epoch_end:
                progress["epoch"] += 1
                progress["next_batch"] = 0
                progress["order"] = []
            progress["elapsed_seconds"] = initial_elapsed + time.monotonic() - start_time
            saved_now = epoch_end or evaluate_now or schedule_end or STOP_REQUESTED or stopped_early or progress["update_step"] % args.save_every_updates == 0 or (args.smoke and not smoke_resumed)
            if saved_now:
                atomic_torch_save(output_dir / "latest.pt", checkpoint_state(model, encoder_optimizer, head_optimizer, scheduler, progress, signature))
                # latest.pt 提交成功后才删除旧快照，崩溃恢复仍可还原旧 best 链接。
                prune_exports(output_dir)
                if args.smoke and not smoke_resumed:
                    restored = load_checkpoint(output_dir / "latest.pt")
                    before_rng = torch.get_rng_state().clone()
                    restored_progress = restore_training_state(restored, model, encoder_optimizer, head_optimizer, scheduler)
                    if restored_progress != progress or not torch.equal(before_rng, torch.get_rng_state()):
                        raise AssertionError("Training checkpoint resume failed")
                    if scheduler.last_epoch != progress["update_step"]:
                        raise AssertionError("Scheduler update counter changed across resume")
                    del restored
                    smoke_resumed = True
                    LOGGER.info("SMOKE: complete model/optimizer/scheduler/RNG checkpoint restore passed")
            snapshot = {key: value for key, value in progress.items() if key not in {"order", "validation_history"}}
            snapshot.update({"status": "stopping" if STOP_REQUESTED else "running", "losses": last_losses,
                             "gradient_norm": float(grad_norm), "encoder_lr": encoder_optimizer.param_groups[0]["lr"],
                             "head_lr": head_optimizer.param_groups[0]["lr"], "checkpoint_saved_this_update": bool(saved_now),
                             "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0})
            atomic_json(output_dir / "progress.json", snapshot)
            LOGGER.info("epoch=%s update=%s/%s micro=%s loss=%s grad=%.4f elapsed=%.1fs", epoch + 1,
                        progress["update_step"], total_updates, progress["micro_step"], last_losses,
                        float(grad_norm), progress["elapsed_seconds"])
            if schedule_end or STOP_REQUESTED or stopped_early:
                break
        if STOP_REQUESTED or stopped_early:
            break
    final_status = "interrupted" if STOP_REQUESTED else "early_stopped" if stopped_early else "completed"
    if args.smoke and not STOP_REQUESTED and (not smoke_resumed or not (output_dir / "best_joint").exists()):
        raise AssertionError("Smoke run did not complete both checkpoint and HF export checks")
    if args.smoke and not STOP_REQUESTED and not all(smoke_positive_counts.values()):
        raise AssertionError(f"Smoke updates did not exercise all four positive-label losses: {smoke_positive_counts}")
    test_prediction_file = None
    combination = None
    if (output_dir / "best_joint").exists():
        for name in ("dev_metrics.json", "dev_predictions.jsonl"):
            shutil.copyfile(output_dir / "best_joint" / name, output_dir / ("best_" + name))
    if args.export_task_bests and all(progress["task_bests"].get(task, {}).get("export_directory") for task in SELECTION_TASKS):
        combination = combine_task_best_predictions(output_dir, progress, gold)
    if final_status in ("completed", "early_stopped") and not args.smoke and not args.skip_test_predictions and (Path(args.data_dir) / "test.jsonl").exists():
        LOGGER.info("Exporting test candidate predictions using best_joint; no test scores available")
        best = MavenEreModel.from_pretrained(output_dir / "best_joint", local_files_only=True)
        model.load_state_dict(best.state_dict())
        del best
        test_dataset, _ = cached_dataset(tokenizer, args.data_dir, "test", args.max_length, cache_dir,
                                         split_manifest["test"]["sha256"])
        test_records = read_gold(Path(args.data_dir) / "test.jsonl")
        _, test_predictions = evaluate_dev(model, test_dataset, test_records, args, device, score=False)
        test_prediction_file = output_dir / "test_predictions.jsonl"
        save_predictions(test_prediction_file, test_predictions)
    summary = {key: value for key, value in progress.items() if key not in {"order", "validation_history"}}
    summary.update({"status": final_status, "smoke": args.smoke, "best_model": str(output_dir / "best_joint"),
                    "latest_checkpoint": str(output_dir / "latest.pt"), "losses": last_losses,
                    "test_evaluated": False, "test_predictions": str(test_prediction_file) if test_prediction_file else None,
                    "smoke_positive_counts": smoke_positive_counts if args.smoke else None,
                    "validation_history": str(output_dir / "history.jsonl"),
                    "validation_evaluations": len(progress["validation_history"]),
                    "task_best_combination": combination,
                    "best_joint_metrics": str(output_dir / "best_dev_metrics.json") if (output_dir / "best_joint").exists() else None,
                    "task_best_combined_metrics": combination["metrics"] if combination else None,
                    "task_best_combined_predictions": combination["predictions"] if combination else None,
                    "validation_protocol": "upstream evaluate.py on raw annotation directions"})
    atomic_json(output_dir / "progress.json", summary)
    atomic_json(output_dir / "training_summary.json", summary)
    LOGGER.info("Training status=%s updates=%s best_dev_overall=%.4f", final_status, progress["update_step"], progress["best_score"])


if __name__ == "__main__":
    main()
