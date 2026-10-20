#!/usr/bin/env python3
"""Run an audited, isolated copy of the authors' joint-training implementation.

This launcher imports only the standard library. --prepare-only extracts and
AST-checks sources without importing torch or executing model code. Full runs
must be launched in the authorized remote training environment.
"""

import argparse
import ast
from datetime import datetime, timezone
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import signal
import subprocess
import sys
import traceback


UPSTREAM_COMMIT = "ac81a9711a69f43f55bfbc50b3bb573fd11c64b0"
HF_ADAMW_SOURCE = "https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/optimization.py#L249-L350"
PROTECTED = (
    "/Users/mayiding/Library/Mobile Documents",
    "/Users/mayiding/Library/CloudStorage",
    "/Users/mayiding/Library/Application Support/CloudDocs",
)


LEGACY_ADAMW_SOURCE = r'''# SPDX-License-Identifier: Apache-2.0
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Adapted for this reference runner from Transformers v4.24.0 optimization.py.
# Source: https://github.com/huggingface/transformers/blob/v4.24.0/src/transformers/optimization.py#L249-L350
# Modifications: standalone imports, concise validation, no deprecation logger;
# retain the original update order, epsilon placement and bias correction.
# A copy of Apache-2.0 is provided in the run's licenses/ directory.
import math
import torch


class AdamW(torch.optim.Optimizer):
    """Historical HF AdamW; intentionally not torch.optim.AdamW's epsilon math."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.0, correct_bias=True, no_deprecation_warning=False):
        if lr < 0 or eps < 0 or not all(0 <= beta < 1 for beta in betas):
            raise ValueError("Invalid legacy AdamW hyperparameters")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay, correct_bias=correct_bias))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("Legacy AdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                first, second = state["exp_avg"], state["exp_avg_sq"]
                first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                second.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                denominator = second.sqrt().add_(group["eps"])
                step_size = group["lr"]
                if group["correct_bias"]:
                    step_size *= math.sqrt(1.0 - beta2 ** state["step"]) / (1.0 - beta1 ** state["step"])
                parameter.addcdiv_(first, denominator, value=-step_size)
                if group["weight_decay"] > 0:
                    parameter.add_(parameter, alpha=-group["lr"] * group["weight_decay"])
        return loss
'''


REFERENCE_HOOKS_SOURCE = r'''"""Observability and diagnostic exports; never replaces the upstream loss."""
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import time
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader, Subset

RUN = Path(os.environ["MAVEN_REFERENCE_OUTPUT"])
CANARY = os.environ.get("MAVEN_REFERENCE_CANARY") == "1"
START = time.monotonic()
PROGRESS = {"status": "initializing", "micro_step": 0, "optimizer_updates": 0,
            "canary": CANARY, "paper_reproduction_success": None}
BEST_SELECTION = {}
BEST_VALIDATION = {}
BEST_RAW_VALIDATION = {}
BEST_DEV_COMPOSITE = {}
FALLBACK_TASKS = []
FINITE_WEIGHT_CHECKS = {}


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, path)


def event(kind, **fields):
    timestamp = datetime.now(timezone.utc).isoformat()
    record = {"timestamp_utc": timestamp, "event": kind, **fields}
    with open(RUN / "events.jsonl", "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False, ensure_ascii=False) + "\n")
    PROGRESS.update(fields)
    PROGRESS["last_event"] = kind
    PROGRESS["timestamp_utc"] = timestamp
    PROGRESS["elapsed_seconds"] = time.monotonic() - START
    _json(RUN / "progress.json", PROGRESS)


def base_model_path():
    return os.environ["MAVEN_REFERENCE_BASE"]


def _longest_and_positive_indices(dataset, limit=64):
    selected = []
    def add(index):
        if index not in selected:
            selected.append(index)
    add(max(range(len(dataset)), key=lambda i: dataset[i]["input_ids"].shape[0]))
    add(max(range(len(dataset)), key=lambda i: sum(map(len, dataset[i]["event_spans"]))))
    for name, none in (("temporal", 6), ("causal", 0), ("subevent", 0)):
        index = next((i for i in range(len(dataset))
                      if bool(((dataset[i][name + "_labels"] >= 0) &
                               (dataset[i][name + "_labels"] != none)).any())), None)
        if index is None:
            raise RuntimeError("Canary dataset lacks positive " + name)
        add(index)
    coref = next((i for i in range(len(dataset)) if any(len(c) > 1 for c in dataset[i]["coref_labels"])), None)
    if coref is None:
        raise RuntimeError("Canary dataset lacks non-singleton coreference")
    add(coref)
    for index in range(len(dataset)):
        if len(selected) >= min(limit, len(dataset)):
            break
        add(index)
    return selected


def reference_loader(loader, split):
    original = loader.dataset
    if CANARY:
        indices = _longest_and_positive_indices(original) if split == "train" else list(range(min(16, len(original))))
        loader = DataLoader(Subset(original, indices), batch_size=loader.batch_size,
                            shuffle=split == "train", collate_fn=loader.collate_fn)
        ids = [original[index]["doc_id"] for index in indices]
        _json(RUN / ("canary_" + split + "_ids.json"), ids)
    event("dataset_loaded", status="loading_data", split=split,
          documents=len(loader.dataset), batches=len(loader))
    return loader


def reference_initialized(model, args, train_loader):
    parameters = list(model.parameters())
    if any(parameter.dtype != torch.float32 for parameter in parameters):
        raise RuntimeError("Author reference requires FP32 parameters")
    if os.environ.get("MAVEN_REFERENCE_GRADIENT_CHECKPOINTING") == "1":
        model.encoder.model.module.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    packages = {}
    for name in ("torch", "transformers", "numpy", "scipy", "scikit-learn"):
        packages[name] = importlib.metadata.version(name)
    _json(RUN / "runtime.json", {"packages": packages, "torch_cuda": torch.version.cuda,
          "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
          "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
          "parameter_count": sum(p.numel() for p in parameters),
          "gradient_checkpointing": os.environ.get("MAVEN_REFERENCE_GRADIENT_CHECKPOINTING") == "1",
          "scheduler_denominator": len(train_loader) * args.epochs,
          "args": vars(args)})
    event("training_initialized", status="training", scheduler_denominator=len(train_loader) * args.epochs,
          max_optimizer_updates=args.reference_max_updates, train_batches_per_epoch=len(train_loader))


def reference_micro(step, epoch, loss):
    value = float(loss.detach())
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite original upstream loss at micro step %s" % step)
    event("micro_step", status="training", micro_step=step, epoch_index=epoch,
          accumulated_scaled_loss=value,
          cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0)


def reference_update(step, epoch, encoder_optimizer, head_optimizer, scheduler):
    event("optimizer_update", status="training", micro_step=step, epoch_index=epoch,
          optimizer_updates=step // 4, encoder_lr=encoder_optimizer.param_groups[0]["lr"],
          head_lr=head_optimizer.param_groups[0]["lr"], scheduler_step=scheduler.last_epoch)


@torch.no_grad()
def reference_before_update(model, step):
    gradients = [(name, parameter.grad) for name, parameter in model.named_parameters()
                 if parameter.grad is not None]
    if gradients and not bool(torch.stack([torch.isfinite(gradient).all() for _, gradient in gradients]).all()):
        bad = [name for name, gradient in gradients if not bool(torch.isfinite(gradient).all())]
        event("nonfinite_gradient", status="failed", micro_step=step, nonfinite_gradient_parameters=bad)
        raise FloatingPointError("Non-finite original upstream gradients: " + ", ".join(bad))
    PROGRESS["gradients_checked_finite_at_micro_step"] = step


@torch.no_grad()
def reference_weight_check(model, task):
    floating = [(name, tensor) for name, tensor in model.state_dict().items()
                if torch.is_floating_point(tensor) or torch.is_complex(tensor)]
    if floating and not bool(torch.stack([torch.isfinite(tensor).all() for _, tensor in floating]).all()):
        bad = [name for name, tensor in floating if not bool(torch.isfinite(tensor).all())]
        event("nonfinite_checkpoint_weights", status="failed", selected_task=task, nonfinite_weight_names=bad)
        raise FloatingPointError("Non-finite saved reference model weights: " + ", ".join(bad))
    FINITE_WEIGHT_CHECKS[task] = True
    _json(RUN / "weights_finite_by_checkpoint.json", FINITE_WEIGHT_CHECKS)


def reference_should_stop(step, args):
    return bool(args.reference_max_updates and step % args.accumulation_steps == 0
                and step // args.accumulation_steps >= args.reference_max_updates)


def reference_validation(result, desc):
    event("validation", status="evaluating", validation_description=desc,
          upstream_augmented_validation=result)


def reference_saved_checkpoint(task, step, epoch, metrics):
    BEST_SELECTION[task] = {"micro_step": step, "optimizer_updates": step // 4,
                            "epoch_index": epoch, "metrics_at_selection": metrics,
                            "selection": "upstream_strict_improvement"}
    _json(RUN / "best_selection.json", BEST_SELECTION)
    event("checkpoint_selected", selected_task=task, micro_step=step, optimizer_updates=step // 4)


def reference_canary_fallback(model, optimizer, scheduler, output_dir, step):
    if not CANARY:
        return
    for task in ("COREFERENCE", "TEMPORAL", "CAUSAL", "SUBEVENT"):
        path = Path(output_dir) / ("best_" + task)
        if not path.exists():
            # Diagnostic export only: full experiments never manufacture a best.
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "canary_fallback": True}, path)
            FALLBACK_TASKS.append(task)
            BEST_SELECTION[task] = {"micro_step": step, "optimizer_updates": step // 4,
                                    "selection": "canary_final_state_diagnostic_fallback"}
    _json(RUN / "best_selection.json", BEST_SELECTION)
    _json(RUN / "canary_diagnostics.json", {"fallback_tasks": FALLBACK_TASKS,
          "paper_comparison_eligible": False, "reason": "Canary is at most two optimizer updates"})


def _valid_candidates():
    source = Path(os.environ["MAVEN_REFERENCE_DATA"]) / "valid.jsonl"
    destination = RUN / "valid_candidates.jsonl"
    records = []
    with open(source, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    if CANARY:
        selected = set(json.loads((RUN / "canary_valid_ids.json").read_text()))
        records = [record for record in records if record["id"] in selected]
    with open(destination, "w", encoding="utf-8") as stream:
        for record in records:
            candidate = {"id": record["id"], "tokens": record["tokens"],
                         "event_mentions": [mention for event in record["events"] for mention in event["mention"]],
                         "TIMEX": record["TIMEX"]}
            stream.write(json.dumps(candidate) + "\n")
    return destination, {record["id"]: record for record in records}


def reference_best_validation(task, model, loader, namespace):
    reference_weight_check(model, task)
    metrics = namespace["evaluate"](model, loader, desc="Final best " + task + " valid")
    BEST_VALIDATION[task] = metrics
    _json(RUN / "validation_by_checkpoint.json", BEST_VALIDATION)
    candidates, gold = _valid_candidates()
    predict_name, dump_name = {
        "COREFERENCE": ("coref_predict", "coref_dump"),
        "TEMPORAL": ("temp_predict", "temporal_dump"),
        "CAUSAL": ("causal_predict", "causal_dump"),
        "SUBEVENT": ("subevent_predict", "subevent_dump"),
    }[task]
    preds = namespace[predict_name](model, loader)
    task_predictions = {}
    namespace[dump_name](str(candidates), preds, task_predictions)
    field = "coreference" if task == "COREFERENCE" else task.lower() + "_relations"
    for doc_id, prediction in task_predictions.items():
        if doc_id not in BEST_DEV_COMPOSITE:
            BEST_DEV_COMPOSITE[doc_id] = prediction
        else:
            BEST_DEV_COMPOSITE[doc_id][field] = prediction[field]
    spec = importlib.util.spec_from_file_location("pinned_best_official_evaluate", RUN / "executed" / "evaluate.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    raw = (official.evaluate_coreference(gold, task_predictions) if task == "COREFERENCE"
           else official.evaluate(gold, task_predictions, task.lower()))
    BEST_RAW_VALIDATION[task] = raw
    _json(RUN / "raw_direction_validation_by_checkpoint.json", BEST_RAW_VALIDATION)
    event("final_best_validation", status="exporting", final_task=task)


def reference_finished():
    _, gold = _valid_candidates()
    if set(BEST_DEV_COMPOSITE) != set(gold):
        raise RuntimeError("Composite best dev predictions have incomplete document coverage")
    predictions_path = RUN / "best_valid_predictions.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as stream:
        for doc_id in sorted(BEST_DEV_COMPOSITE):
            stream.write(json.dumps(BEST_DEV_COMPOSITE[doc_id]) + "\n")
    spec = importlib.util.spec_from_file_location("pinned_official_evaluate", RUN / "executed" / "evaluate.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    raw_scores = {"coreference": official.evaluate_coreference(gold, BEST_DEV_COMPOSITE)}
    for name in ("temporal", "causal", "subevent"):
        raw_scores[name] = official.evaluate(gold, BEST_DEV_COMPOSITE, name)
    coref = raw_scores["coreference"]
    coref_mean = sum(coref[name + "_f1"] for name in ("muc", "b_cubed", "ceaf", "blanc")) / 4
    raw_scores["coreference_mean_f1"] = coref_mean
    raw_scores["overall_f1"] = (coref_mean + sum(raw_scores[n][n + "_f1"] for n in ("temporal", "causal", "subevent"))) / 4
    selected = {task: metrics[task] for task, metrics in BEST_VALIDATION.items()}
    _json(RUN / "best_valid_metrics.json", {"split": "valid", "canary": CANARY,
          "selection": "four independently selected upstream task checkpoints",
          "upstream_augmented_protocol_fraction": selected,
          "official_raw_direction_protocol_percent": raw_scores,
          "canary_fallback_tasks": FALLBACK_TASKS,
          "weights_finite_by_checkpoint": FINITE_WEIGHT_CHECKS,
          "test_scored": False, "paper_reproduction_success": None})
    event("artifacts_complete", status="artifacts_complete", paper_reproduction_success=None,
          test_scored=False, canary_fallback_tasks=FALLBACK_TASKS)
'''


def checked_path(value):
    path = Path(value).expanduser().resolve()
    if any(str(path) == banned or str(path).startswith(banned + "/") for banned in PROTECTED):
        raise ValueError(f"Protected iCloud path: {path}")
    return path


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def git_bytes(repo, *arguments):
    return subprocess.check_output(["git", "-C", str(repo), *arguments])


def extract_sources(repo, destination):
    """Read Git blobs, never the modified checkout or a network remote."""
    names = git_bytes(repo, "ls-tree", "-r", "--name-only", UPSTREAM_COMMIT, "joint", "utils").decode().splitlines()
    names += ["evaluate.py", "LICENSE"]
    blobs = {}
    for name in names:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unexpected Git path")
        blob = git_bytes(repo, "show", UPSTREAM_COMMIT + ":" + name)
        target = checked_path(destination / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        blobs[name] = blob
    return blobs


def replace_exact(text, before, after, count=1):
    actual = text.count(before)
    if actual != count:
        raise ValueError(f"Pinned-source patch mismatch: expected {count}, found {actual}: {before!r}")
    return text.replace(before, after)


def patched_sources(blobs):
    """Every replacement is anchored to the pinned source and audited in a diff."""
    text = blobs["joint/main.py"].decode().replace("\r\n", "\n")
    text = replace_exact(text,
        "from transformers import AdamW, RobertaTokenizer, get_linear_schedule_with_warmup",
        "from transformers import RobertaTokenizer, get_linear_schedule_with_warmup\nfrom src.legacy_hf_adamw import AdamW\nfrom src.reference_hooks import (base_model_path, reference_loader, reference_initialized, reference_micro, reference_update, reference_before_update, reference_should_stop, reference_validation, reference_saved_checkpoint, reference_canary_fallback, reference_best_validation, reference_finished)")
    text = replace_exact(text, "def evaluate(model, dataloader, desc=\"\"):", "@torch.no_grad()\ndef evaluate(model, dataloader, desc=\"\"):")
    text = replace_exact(text, "    return result_collection", "    reference_validation(result_collection, desc)\n    return result_collection")
    text = replace_exact(text, "    args = parser.parse_args()", "    parser.add_argument(\"--reference_max_updates\", type=int, default=None)\n    args = parser.parse_args()")
    text = replace_exact(text, '    output_dir = Path(f"./output/{args.seed}/MAVEN-ERE")', '    output_dir = Path(os.environ["MAVEN_REFERENCE_OUTPUT"]) / "artifacts"')
    text = replace_exact(text, 'sys.stdout = open(os.path.join(output_dir, "log.txt"), \'w\')', 'sys.stdout = open(os.path.join(output_dir, "log.txt"), \'w\', buffering=1)')
    text = replace_exact(text, 'RobertaTokenizer.from_pretrained("/data/MODELS/roberta-base")', 'RobertaTokenizer.from_pretrained(base_model_path(), local_files_only=True)')
    for split, variable in (("train", "train_dataloader"), ("valid", "dev_dataloader"), ("test", "test_dataloader")):
        text = replace_exact(text, f'get_dataloader(tokenizer, "{split}", max_length=256,', f'get_dataloader(tokenizer, "{split}", data_dir=os.environ["MAVEN_REFERENCE_DATA"], max_length=256,')
        pattern = re.compile(r"^( +)" + variable + r" = .*get_dataloader.*$", re.MULTILINE)
        matches = list(pattern.finditer(text))
        if len(matches) != 1:
            raise ValueError("Unexpected loader assignment " + variable)
        match = matches[0]
        original = match.group(0)
        text = text[:match.start()] + original + "\n" + match.group(1) + f'{variable} = reference_loader({variable}, "{split}")' + text[match.end():]
    text = replace_exact(text, "    model = to_cuda(model)", "    model = to_cuda(model)\n    reference_initialized(model, args, train_dataloader)")
    text = replace_exact(text, "                glb_step += 1", "                glb_step += 1\n                reference_micro(glb_step, epoch, loss)")
    text = replace_exact(text, "                    optimizer.step()", "                    reference_before_update(model, glb_step)\n                    optimizer.step()")
    text = replace_exact(text, "                    bert_optimizer.zero_grad()", "                    bert_optimizer.zero_grad()\n                    reference_update(glb_step, epoch, bert_optimizer, optimizer, scheduler)")
    text = replace_exact(text, "                if glb_step % args.eval_steps == 0:", "                if glb_step % args.eval_steps == 0 or reference_should_stop(glb_step, args):")
    text = replace_exact(text,
        '                            torch.save(state, os.path.join(output_dir, "best_%s"%(k)))',
        '                            torch.save(state, os.path.join(output_dir, "best_%s"%(k)))\n                            reference_saved_checkpoint(k, glb_step, epoch, res)')
    text = replace_exact(text, "    dump_results={}",
        "                if reference_should_stop(glb_step, args):\n                    break\n            if reference_should_stop(glb_step, args):\n                break\n    reference_canary_fallback(model, optimizer, scheduler, output_dir, glb_step)\n    dump_results={}")
    text = replace_exact(text,
        '        state = torch.load(os.path.join(output_dir, "best_%s"%(k)))',
        '        state = torch.load(os.path.join(output_dir, "best_%s"%(k)), weights_only=False)')
    text = replace_exact(text, '        model.load_state_dict(state["model"])',
        '        model.load_state_dict(state["model"])\n        reference_best_validation(k, model, dev_dataloader, globals())')
    text = replace_exact(text, '"../data/MAVEN_ERE/test.jsonl"', 'os.path.join(os.environ["MAVEN_REFERENCE_DATA"], "test.jsonl")', count=4)
    text = replace_exact(text, "    sys.stdout.close()", "    reference_finished()\n    sys.stdout.close()")
    model = blobs["joint/src/model.py"].decode().replace("\r\n", "\n")
    model = replace_exact(model, "from transformers import AutoConfig, AutoModel, BertConfig, RobertaModel", "from transformers import AutoConfig, AutoModel, RobertaConfig, RobertaModel\nfrom .reference_hooks import base_model_path")
    model = replace_exact(model, 'model_name="/data/MODELS/roberta-base"', 'model_name=base_model_path()', count=2)
    model = replace_exact(model, "AutoConfig.from_pretrained(model_name)", "AutoConfig.from_pretrained(model_name, local_files_only=True)")
    model = replace_exact(model, "isinstance(config, BertConfig)", "isinstance(config, RobertaConfig)")
    model = replace_exact(model, "RobertaModel.from_pretrained(model_name)", 'RobertaModel.from_pretrained(model_name, local_files_only=True, attn_implementation="eager")')
    metrics = blobs["joint/src/metrics.py"].decode().replace("\r\n", "\n")
    metrics = replace_exact(metrics, "from pyparsing import col\n", "")
    assignment = blobs["joint/src/linear_assignment.py"].decode().replace("\r\n", "\n")
    assignment = re.sub(r"np\.bool\b", "bool", assignment)
    assignment = re.sub(r"np\.int\b", "int", assignment)
    return {"joint/main.py": text, "joint/src/model.py": model,
            "joint/src/metrics.py": metrics, "joint/src/linear_assignment.py": assignment}


def capture_apache_license(destination, required):
    """Copy the complete license from an explicit OS/package metadata location."""
    candidates = [Path("/usr/share/common-licenses/Apache-2.0")]
    try:
        distribution = importlib.metadata.distribution("transformers")
        for relative in distribution.files or []:
            if relative.name.upper() in ("LICENSE", "LICENSE.TXT", "LICENSE.MD"):
                candidates.append(distribution.locate_file(relative))
    except importlib.metadata.PackageNotFoundError:
        pass
    destination.mkdir(exist_ok=True)
    for candidate in candidates:
        candidate = checked_path(candidate)
        if candidate.is_file():
            content = candidate.read_text(encoding="utf-8")
            if "Apache License" in content and "Version 2.0" in content:
                target = destination / "Apache-2.0.txt"
                target.write_text(content, encoding="utf-8")
                return {"source": str(candidate), "sha256": sha256_file(target)}
    if required:
        raise RuntimeError("Full Apache-2.0 license unavailable in explicit OS/Transformers metadata locations")
    return {"source": None, "note": "Prepare-only did not require installed Transformers; runtime requires full license"}


def prepare_run(args, repo, output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("Reference output directory must be empty; existing artifacts are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    blobs = extract_sources(repo, output / "original")
    patched = patched_sources(blobs)
    executed = output / "executed"
    changes, hashes = [], {}
    for name, original in blobs.items():
        target = checked_path(executed / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = patched[name].encode() if name in patched else original
        target.write_bytes(content)
        hashes[name] = {"original_sha256": sha256_bytes(original), "executed_sha256": sha256_bytes(content), "modified": original != content}
        if name in patched:
            changes.extend(difflib.unified_diff(original.decode().replace("\r\n", "\n").splitlines(True),
                patched[name].splitlines(True), fromfile="a/" + name, tofile="b/" + name))
    generated = {"joint/src/legacy_hf_adamw.py": LEGACY_ADAMW_SOURCE,
                 "joint/src/reference_hooks.py": REFERENCE_HOOKS_SOURCE,
                 "joint/src/__init__.py": "# Isolate the extracted author src package from unrelated installed src packages.\n"}
    for name, content in generated.items():
        target = checked_path(executed / name)
        target.write_text(content, encoding="utf-8")
        hashes[name] = {"generated_sha256": sha256_file(target), "source": HF_ADAMW_SOURCE if "adamw" in name else "this launcher"}
        changes.extend(difflib.unified_diff([], content.splitlines(True), fromfile="/dev/null", tofile="b/" + name))
    # AST parsing does not import or run model code and creates no .pyc files.
    parsed = []
    for name in hashes:
        if name.endswith(".py"):
            ast.parse((executed / name).read_text(encoding="utf-8"), filename=name)
            parsed.append(name)
    (output / "compatibility.patch").write_text("".join(changes), encoding="utf-8")
    apache = capture_apache_license(output / "licenses", required=not args.prepare_only)
    (output / "licenses" / "HF-AdamW-NOTICE.txt").write_text(
        "SPDX-License-Identifier: Apache-2.0\nCopyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.\n"
        + "Legacy AdamW adapted from " + HF_ADAMW_SOURCE + "\n"
        + "Standalone adapter retains update order, uncorrected-v epsilon placement, bias correction and post-update weight decay.\n", encoding="utf-8")
    manifest = {"upstream_commit": UPSTREAM_COMMIT, "source_repository": "https://github.com/THU-KEG/MAVEN-ERE",
        "source_files": hashes, "ast_parsed_files": parsed, "apache_license": apache,
        "compatibility_patch_sha256": sha256_file(output / "compatibility.patch"),
        "line_endings": "Original snapshots retain Git blob bytes; patched text files normalize CRLF to LF",
        "runner_sha256": sha256_file(Path(__file__).resolve()), "args": vars(args),
        "protocol": {"batch_size": 8, "accumulation_steps": 4, "nominal_effective_documents": 32,
          "chunk_length": 256, "precision": "float32", "head_lr": 3e-4, "encoder_lr": 2e-5,
          "warmup_optimizer_updates": 200, "eval_interval_microsteps": 200,
          "loss_weights": [0.4, 2.0, 4.0, 4.0], "scheduler_denominator": "len(train_dataloader)*epochs",
          "accumulation": "global microstep modulo 4, carries across epoch boundaries; no final flush",
          "selection": "upstream four task checkpoints; coreference updates when any of four metrics improves",
          "coreference_targets": "unmodified upstream fill_expand, including first-mention omission",
          "gradient_checkpointing": args.gradient_checkpointing, "test_scored": False},
        "diagnostic_deviations": ["64 train docs selected to cover long documents and task positives", "first 16 valid/test docs", "forced dev evaluation at max-update boundary", "explicitly marked fallback only for canary tasks whose F1 never exceeded zero"] if args.canary else ["forced dev evaluation at max-update boundary"] if args.max_steps else [],
        "created_utc": datetime.now(timezone.utc).isoformat(), "host_platform": platform.platform(),
        "paper_reproduction_success": None}
    write_json(output / "source_manifest.json", manifest)
    return manifest


def dataset_manifest(data_dir):
    result, seen = {}, set()
    for split in ("train", "valid", "test"):
        source = checked_path(data_dir / (split + ".jsonl"))
        ids = set()
        with open(source, encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                document = json.loads(line)
                if document["id"] in ids or document["id"] in seen:
                    raise ValueError("Duplicate or overlapping document ID in " + split)
                ids.add(document["id"])
                if split != "test" and "events" not in document:
                    raise ValueError("Gold relations/events missing from " + split)
        if not ids:
            raise ValueError("Empty dataset split " + split)
        seen.update(ids)
        result[split] = {"documents": len(ids), "sha256": sha256_file(source)}
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Existing local RoBERTa snapshot")
    parser.add_argument("--data-dir", required=True, help="train/valid/test.jsonl, never modified")
    parser.add_argument("--output-dir", required=True, help="New empty run directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, help="Optimizer updates, not microbatches")
    parser.add_argument("--canary", "--smoke", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False)
    parser.add_argument("--prepare-only", action="store_true", help="Extract, patch, AST-check; no torch/model execution")
    args = parser.parse_args(argv)
    if args.epochs < 1 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("epochs/max-steps must be positive")
    if args.canary:
        args.max_steps = min(args.max_steps or 2, 2)
    return args


def inherited_gpu_lock_fds(environment):
    """Keep the campaign's flock open in the real trainer if this wrapper dies.

    The descriptor number remains an internal process detail, never a logged
    identity. Absent, malformed or already closed descriptors are not passed.
    """
    raw_descriptor = environment.get("MAVEN_CAMPAIGN_GPU_LOCK_FD")
    if raw_descriptor is None:
        return ()
    try:
        descriptor = int(raw_descriptor)
        if descriptor < 0:
            return ()
        os.fstat(descriptor)
    except (ValueError, TypeError, OSError, OverflowError):
        return ()
    return (descriptor,)


def main(argv=None):
    args = parse_args(argv)
    repo = checked_path(Path(__file__).resolve().parents[1])
    output = checked_path(args.output_dir)
    base = checked_path(args.base_model)
    data = checked_path(args.data_dir)
    if not base.is_dir() or not data.is_dir():
        raise ValueError("base-model and data-dir must be existing local directories")
    datasets = dataset_manifest(data)
    prepare_run(args, repo, output)
    write_json(output / "dataset_manifest.json", datasets)
    write_json(output / "progress.json", {"status": "prepared", "paper_reproduction_success": None})
    if args.prepare_only:
        print(json.dumps({"status": "prepared_only", "output": str(output), "torch_imported": False}))
        return 0
    environment = os.environ.copy()
    environment.update({"MAVEN_REFERENCE_BASE": str(base), "MAVEN_REFERENCE_DATA": str(data),
        "MAVEN_REFERENCE_OUTPUT": str(output), "MAVEN_REFERENCE_CANARY": "1" if args.canary else "0",
        "MAVEN_REFERENCE_GRADIENT_CHECKPOINTING": "1" if args.gradient_checkpointing else "0",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"})
    command = [sys.executable, "-u", "main.py", "--epochs", str(args.epochs),
        "--seed", str(args.seed), "--batch_size", "8", "--accumulation_steps", "4",
        "--eval_steps", "200", "--lr", "3e-4", "--bert_lr", "2e-5"]
    if args.max_steps:
        command.extend(["--reference_max_updates", str(args.max_steps)])
    lock_fds = inherited_gpu_lock_fds(environment)
    recorded_environment = {key: value for key, value in environment.items()
                            if key.startswith("MAVEN_REFERENCE_") or key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONUNBUFFERED")}
    recorded_environment["NVIDIA_TF32_OVERRIDE"] = environment.get("NVIDIA_TF32_OVERRIDE")
    write_json(output / "launch.json", {"command": command, "cwd": str(output / "executed" / "joint"),
        "environment_overrides": recorded_environment, "campaign_gpu_lock_inherited": bool(lock_fds)})
    with open(output / "console.log", "w", encoding="utf-8", buffering=1) as console:
        process = subprocess.Popen(command, cwd=output / "executed" / "joint", env=environment,
                                   stdout=console, stderr=subprocess.STDOUT, pass_fds=lock_fds)
        def forward_signal(signum, frame):
            if process.poll() is None:
                process.send_signal(signum)
        previous = {signum: signal.signal(signum, forward_signal) for signum in (signal.SIGTERM, signal.SIGINT)}
        try:
            return_code = process.wait()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    progress_file = output / "progress.json"
    progress = json.loads(progress_file.read_text())
    required = [output / "best_valid_metrics.json", output / "best_valid_predictions.jsonl",
                output / "artifacts" / "test_prediction.jsonl"]
    required += [output / "artifacts" / ("best_" + task) for task in ("COREFERENCE", "TEMPORAL", "CAUSAL", "SUBEVENT")]
    missing = [str(path) for path in required if not path.is_file()]
    ok = return_code == 0 and not missing
    status = ("smoke" if args.canary else "completed") if ok else ("interrupted" if return_code < 0 or return_code in (130, 143) else "failed")
    progress.update({"status": status,
                     "return_code": return_code, "missing_artifacts": missing,
                     "paper_reproduction_success": None, "test_scored": False})
    write_json(progress_file, progress)
    artifacts = {str(path.relative_to(output)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                 for path in required if path.is_file()}
    write_json(output / "run_summary.json", {**progress, "canary": args.canary,
        "artifacts": artifacts,
        "comparison_note": "Validation single-seed results do not establish equivalence to paper test-set five-run means."})
    metrics_file = output / "best_valid_metrics.json"
    metrics = json.loads(metrics_file.read_text()) if metrics_file.is_file() else {}
    selection_file = output / "best_selection.json"
    selection = json.loads(selection_file.read_text()) if selection_file.is_file() else {}
    checkpoints = {}
    for task in ("COREFERENCE", "TEMPORAL", "CAUSAL", "SUBEVENT"):
        relative = "artifacts/best_" + task
        identity = selection.get(task, {})
        checkpoints[task.lower()] = {"path": str(output / relative),
            "sha256": artifacts.get(relative, {}).get("sha256"),
            "selection": identity.get("selection"),
            "selected_micro_step": identity.get("micro_step"),
            "selected_optimizer_updates": identity.get("optimizer_updates")}
    complete = ok and not args.canary and args.max_steps is None and args.epochs == 100
    write_json(output / "reference_summary.json", {"status": status,
        "official_protocol": metrics.get("official_raw_direction_protocol_percent"),
        "upstream_augmented_protocol_fraction": metrics.get("upstream_augmented_protocol_fraction"),
        "task_checkpoints": checkpoints, "checkpoint_count": 4, "split": "valid",
        "training": {"requested_epochs": args.epochs,
            "epochs_completed": progress.get("micro_step", 0) // progress["train_batches_per_epoch"] if progress.get("train_batches_per_epoch") else 0,
            "micro_steps": progress.get("micro_step", 0), "optimizer_updates": progress.get("optimizer_updates", 0),
            "max_steps": args.max_steps, "precision": "float32", "batch_size": 8, "accumulation_steps": 4,
            "gradient_checkpointing": args.gradient_checkpointing},
        "upstream_commit": UPSTREAM_COMMIT, "experiment_complete": complete, "leaderboard_eligible": complete,
        "canary_fallback_tasks": metrics.get("canary_fallback_tasks", []),
        "weights_finite_by_checkpoint": metrics.get("weights_finite_by_checkpoint", {}),
        "test_predictions": str(output / "artifacts/test_prediction.jsonl") if ok else None,
        "test_scored": False, "return_code": return_code, "missing_artifacts": missing,
        "paper_reproduction_success": None})
    print(json.dumps({"status": progress["status"], "return_code": return_code, "output": str(output)}))
    return 0 if ok else (return_code if return_code > 0 else 1)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
