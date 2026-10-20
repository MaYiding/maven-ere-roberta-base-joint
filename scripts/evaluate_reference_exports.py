#!/usr/bin/env python3
"""V2 CPU-only, read-only assessment of completed MAVEN-ERE reference checkpoints.

Writes only a fresh postprocessing output directory. It does not train, publish,
acquire the GPU lock, modify a source checkout, or change a completed run.
--canary-only audits conversion/AutoModel loading on real validation documents,
including the longest documents, without making full-validation improvement claims.
"""

import argparse
from collections import Counter, OrderedDict
import copy
from datetime import datetime, timezone
import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace


FROZEN_COMMIT = "8a49171cfe160d85877ffea5e7247c2141bcf3d4"
UPSTREAM_COMMIT = "ac81a9711a69f43f55bfbc50b3bb573fd11c64b0"
FROZEN_HASHES = {
    "maven_ere/__init__.py": "84ef6a2e20c0f2838fb2e3db644b022fbbb9d3714efac815605e24e495e166ba",
    "maven_ere/configuration_maven_ere.py": "f57ad7d239ab5fa3520b59fdd76163f3eb7dfe3eba0a1b98002f09c023a7839d",
    "maven_ere/modeling_maven_ere.py": "b7d09b6622b7b5f4a5c08412f6ada89eb3923f98868b91f4afef947098ae9104",
}
TASKS = ("COREFERENCE", "TEMPORAL", "CAUSAL", "SUBEVENT")
RELATIONS = {
    "temporal": {"BEFORE": 0, "OVERLAP": 1, "CONTAINS": 2, "SIMULTANEOUS": 3,
                 "ENDS-ON": 4, "BEGINS-ON": 5, "NONE": 6},
    "causal": {"NONE": 0, "PRECONDITION": 1, "CAUSE": 2},
    "subevent": {"NONE": 0, "subevent": 1},
}
PROTECTED = ("/Users/mayiding/Library/Mobile Documents", "/Users/mayiding/Library/CloudStorage",
             "/Users/mayiding/Library/Application Support/CloudDocs")
TORCH = None
START = time.monotonic()
ACTIVE_OUTPUT = None
PROGRESS = {"status": "initializing", "test_scored": False}
ASSESSMENT_VERSION = 2


def checked_path(value):
    path = Path(value).expanduser().resolve()
    if any(str(path) == prefix or str(path).startswith(prefix + "/") for prefix in PROTECTED):
        raise ValueError("Protected iCloud path: " + str(path))
    return path


def json_scalar(value):
    """Convert NumPy metric scalars; reject arrays and arbitrary objects."""
    if type(value).__module__.split('.', 1)[0] == 'numpy' and getattr(value, 'ndim', None) == 0:
        return value.item()
    raise TypeError('Unsupported JSON value: ' + type(value).__name__)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False, default=json_scalar)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def update(stage, **values):
    PROGRESS.update(values)
    PROGRESS.update({"stage": stage, "elapsed_seconds": time.monotonic() - START,
                     "timestamp_utc": datetime.now(timezone.utc).isoformat()})
    atomic_json(ACTIVE_OUTPUT / "progress.json", PROGRESS)
    with open(ACTIVE_OUTPUT / "events.jsonl", "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"stage": stage, **values,
                                "timestamp_utc": PROGRESS["timestamp_utc"]}, allow_nan=False, default=json_scalar) + "\n")
    print(json.dumps({"stage": stage, **values}, ensure_ascii=False, allow_nan=False, default=json_scalar), flush=True)


def verify_hash(path, expected, description):
    actual = sha256_file(path)
    if expected and actual != expected:
        raise ValueError(description + " SHA256 mismatch")
    return actual


def snapshot_code(args, output, source_manifest):
    frozen = output / "private" / "frozen_code"
    for relative, digest in FROZEN_HASHES.items():
        source = checked_path(args.source_root / relative)
        verify_hash(source, digest, relative + " frozen " + FROZEN_COMMIT)
        target = frozen / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    legacy = output / "private" / "legacy_code"
    manifest_files = source_manifest["source_files"]
    for relative, identity in manifest_files.items():
        if not relative.startswith("joint/src/") and relative not in ("evaluate.py", "LICENSE"):
            continue
        source = checked_path(args.reference_run / "executed" / relative)
        expected = identity.get("executed_sha256") or identity.get("generated_sha256")
        if not expected:
            raise ValueError("Missing executed source identity for " + relative)
        verify_hash(source, expected, "Reference executed source " + relative)
        if relative == "joint/src/utils.py" and identity["original_sha256"] != expected:
            raise ValueError("Reference utils is modified; original fill_expand required")
        target = legacy / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    if not (legacy / "joint/src/__init__.py").exists():
        (legacy / "joint/src/__init__.py").write_text("# Isolated reference source package.\n")
    # All legacy hooks, if imported by model.py, point exclusively to new output.
    hook_output = output / "private" / "legacy_hook_output"
    hook_output.mkdir(parents=True)
    os.environ.update({"MAVEN_REFERENCE_OUTPUT": str(hook_output), "MAVEN_REFERENCE_BASE": str(args.base_model),
                       "MAVEN_REFERENCE_DATA": str(args.data_dir), "MAVEN_REFERENCE_CANARY": "0"})
    sys.path.insert(0, str(legacy / "joint"))
    sys.path.insert(0, str(frozen))
    from maven_ere import MavenEreConfig, MavenEreModel
    legacy_model = importlib.import_module("src.model")
    legacy_data = importlib.import_module("src.data")
    legacy_metrics = importlib.import_module("src.metrics")
    spec = importlib.util.spec_from_file_location("reference_export_official_evaluate", legacy / "evaluate.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    return MavenEreConfig, MavenEreModel, legacy_model, legacy_data, legacy_metrics, official, legacy


def read_valid(path):
    records = OrderedDict()
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["id"] in records or "events" not in record:
                raise ValueError("Duplicate document ID or absent validation gold")
            records[record["id"]] = record
    return records


def semantic_probe_indices(dataset, maximum=12):
    selected = []
    def add(index):
        if index not in selected:
            selected.append(index)
    add(0)
    add(min(range(len(dataset)), key=lambda i: dataset[i]["input_ids"].shape[0]))
    add(max(range(len(dataset)), key=lambda i: dataset[i]["input_ids"].shape[0]))
    add(max(range(len(dataset)), key=lambda i: sum(map(len, dataset[i]["event_spans"]))))
    for task, labels in RELATIONS.items():
        match = next((i for i in range(len(dataset)) if bool(((dataset[i][task + "_labels"] >= 0) &
                       (dataset[i][task + "_labels"] != labels["NONE"])).any())), None)
        if match is None:
            raise ValueError("Validation lacks positive " + task)
        add(match)
    coref = next((i for i in range(len(dataset)) if any(len(c) > 1 for c in dataset[i]["coref_labels"])), None)
    if coref is None:
        raise ValueError("Validation lacks a non-singleton coreference cluster")
    add(coref)
    for index in range(len(dataset)):
        if len(selected) >= min(maximum, len(dataset)):
            break
        add(index)
    return selected


def check_finite_weights(state, label):
    for name, tensor in state.items():
        if not TORCH.is_tensor(tensor):
            raise TypeError(label + " contains a non-tensor model entry: " + name)
        if TORCH.is_floating_point(tensor):
            if tensor.dtype != TORCH.float32 or not bool(TORCH.isfinite(tensor).all()):
                raise ValueError(label + " has non-FP32 or nonfinite weights: " + name)


def strict_map_state(source, target):
    prefix = "encoder.model.module."
    mapped = OrderedDict()
    for original, tensor in source.items():
        if original.startswith(prefix):
            name = "encoder." + original[len(prefix):]
        elif original.startswith(("coref_scorer.", "temporal_scorer.", "causal_scorer.", "subevent_scorer.")):
            name = original
        else:
            raise ValueError("Unexpected legacy model state key: " + original)
        if name in mapped:
            raise ValueError("Checkpoint key collision: " + name)
        mapped[name] = tensor
    expected = target.state_dict()
    if set(mapped) != set(expected):
        raise ValueError("Strict mapping mismatch; missing=" + str(sorted(set(expected) - set(mapped)))
                         + "; unexpected=" + str(sorted(set(mapped) - set(expected))))
    for name, tensor in mapped.items():
        if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
            raise ValueError("Strict shape/dtype mismatch: " + name)
    target.load_state_dict(mapped, strict=True)
    for name, tensor in target.state_dict().items():
        if not TORCH.equal(tensor, mapped[name]):
            raise ValueError("State did not load exactly: " + name)
    return {"source_prefix": prefix, "target_prefix": "encoder.", "mapped_keys": len(mapped),
            "head_keys_unchanged": True, "strict": True, "all_loaded_tensors_exact": True}


def canonical_clusters(scores):
    n = scores.shape[0]
    if n == 0:
        return []
    antecedents = scores.argmax(-1).tolist()
    parent = list(range(n))
    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    for index, antecedent in enumerate(antecedents):
        if not 0 <= antecedent <= index:
            raise ValueError("Invalid antecedent order")
        if antecedent != index:
            parent[root(index)] = root(antecedent)
    groups = {}
    for index in range(n):
        groups.setdefault(root(index), []).append(index)
    return sorted(groups.values(), key=lambda group: group[0])


def unpack(output, legacy=False):
    if legacy:
        return output[0], {task: scores for task, scores in zip(RELATIONS, output[1:])}
    return output.coref_log_probs, {task: getattr(output, task + "_logits") for task in RELATIONS}


def mentions(record):
    combined = sorted([m for event in record["events"] for m in event["mention"]] + record["TIMEX"],
                      key=lambda m: (m["sent_id"], m["offset"][0]))
    return combined, [m for m in combined if not m["id"].startswith("TIME")]


def decode_batch(output, batch, records, legacy=False):
    corefs, relations = unpack(output, legacy)
    result = {}
    for doc_index, doc_id in enumerate(batch["doc_id"]):
        combined, events = mentions(records[doc_id])
        clusters = canonical_clusters(corefs[doc_index])
        prediction = {"id": doc_id, "coreference": [[events[i]["id"] for i in cluster] for cluster in clusters],
            "temporal_relations": {name: [] for name in RELATIONS["temporal"] if name != "NONE"},
            "causal_relations": {"CAUSE": [], "PRECONDITION": []}, "subevent_relations": []}
        ids = {task: logits[doc_index].argmax(-1).tolist() for task, logits in relations.items()}
        index = 0
        for left_index, left in enumerate(combined):
            for right_index, right in enumerate(combined):
                if left_index == right_index:
                    continue
                for task, label_map in RELATIONS.items():
                    if task != "temporal" and (left["id"].startswith("TIME") or right["id"].startswith("TIME")):
                        continue
                    label = ids[task][index]
                    if label == label_map["NONE"]:
                        continue
                    pair = [left["id"], right["id"]]
                    if task == "subevent":
                        prediction["subevent_relations"].append(pair)  # [parent, child]
                    else:
                        name = next(name for name, value in label_map.items() if value == label)
                        prediction[task + "_relations"][name].append(pair)
                index += 1
        result[doc_id] = prediction
    return result


def comparison_error(actual, expected, label, atol=1e-8):
    """Strict recursive metric comparison; report the first mismatch."""
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                raise AssertionError(label + " missing " + key)
            comparison_error(actual[key], value, label + "." + key, atol)
    elif isinstance(expected, (float, int)):
        if not math.isfinite(float(actual)) or abs(float(actual) - float(expected)) > atol:
            raise AssertionError(f"{label}: actual={actual!r}, reference={expected!r}")
    elif actual != expected:
        raise AssertionError(label + " differs")


def require_prediction_coverage(predictions, expected_document_ids, label):
    expected = set(expected_document_ids)
    if not expected or set(predictions) != expected:
        raise ValueError(label + " prediction coverage differs from expected documents")


def assert_cpu_equivalent(expected_document_ids, first_metrics, second_metrics,
                          first_predictions, second_predictions, label):
    """No score tolerance can substitute for exact complete CPU predictions."""
    require_prediction_coverage(first_predictions, expected_document_ids, label + " first")
    require_prediction_coverage(second_predictions, expected_document_ids, label + " second")
    if first_predictions != second_predictions:
        raise AssertionError(label + " CPU predictions differ, even if aggregate metrics agree")
    if first_metrics != second_metrics:
        comparison_error(second_metrics, first_metrics, label + " CPU metrics", atol=0.0)
        raise AssertionError(label + " CPU metrics/fields differ")
    return {"all_predictions_exact": True, "all_metrics_exact": True,
            "documents": len(set(expected_document_ids)), "metric_tolerance": 0.0}


def exact_metric_differences(actual, reference, path=""):
    """Report every unequal recorded number, including sub-ULP-scale rounding.

    Inputs and their original values are retained; no thresholds, score edits,
    rounding, or success assertions are applied to GPU-vs-CPU comparisons.
    """
    differences = []
    if isinstance(actual, dict) and isinstance(reference, dict):
        for key in sorted(set(actual) | set(reference)):
            child = path + "." + key if path else key
            if key not in actual or key not in reference:
                differences.append({"path": child, "gpu_reference": copy.deepcopy(reference.get(key)),
                                    "cpu_actual": copy.deepcopy(actual.get(key)), "kind": "missing_metric_field"})
            else:
                differences.extend(exact_metric_differences(actual[key], reference[key], child))
    elif actual != reference:
        entry = {"path": path, "gpu_reference": copy.deepcopy(reference), "cpu_actual": copy.deepcopy(actual)}
        if isinstance(actual, (int, float)) and isinstance(reference, (int, float)):
            entry["cpu_minus_gpu"] = actual - reference
        differences.append(entry)
    return differences


def prediction_field_differences(task, gpu_predictions, cpu_predictions, expected_document_ids):
    require_prediction_coverage(gpu_predictions, expected_document_ids, "GPU selected field")
    require_prediction_coverage(cpu_predictions, expected_document_ids, "CPU selected field")
    field = "coreference" if task == "COREFERENCE" else task.lower() + "_relations"
    documents, transitions = [], Counter()
    for doc_id in sorted(expected_document_ids):
        gpu_value, cpu_value = gpu_predictions[doc_id][field], cpu_predictions[doc_id][field]
        if task == "COREFERENCE":
            normalized_gpu = sorted(tuple(sorted(cluster)) for cluster in gpu_value)
            normalized_cpu = sorted(tuple(sorted(cluster)) for cluster in cpu_value)
            if normalized_gpu != normalized_cpu:
                old = {mention: tuple(cluster) for cluster in normalized_gpu for mention in cluster}
                new = {mention: tuple(cluster) for cluster in normalized_cpu for mention in cluster}
                documents.append({"document_id": doc_id, "gpu_clusters": normalized_gpu, "cpu_clusters": normalized_cpu,
                    "changed_mentions": [mention for mention in sorted(set(old) | set(new)) if old.get(mention) != new.get(mention)]})
            continue
        def pair_map(value):
            if task == "SUBEVENT":
                labeled = (("subevent", value),)
            else:
                labeled = value.items()
            result = {}
            for label, pairs in labeled:
                for pair in pairs:
                    key = tuple(pair)
                    if key in result:
                        raise ValueError("Duplicate/conflicting predicted relation pair")
                    result[key] = label
            return result
        old, new = pair_map(gpu_value), pair_map(cpu_value)
        changed = []
        for pair in sorted(set(old) | set(new)):
            before, after = old.get(pair, "NONE"), new.get(pair, "NONE")
            if before != after:
                changed.append({"pair": list(pair), "gpu_label": before, "cpu_label": after})
                transitions[before + " -> " + after] += 1
        if changed:
            documents.append({"document_id": doc_id, "differences": changed})
    return {"selected_task": task.lower(), "documents_compared": len(set(expected_document_ids)),
        "changed_document_count": len(documents),
        "changed_relation_pair_count": sum(len(doc.get("differences", [])) for doc in documents) if task != "COREFERENCE" else None,
        "relation_label_transitions": dict(transitions), "document_differences": documents}


def gpu_reference_report(task, gpu_augmented, gpu_selected_raw, cpu_metrics,
                         gpu_predictions, cpu_predictions, expected_document_ids, canary=False):
    selected_predictions = prediction_field_differences(task, gpu_predictions, cpu_predictions, expected_document_ids)
    return {"selected_task": task.lower(), "comparison_scope": "valid_subset" if canary else "valid",
        "original_gpu_augmented_protocol_fraction": copy.deepcopy(gpu_augmented),
        "original_gpu_selected_task_raw_protocol_percent": copy.deepcopy(gpu_selected_raw),
        "actual_cpu_augmented_protocol_fraction": copy.deepcopy(cpu_metrics["upstream_augmented_protocol_fraction"]),
        "actual_cpu_selected_task_raw_protocol_percent": copy.deepcopy(cpu_metrics["official_protocol"][task.lower()]),
        "metric_comparison_performed": not canary,
        "metric_comparison_omission_reason": "Canary CPU subset is not the original full GPU validation split" if canary else None,
        "augmented_exact_metric_differences": exact_metric_differences(cpu_metrics["upstream_augmented_protocol_fraction"], gpu_augmented) if not canary else None,
        "selected_raw_exact_metric_differences": exact_metric_differences(cpu_metrics["official_protocol"][task.lower()], gpu_selected_raw) if not canary else None,
        "selected_task_prediction_differences": selected_predictions,
        "gpu_logits_available": False, "scores_modified": False, "tolerances_relaxed": False,
        "interpretation": "GPU records and CPU actual results are separate evidence. CPU legacy/converted equivalence is verified independently; no GPU numerical margin is asserted."}


def semantic_compatibility(legacy, converted, dataset, indices, collate, records):
    checks = []
    legacy.eval()
    converted.eval()
    with TORCH.inference_mode():
        for index in indices:
            batch = collate([dataset[index]])
            before = legacy(batch)
            after = converted(**batch)
            old_coref, old_rel = unpack(before, True)
            new_coref, new_rel = unpack(after)
            differences = {}
            for task in RELATIONS:
                TORCH.testing.assert_close(old_rel[task], new_rel[task], rtol=1e-5, atol=1e-6)
                differences[task] = float((old_rel[task] - new_rel[task]).abs().max()) if old_rel[task].numel() else 0.0
            for left, right in zip(old_coref, new_coref):
                lower = TORCH.tril(TORCH.ones_like(right, dtype=TORCH.bool))
                # Legacy padding is -1000; new padding is -inf. Only true
                # antecedents/dummy are probabilities and should be compared.
                TORCH.testing.assert_close(left[lower], right.exp()[lower], rtol=1e-5, atol=1e-6)
                if not TORCH.equal(left.argmax(-1), right.argmax(-1)):
                    raise AssertionError("Coreference argmax changed during conversion")
            if decode_batch(before, batch, records, True) != decode_batch(after, batch, records):
                raise AssertionError("Decoded real-document predictions changed during conversion")
            checks.append({"document_id": dataset[index]["doc_id"], "chunks": int(batch["input_ids"].shape[0]),
                           "mentions": sum(map(len, dataset[index]["event_spans"])),
                           "relation_max_abs_error": differences, "decoded_predictions_exact": True})
    return {"passed": True, "probe_documents": checks,
            "semantic_notes": ["Identical legacy tokenizer, 256 padding and ordered pairs",
                "eval dropout disabled; eager encoder; FP32 CPU",
                "legacy coref softmax vs new float32 log_softmax; compare valid probabilities and exact argmax",
                "coref padding -1000 vs -inf excluded from probability comparison"]}


def classification_from_confusion(matrix, mapping):
    none = mapping["NONE"]
    rows = []
    report = {}
    for name, label in mapping.items():
        if label == none:
            continue
        tp = int(matrix[label, label])
        predicted, support = int(matrix[:, label].sum()), int(matrix[label, :].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (predicted + support) if predicted + support else 0.0
        report[name] = {"precision": precision, "recall": recall, "f1-score": f1, "support": float(support)}
        rows.append((tp, predicted, support, precision, recall, f1))
    tp, predicted, support = (sum(row[column] for row in rows) for column in range(3))
    report["micro avg"] = {"precision": tp / predicted if predicted else 0.0,
                           "recall": tp / support if support else 0.0,
                           "f1-score": 2 * tp / (predicted + support) if predicted + support else 0.0,
                           "support": float(support)}
    report["macro avg"] = {name: sum(row[column] for row in rows) / len(rows)
                           for name, column in (("precision", 3), ("recall", 4), ("f1-score", 5))}
    report["weighted avg"] = {name: sum(row[column] * row[2] for row in rows) / support if support else 0.0
                              for name, column in (("precision", 3), ("recall", 4), ("f1-score", 5))}
    report["macro avg"]["support"] = report["weighted avg"]["support"] = float(support)
    return report


def official_scores(gold, predictions, evaluator):
    if set(gold) != set(predictions):
        raise ValueError("Prediction coverage differs from validation gold")
    scores = {"coreference": evaluator.evaluate_coreference(gold, predictions)}
    for task in RELATIONS:
        scores[task] = evaluator.evaluate(gold, predictions, task)
    scores["coreference_mean_f1"] = sum(scores["coreference"][name + "_f1"]
                                        for name in ("muc", "b_cubed", "ceaf", "blanc")) / 4
    scores["overall_f1"] = (scores["coreference_mean_f1"] + sum(scores[task][task + "_f1"] for task in RELATIONS)) / 4
    return scores


def evaluate(model, dataset, collate, records, legacy_metrics, evaluator, args, label, legacy=False):
    model.eval()
    loader = TORCH.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    predictions, coref_documents = {}, []
    confusion = {task: TORCH.zeros((len(mapping), len(mapping)), dtype=TORCH.int64) for task, mapping in RELATIONS.items()}
    with TORCH.inference_mode():
        for batch_index, batch in enumerate(loader):
            output = model(batch) if legacy else model(**batch)
            corefs, relation_logits = unpack(output, legacy)
            decoded = decode_batch(output, batch, records, legacy)
            predictions.update(decoded)
            for document_index, clusters in enumerate(batch["coref_labels"]):
                predicted = canonical_clusters(corefs[document_index])
                def inverse(groups):
                    return {index: tuple(group) for group in groups for index in group}
                coref_documents.append(SimpleNamespace(gold=clusters, mention_to_gold=inverse(clusters),
                    clusters=predicted, mention_to_cluster=inverse(predicted)))
            for task, mapping in RELATIONS.items():
                labels = batch[task + "_labels"].reshape(-1)
                predicted = relation_logits[task].argmax(-1).reshape(-1)
                valid = labels >= 0
                counts = TORCH.bincount(labels[valid] * len(mapping) + predicted[valid], minlength=len(mapping) ** 2)
                confusion[task] += counts.reshape(len(mapping), len(mapping))
            if batch_index % 10 == 0 or batch_index + 1 == len(loader):
                update("evaluating", model=label, evaluated_documents=len(predictions), total_documents=len(dataset))
    augmented = {"COREFERENCE": {}}
    for name, metric in (("B-cubed", legacy_metrics.b_cubed), ("CEAF", legacy_metrics.ceafe),
                         ("MUC", legacy_metrics.muc), ("BLANC", legacy_metrics.blanc)):
        precision, recall, f1 = legacy_metrics.evaluate_documents(coref_documents, metric)
        augmented["COREFERENCE"][name] = {"precision": precision, "recall": recall, "f1": f1}
    for task, mapping in RELATIONS.items():
        augmented[task.upper()] = classification_from_confusion(confusion[task], mapping)
    return {"split": "valid_subset" if args.canary_only else "valid", "documents": len(dataset),
            "official_protocol": official_scores(records, predictions, evaluator),
            "upstream_augmented_protocol_fraction": augmented}, predictions


def save_predictions(path, predictions):
    temporary = Path(str(path) + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        for doc_id in sorted(predictions):
            stream.write(json.dumps(predictions[doc_id], ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return sha256_file(path)


def score_vector(scores):
    return {"muc_f1": scores["coreference"]["muc_f1"], "b_cubed_f1": scores["coreference"]["b_cubed_f1"],
            "ceaf_f1": scores["coreference"]["ceaf_f1"], "blanc_f1": scores["coreference"]["blanc_f1"],
            "coreference_mean_f1": scores["coreference_mean_f1"],
            **{task + "_f1": scores[task][task + "_f1"] for task in RELATIONS}, "overall_f1": scores["overall_f1"]}


def public_config(value):
    if isinstance(value, dict):
        omitted = {"_name_or_path", "name_or_path", "vocab_file", "merges_file", "tokenizer_file", "cache_dir", "_commit_hash"}
        return {key: public_config(item) for key, item in value.items() if key not in omitted}
    if isinstance(value, list):
        return [public_config(item) for item in value]
    if isinstance(value, str) and (value.startswith("/") or (len(value) > 2 and value[1:3] in (":\\", ":/"))):
        raise ValueError("An absolute path would leak into the public HF export")
    return value


def export_model(model, tokenizer, destination, provenance, license_path):
    model.config._name_or_path = ""
    model.encoder.config._name_or_path = ""
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    for name in ("config.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = destination / name
        if path.exists():
            atomic_json(path, public_config(read_json(path)))
    atomic_json(destination / "reference_provenance.json", public_config(provenance))
    shutil.copyfile(license_path, destination / "LICENSE")
    return {path.name: sha256_file(path) for path in destination.iterdir() if path.is_file()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", "--reference-dir")
    parser.add_argument("--source-root", help="Read-only source matching frozen 8a49171")
    parser.add_argument("--base-model")
    parser.add_argument("--data-dir")
    parser.add_argument("--baseline-metrics", help="Old single joint best_dev_metrics.json")
    parser.add_argument("--baseline-manifest", help="Old single run split_manifest.json for exact valid SHA")
    parser.add_argument("--output-dir", help="New empty postprocessing output")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8, help="Keep reference validation batch=8 by default")
    parser.add_argument("--expected-documents", type=int, default=710)
    parser.add_argument("--canary-only", action="store_true")
    args = parser.parse_args(argv)
    for name in ("reference_run", "source_root", "base_model", "data_dir", "baseline_metrics", "baseline_manifest", "output_dir"):
        if not getattr(args, name):
            parser.error("Assessment requires --" + name.replace("_", "-"))
        setattr(args, name, checked_path(getattr(args, name)))
    if args.threads < 1 or args.batch_size < 1 or args.expected_documents < 1:
        parser.error("threads, batch-size and expected-documents must be positive")
    return args


def main(argv=None):
    global ACTIVE_OUTPUT, TORCH
    args = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output must be new or empty; completed runs are never modified")
    # Refuse writing into either an old run or the immutable training checkout.
    for forbidden in (args.reference_run, args.source_root, args.base_model, args.data_dir):
        if args.output_dir == forbidden or forbidden in args.output_dir.parents:
            raise ValueError("Postprocessing output cannot be inside any read-only input")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ACTIVE_OUTPUT = args.output_dir
    for directory in ("private", "per_model", "four_checkpoint_combination"):
        (args.output_dir / directory).mkdir()
    # Set before importing torch/Transformers: this process never sees the GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(args.threads)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(args.output_dir / "private/hf_cache"),
        "HF_MODULES_CACHE": str(args.output_dir / "private/hf_modules"),
        "PYTHONDONTWRITEBYTECODE": "1"})
    sys.dont_write_bytecode = True
    update("checking_inputs", status="running", cpu_threads=args.threads, gpu_visible=False,
           canary_only=args.canary_only, assessment_version=ASSESSMENT_VERSION)
    summary = read_json(args.reference_run / "reference_summary.json")
    source_manifest = read_json(args.reference_run / "source_manifest.json")
    if summary["status"] != "completed" or not summary.get("experiment_complete") or source_manifest["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("Input is not a completed formal upstream reference run")
    if summary["training"]["epochs_completed"] != 100:
        raise ValueError("Reference did not complete 100 epochs")
    dataset_manifest = read_json(args.reference_run / "dataset_manifest.json")
    valid_sha = verify_hash(args.data_dir / "valid.jsonl", dataset_manifest["valid"]["sha256"], "Reference valid split")
    baseline_manifest = read_json(args.baseline_manifest)
    if baseline_manifest["valid"]["sha256"] != valid_sha:
        raise ValueError("Old single baseline validation split differs")
    baseline_payload = read_json(args.baseline_metrics)
    if baseline_payload.get("split", "valid") != "valid":
        raise ValueError("Baseline is not a full validation result")
    baseline = baseline_payload.get("official_protocol", baseline_payload)
    baseline_vector = score_vector(baseline)
    all_records = read_valid(args.data_dir / "valid.jsonl")
    if len(all_records) != args.expected_documents or len(all_records) != dataset_manifest["valid"]["documents"]:
        raise ValueError("Validation document count differs from expected full split")
    source_augmented = read_json(args.reference_run / "validation_by_checkpoint.json")
    source_raw = read_json(args.reference_run / "raw_direction_validation_by_checkpoint.json")
    selection = read_json(args.reference_run / "best_selection.json")
    gpu_reference_predictions = read_valid_predictions(args.reference_run / "best_valid_predictions.jsonl")
    require_prediction_coverage(gpu_reference_predictions, all_records, "Original GPU reference")
    atomic_json(args.output_dir / "private/input_manifest.json", {"args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "reference_summary_sha256": sha256_file(args.reference_run / "reference_summary.json"),
        "source_manifest_sha256": sha256_file(args.reference_run / "source_manifest.json"),
        "baseline_metrics_sha256": sha256_file(args.baseline_metrics), "baseline_manifest_sha256": sha256_file(args.baseline_manifest),
        "script_sha256": sha256_file(Path(__file__).resolve()), "frozen_source_commit": FROZEN_COMMIT,
        "assessment_version": ASSESSMENT_VERSION,
        "original_gpu_prediction_sha256": sha256_file(args.reference_run / "best_valid_predictions.jsonl"),
        "frozen_model_source_sha256": FROZEN_HASHES, "valid_sha256": valid_sha,
        "reference_metadata_sha256": {name: sha256_file(args.reference_run / name) for name in (
            "reference_summary.json", "source_manifest.json", "dataset_manifest.json", "best_selection.json",
            "best_valid_metrics.json", "validation_by_checkpoint.json", "raw_direction_validation_by_checkpoint.json")},
        "baseline_reported_official_protocol": baseline})
    import torch
    TORCH = torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only process unexpectedly sees CUDA")
    from transformers import AutoModel, RobertaTokenizer
    Config, Model, legacy_model_module, legacy_data, legacy_metrics, evaluator, legacy_source = snapshot_code(args, args.output_dir, source_manifest)
    tokenizer = RobertaTokenizer.from_pretrained(args.base_model, local_files_only=True)
    update("tokenizing_original_pipeline")
    dataset = legacy_data.myDataset(tokenizer, str(args.data_dir), "valid", max_length=256)
    if len(dataset) != args.expected_documents or set(item["doc_id"] for item in dataset.tokenized_samples) != set(all_records):
        raise ValueError("Original dataset processing omitted or changed validation documents")
    probes = semantic_probe_indices(dataset)
    atomic_json(args.output_dir / "private/semantic_probe_documents.json", [{"index": index, "document_id": dataset[index]["doc_id"],
        "chunks": int(dataset[index]["input_ids"].shape[0]), "mentions": sum(map(len, dataset[index]["event_spans"]))} for index in probes])
    if args.canary_only:
        evaluation_dataset = torch.utils.data.Subset(dataset, probes)
        records = OrderedDict((dataset[index]["doc_id"], all_records[dataset[index]["doc_id"]]) for index in probes)
    else:
        evaluation_dataset, records = dataset, all_records
    legacy = legacy_model_module.Model(len(tokenizer), model_name=str(args.base_model)).float().cpu().eval()
    encoder_config = public_config(legacy.encoder.model.module.config.to_dict())
    results = {}
    for task in TASKS:
        update("loading_complete_checkpoint", selected_task=task)
        checkpoint = checked_path(args.reference_run / "artifacts" / ("best_" + task))
        checkpoint_sha = verify_hash(checkpoint, summary["task_checkpoints"][task.lower()]["sha256"], task + " checkpoint")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("canary_fallback"):
            raise ValueError("Diagnostic fallback checkpoint cannot be assessed as a full reference model")
        weights = payload["model"]
        check_finite_weights(weights, task)
        legacy.load_state_dict(weights, strict=True)
        config = Config(encoder_config=encoder_config, encoder_attention_implementation="eager",
            scorer_hidden_size=200, scorer_num_layers=3, span_pooling="mean", dropout=0.2, chunk_length=256)
        converted = Model(config).float().cpu().eval()
        mapping = strict_map_state(weights, converted)
        del payload, weights
        gc.collect()
        model_output = args.output_dir / "per_model" / task.lower()
        model_output.mkdir()
        update("checking_real_document_compatibility", selected_task=task)
        compatibility = semantic_compatibility(legacy, converted, dataset, probes, legacy_data.collator, all_records)
        atomic_json(model_output / "compatibility.json", compatibility)
        identity = selection[task]
        provenance = {"upstream_commit": UPSTREAM_COMMIT, "frozen_export_code_commit": FROZEN_COMMIT,
            "original_checkpoint_sha256": checkpoint_sha, "selected_for_task": task.lower(),
            "reference_seed": source_manifest["args"]["seed"],
            "selection": {key: identity[key] for key in ("micro_step", "optimizer_updates", "epoch_index", "selection") if key in identity},
            "state_mapping": mapping, "complete_model": True, "checkpoint_count": 1,
            "architecture": {"scorer_hidden_size": 200, "scorer_num_layers": 3, "chunk_length": 256,
                             "dtype": "float32", "span_pooling": "mean", "encoder_attention_implementation": "eager"}}
        atomic_json(model_output / "provenance.json", {**provenance, "private_original_checkpoint_path": str(checkpoint)})
        legacy_cpu_metrics, legacy_cpu_predictions = evaluate(legacy, evaluation_dataset, legacy_data.collator, records,
            legacy_metrics, evaluator, args, task.lower() + "/original_legacy_cpu", legacy=True)
        atomic_json(model_output / "legacy_cpu_metrics.json", legacy_cpu_metrics)
        legacy_prediction_sha = save_predictions(model_output / "legacy_cpu_valid_predictions.jsonl", legacy_cpu_predictions)
        metrics, predictions = evaluate(converted, evaluation_dataset, legacy_data.collator, records,
            legacy_metrics, evaluator, args, task.lower() + "/converted")
        atomic_json(model_output / "metrics.json", metrics)
        prediction_sha = save_predictions(model_output / "valid_predictions.jsonl", predictions)
        cpu_equivalence = assert_cpu_equivalent(records, legacy_cpu_metrics, metrics,
            legacy_cpu_predictions, predictions, task + " original legacy vs converted")
        if legacy_prediction_sha != prediction_sha:
            raise AssertionError("Serialized legacy/converted CPU full predictions changed")
        cpu_equivalence["prediction_sha256"] = prediction_sha
        atomic_json(model_output / "cpu_legacy_vs_converted.json", cpu_equivalence)
        scoped_gpu_predictions = {doc_id: gpu_reference_predictions[doc_id] for doc_id in records}
        gpu_comparison = gpu_reference_report(task, source_augmented[task], source_raw[task], metrics,
            scoped_gpu_predictions, predictions, records, canary=args.canary_only)
        atomic_json(model_output / "gpu_reference_comparison.json", gpu_comparison)
        export_path = model_output / "hf_model"
        exported_files = export_model(converted, tokenizer, export_path, provenance, legacy_source / "LICENSE")
        update("reloading_hf_automodel", selected_task=task)
        reloaded = AutoModel.from_pretrained(export_path, trust_remote_code=True, local_files_only=True).float().cpu().eval()
        first_state, second_state = converted.state_dict(), reloaded.state_dict()
        if set(first_state) != set(second_state) or any(not torch.equal(first_state[key], second_state[key]) for key in first_state):
            raise AssertionError("HF AutoModel reload changed state_dict")
        reload_metrics, reload_predictions = evaluate(reloaded, evaluation_dataset, legacy_data.collator, records,
            legacy_metrics, evaluator, args, task.lower() + "/automodel_reload")
        assert_cpu_equivalent(records, metrics, reload_metrics, predictions, reload_predictions,
                              task + " converted vs HF AutoModel reload")
        reload_sha = save_predictions(model_output / "automodel_valid_predictions.jsonl", reload_predictions)
        if reload_sha != prediction_sha:
            raise AssertionError("Serialized AutoModel full predictions changed")
        roundtrip = {"passed": True, "documents": len(evaluation_dataset), "all_state_tensors_exact": True,
            "all_predictions_exact": True, "all_metrics_exact": True, "prediction_sha256": prediction_sha,
            "exported_files_sha256": exported_files, "loader": "AutoModel.from_pretrained(trust_remote_code=True, local_files_only=True)"}
        atomic_json(model_output / "automodel_roundtrip.json", roundtrip)
        vector = score_vector(metrics["official_protocol"])
        deltas = {key: vector[key] - baseline_vector[key] for key in vector} if not args.canary_only else None
        results[task.lower()] = {"checkpoint_sha256": checkpoint_sha, "checkpoint_count": 1,
            "selected_for_task": task.lower(), "official_protocol": metrics["official_protocol"],
            "delta_vs_old_single_percentage_points": deltas, "compatibility_passed": True,
            "cpu_legacy_vs_converted": cpu_equivalence, "gpu_reference_comparison": gpu_comparison,
            "hf_automodel_roundtrip_passed": True,
            "hf_export": str(export_path), "validation_predictions_sha256": prediction_sha}
        atomic_json(model_output / "assessment.json", results[task.lower()])
        atomic_json(args.output_dir / "single_checkpoint_results.json", results)
        del converted, reloaded, first_state, second_state, predictions, reload_predictions, legacy_cpu_predictions
        gc.collect()
    # Preserve the historical reference and, in full mode, independently rebuild
    # the combination from actual reloaded AutoModel predictions. No parameters
    # or encoder/head fragments are assembled into a new mixed model.
    composite_file = args.reference_run / "best_valid_predictions.jsonl"
    composite_predictions = read_valid_predictions(composite_file)
    original_combo_scores = official_scores(all_records, composite_predictions, evaluator)
    comparison_error(original_combo_scores, summary["official_protocol"], "Original four-checkpoint composite")
    combo_output = args.output_dir / "four_checkpoint_combination"
    shutil.copyfile(composite_file, combo_output / "original_reference_valid_predictions.jsonl")
    original_combo_sha = sha256_file(composite_file)
    auto_prediction_sources = {}
    combo_gpu_comparison = {"comparison_performed": False,
        "reason": "Canary does not regenerate the full 710-document CPU combination",
        "original_gpu_official_protocol": original_combo_scores}
    if args.canary_only:
        # These 710 documents come from the old completed run, not the 12-document
        # HF canary. Keep that distinction explicit in every combination report.
        shutil.copyfile(composite_file, combo_output / "valid_predictions.jsonl")
        combo_scores = original_combo_scores
        combined_prediction_sha = original_combo_sha
        combination_source = "completed_reference_predictions_only"
    else:
        combined_from_exports = {doc_id: {"id": doc_id} for doc_id in all_records}
        for task in TASKS:
            task_name = task.lower()
            source = args.output_dir / "per_model" / task_name / "automodel_valid_predictions.jsonl"
            source_sha = verify_hash(source, results[task_name]["validation_predictions_sha256"],
                                     task + " actual AutoModel predictions")
            task_predictions = read_valid_predictions(source)
            if set(task_predictions) != set(all_records):
                raise ValueError(task + " AutoModel prediction coverage is not the full validation split")
            field = "coreference" if task == "COREFERENCE" else task_name + "_relations"
            for doc_id in all_records:
                if field not in task_predictions[doc_id]:
                    raise ValueError(task + " AutoModel prediction field missing for " + doc_id)
                combined_from_exports[doc_id][field] = task_predictions[doc_id][field]
            auto_prediction_sources[task_name] = {"filename": str(source.relative_to(args.output_dir)),
                                                  "sha256": source_sha, "field": field}
        update("scoring_actual_automodel_combination", evaluated_documents=len(combined_from_exports))
        combo_scores = official_scores(all_records, combined_from_exports, evaluator)
        combo_gpu_comparison = {"comparison_performed": True,
            "original_gpu_official_protocol": copy.deepcopy(original_combo_scores),
            "original_gpu_summary_official_protocol": copy.deepcopy(summary["official_protocol"]),
            "actual_cpu_automodel_official_protocol": copy.deepcopy(combo_scores),
            "exact_metric_differences": exact_metric_differences(combo_scores, original_combo_scores),
            "selected_field_prediction_differences": {task.lower(): prediction_field_differences(
                task, composite_predictions, combined_from_exports, all_records) for task in TASKS},
            "gpu_logits_available": False, "scores_modified": False, "tolerances_relaxed": False,
            "interpretation": "Actual CPU exported-model predictions and original GPU predictions are reported separately; legacy/converted/AutoModel CPU equivalence is strict."}
        combined_prediction_sha = save_predictions(combo_output / "valid_predictions.jsonl", combined_from_exports)
        combination_source = "four_actual_automodel_full_validation_prediction_files"
    atomic_json(combo_output / "gpu_reference_comparison.json", combo_gpu_comparison)
    combination = {"checkpoint_count": 4, "composition": "prediction fields from four complete task-selected models",
        "complete_model_count": 4, "document_forward_passes": 4, "shared_encoder_checkpoint": False,
        "source": combination_source, "regenerated_from_exported_automodels": not args.canary_only,
        "documents": len(all_records), "combined_prediction_sha256": combined_prediction_sha,
        "automodel_prediction_sources": auto_prediction_sources,
        "original_reference_prediction_sha256": original_combo_sha,
        "original_reference_official_protocol": original_combo_scores,
        "gpu_reference_comparison": combo_gpu_comparison,
        "official_protocol": combo_scores, "task_checkpoint_sha256": {key: value["checkpoint_sha256"] for key, value in results.items()},
        "single_model": False, "test_scored": False}
    atomic_json(combo_output / "metrics.json", combination)
    best_key = max(results, key=lambda key: results[key]["official_protocol"]["overall_f1"])
    best = results[best_key]
    improved = bool(best["official_protocol"]["overall_f1"] > baseline["overall_f1"] + 1e-9) if not args.canary_only else None
    final = {"status": "smoke" if args.canary_only else "completed", "split": "valid_subset" if args.canary_only else "valid",
        "documents": len(evaluation_dataset), "valid_sha256": valid_sha, "cpu_only": True, "threads": args.threads,
        "source_commit": FROZEN_COMMIT, "single_checkpoint_results": results,
        "best_complete_single_checkpoint": best_key, "single_model_observed_validation_improvement": improved,
        "baseline_single_reported_official_protocol": baseline,
        "baseline_comparison_note": "Baseline is its recorded validation result; candidate conversion/reload use FP32 CPU. No test or multi-seed improvement is claimed.",
        "four_checkpoint_combination": combination, "test_scored": False, "published": False,
        "cpu_equivalence_passed": all(value["cpu_legacy_vs_converted"]["all_predictions_exact"]
            and value["cpu_legacy_vs_converted"]["all_metrics_exact"]
            and value["hf_automodel_roundtrip_passed"] for value in results.values()) and len(results) == 4,
        "validation_protocol": {"version": ASSESSMENT_VERSION,
            "inputs": "unchanged original tokenizer, max_length=256, original legacy collator and validation ordering",
            "cpu_legacy_vs_converted": "strict complete predictions and all metrics; zero metric tolerance",
            "hf_automodel_reload": "strict state tensors, complete predictions and all metrics",
            "gpu_vs_cpu": "exact differences and both original/actual scores; no equality claim or tolerance relaxation",
            "gpu_logits_available": False, "precision": "float32", "device": "cpu",
            "documents_per_cpu_model": len(evaluation_dataset)},
        "training_campaign_modified": False, "training_campaign_completion_assessed": False,
        "elapsed_seconds": time.monotonic() - START}
    atomic_json(args.output_dir / "summary.json", final)
    update("complete", status=final["status"], best_complete_single_checkpoint=best_key,
           single_model_observed_validation_improvement=improved)
    return 0


def read_valid_predictions(path):
    result = {}
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                if record["id"] in result:
                    raise ValueError("Duplicate document in saved reference predictions")
                result[record["id"]] = record
    return result


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        if ACTIVE_OUTPUT is not None:
            failure = {"status": "failed", "stage": PROGRESS.get("stage"), "error_type": type(error).__name__,
                "error": str(error), "test_scored": False, "published": False,
                "elapsed_seconds": time.monotonic() - START}
            atomic_json(ACTIVE_OUTPUT / "summary.json", failure)
            update("failed", **{key: value for key, value in failure.items() if key != "stage"})
        traceback.print_exc()
        raise
