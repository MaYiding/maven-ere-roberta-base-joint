"""Combine task predictions from four complete models without mixing weights.

Generate each input with train_joint.py --predict-only. Each input must cover
exactly the same document IDs. The result is a multiple-checkpoint combination,
not the predictions of one shared-encoder checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid


FIELDS = ("coreference", "temporal_relations", "causal_relations", "subevent_relations")


def read_predictions(path, field):
    records = {}
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            doc_id = record.get("id")
            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError(f"Invalid document ID at {path}:{number}")
            if doc_id in records:
                raise ValueError(f"Duplicate document ID {doc_id} in {path}")
            expected = dict if field in {"temporal_relations", "causal_relations"} else list
            if not isinstance(record.get(field), expected):
                raise ValueError(f"Missing or invalid {field} at {path}:{number}")
            records[doc_id] = record[field]
    if not records:
        raise ValueError(f"Empty prediction file: {path}")
    return records


def combine_predictions(sources, expected_documents=None):
    if set(sources) != set(FIELDS):
        raise ValueError("Provide one prediction source for every relation task")
    predictions = {field: read_predictions(sources[field], field) for field in FIELDS}
    doc_ids = set(predictions[FIELDS[0]])
    for field in FIELDS[1:]:
        if set(predictions[field]) != doc_ids:
            raise ValueError(f"Document IDs differ for {field}")
    if expected_documents is not None and len(doc_ids) != expected_documents:
        raise ValueError(f"Expected {expected_documents} documents, received {len(doc_ids)}")
    return [{"id": doc_id, **{field: predictions[field][doc_id] for field in FIELDS}}
            for doc_id in sorted(doc_ids)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in FIELDS:
        parser.add_argument("--" + field.replace("_", "-") + "-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-documents", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.expected_documents is not None and args.expected_documents < 1:
        parser.error("--expected-documents must be positive")
    sources = {field: getattr(args, field + "_predictions") for field in FIELDS}
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        parser.error("Output already exists; select a new path or explicitly use --overwrite")
    if output in {path.expanduser().resolve() for path in sources.values()}:
        parser.error("Output must not replace an input prediction file")
    records = combine_predictions(sources, args.expected_documents)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if output.exists() and not args.overwrite:
            raise FileExistsError(output)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"documents": len(records), "output": str(output),
                      "artifact_type": "multiple_checkpoint_task_combination", "is_single_model": False}))


if __name__ == "__main__":
    main()
