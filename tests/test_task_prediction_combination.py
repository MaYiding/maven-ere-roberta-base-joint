import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "combine_task_predictions", Path(__file__).resolve().parents[1] / "scripts/combine_task_predictions.py"
)
combiner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(combiner)


def make_sources(tmp_path):
    sources = {}
    for index, field in enumerate(combiner.FIELDS):
        value = {"selected": [[f"e{index}", "e9"]]} if "relations" in field and field != "subevent_relations" else [[f"e{index}", "e9"]]
        path = tmp_path / (field + ".jsonl")
        path.write_text(json.dumps({"id": "document", field: value, "private_extra": "omit"}) + "\n")
        sources[field] = path
    return sources


def test_selects_only_the_requested_task_from_each_source(tmp_path):
    sources = make_sources(tmp_path)
    records = combiner.combine_predictions(sources, expected_documents=1)
    assert set(records[0]) == {"id", *combiner.FIELDS}
    for field in combiner.FIELDS:
        assert records[0][field] == json.loads(sources[field].read_text())[field]


def test_rejects_mismatched_or_duplicate_document_coverage(tmp_path):
    sources = make_sources(tmp_path)
    temporal = sources["temporal_relations"]
    original = temporal.read_text()
    temporal.write_text(original.replace('"document"', '"other"'))
    with pytest.raises(ValueError, match="Document IDs differ"):
        combiner.combine_predictions(sources)
    temporal.write_text(original + original)
    with pytest.raises(ValueError, match="Duplicate document ID"):
        combiner.combine_predictions(sources)
