"""Reference inference shape contracts; run only in the remote test environment."""
import copy
from types import SimpleNamespace

import pytest
import torch

from joint.src.data import collator
from test_training_contracts import fixture_dataset, gold_document
from train_joint import evaluate_dev, make_loader, parse_args, preserve_padding_collator


def test_reference_collator_restores_original_chunks_without_changing_other_inputs(tmp_path):
    dataset = fixture_dataset(tmp_path, max_length=32)
    samples = [dataset[0], copy.deepcopy(dataset[0])]
    samples[1]["doc_id"] = "second"
    before = copy.deepcopy(samples)
    trimmed = collator(samples)
    fixed = preserve_padding_collator(samples)
    assert trimmed["input_ids"].shape[1] < 32
    torch.testing.assert_close(fixed["input_ids"], torch.cat([x["input_ids"] for x in samples]), rtol=0, atol=0)
    torch.testing.assert_close(fixed["attention_mask"], torch.cat([x["attention_mask"] for x in samples]), rtol=0, atol=0)
    assert not fixed["attention_mask"][:, trimmed["input_ids"].shape[1]:].any()
    for key in fixed:
        if key in {"input_ids", "attention_mask"}:
            continue
        if torch.is_tensor(fixed[key]):
            torch.testing.assert_close(fixed[key], trimmed[key], rtol=0, atol=0)
        else:
            assert fixed[key] == trimmed[key]
    for left, right in zip(samples, before):
        for key in left:
            if torch.is_tensor(left[key]):
                torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
            else:
                assert left[key] == right[key]


def test_loader_default_remains_trimmed_and_reference_mode_keeps_full_shape(tmp_path):
    dataset = fixture_dataset(tmp_path, max_length=32)
    default = next(iter(make_loader(dataset, [0], 1, 0, 42)))
    reference = next(iter(make_loader(dataset, [0], 1, 0, 42, preserve_padding=True)))
    assert default["input_ids"].shape[1] < 32
    assert reference["input_ids"].shape[1] == 32
    assert default["doc_id"] == reference["doc_id"]


def test_reference_flag_is_prediction_only(tmp_path):
    common = ["--data-dir", str(tmp_path)]
    with pytest.raises(SystemExit):
        parse_args(common + ["--base-model", "base", "--output-dir", "run", "--reference-padding"])
    args = parse_args(common + ["--predict-only", "--checkpoint", "model",
                               "--prediction-file", str(tmp_path / "predictions.jsonl"), "--reference-padding"])
    assert args.predict_only and args.reference_padding
    training = parse_args(common + ["--base-model", "base", "--output-dir", "run"])
    assert not training.reference_padding


def test_reference_predictions_order_clusters_without_changing_membership_or_defaults(tmp_path, monkeypatch):
    dataset = fixture_dataset(tmp_path, max_length=32)
    monkeypatch.setattr('train_joint.get_predicted_clusters', lambda _: ([[3], [0, 2], [1]], {}))

    class Scores:
        def eval(self):
            return self

        def __call__(self, **batch):
            pairs = batch['temporal_labels'].shape[-1]
            temporal = torch.zeros((1, pairs, 7)); temporal[:, :, 6] = 1
            causal = torch.zeros((1, pairs, 3)); causal[:, :, 0] = 1
            subevent = torch.zeros((1, pairs, 2)); subevent[:, :, 0] = 1
            return SimpleNamespace(coref_log_probs=[torch.eye(4)], temporal_logits=temporal,
                                   causal_logits=causal, subevent_logits=subevent)

    args = SimpleNamespace(eval_batch_size=1, batch_size=1, num_workers=0, seed=42,
                           bf16=False, reference_padding=True)
    gold = {'fixture-train': gold_document()}
    _, reference = evaluate_dev(Scores(), dataset, gold, args, torch.device('cpu'), score=False)
    assert reference['fixture-train']['coreference'] == [['m_a1', 'm_a2'], ['m_b'], ['m_c']]
    args.reference_padding = False
    _, normal = evaluate_dev(Scores(), dataset, gold, args, torch.device('cpu'), score=False)
    assert normal['fixture-train']['coreference'] == [['m_c'], ['m_a1', 'm_a2'], ['m_b']]
    for field in ['temporal_relations', 'causal_relations', 'subevent_relations']:
        assert reference['fixture-train'][field] == normal['fixture-train'][field]
