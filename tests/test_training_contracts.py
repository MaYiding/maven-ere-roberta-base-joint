"""Training contracts; execute in the authorized remote training environment.

Fixtures use the published MAVEN-ERE JSON schema, including TIMEX, a repeated
event, event-level gold edges, and [parent, child] subevent direction. No network
access or downloaded pretrained weights are needed by these tests.
"""

import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import RobertaConfig

from joint.src.data import (
    CAUSALREL2ID, SUBEVENTREL2ID, TEMPREL2ID, Document, collator, myDataset,
)
from joint.src.utils import fill_expand, get_predicted_clusters
from maven_ere import MavenEreConfig, MavenEreModel
from train_joint import (
    atomic_torch_save, build_optimizers, checkpoint_state, coreference_loss,
    joint_loss, load_checkpoint, relation_loss, restore_training_state,
    set_seed, validate_splits,
    load_official_evaluator,
)


def gold_document(doc_id="fixture-train"):
    return {
        "id": doc_id,
        "title": "Storm fixture",
        "tokens": [["Storm", "struck", "and", "flooded", "town", "on", "Monday", "."],
                   ["The", "storm", "destroyed", "homes", "."]],
        "sentences": ["Storm struck and flooded town on Monday.", "The storm destroyed homes."],
        "events": [
            {"id": "EVENT_A", "type": "NaturalDisaster", "type_id": 1,
             "mention": [{"id": "m_a1", "trigger_word": "struck", "sent_id": 0, "offset": [1, 2]},
                         {"id": "m_a2", "trigger_word": "storm", "sent_id": 1, "offset": [1, 2]}]},
            {"id": "EVENT_B", "type": "Damaging", "type_id": 2,
             "mention": [{"id": "m_b", "trigger_word": "flooded", "sent_id": 0, "offset": [3, 4]}]},
            {"id": "EVENT_C", "type": "Destroying", "type_id": 3,
             "mention": [{"id": "m_c", "trigger_word": "destroyed", "sent_id": 1, "offset": [2, 3]}]},
        ],
        "TIMEX": [{"id": "TIME_MON", "mention": "Monday", "type": "DATE", "sent_id": 0, "offset": [6, 7]}],
        "temporal_relations": {"BEFORE": [["EVENT_A", "EVENT_B"]],
                               "SIMULTANEOUS": [["EVENT_B", "EVENT_C"]]},
        "causal_relations": {"CAUSE": [["EVENT_A", "EVENT_B"]],
                             "PRECONDITION": [["EVENT_B", "EVENT_C"]]},
        "subevent_relations": [["EVENT_A", "EVENT_B"]],
    }


class FixtureTokenizer:
    cls_token_id = 0
    pad_token_id = 1
    sep_token_id = 2

    def __call__(self, words, **kwargs):
        # One subtoken per supplied word is sufficient to test alignment,
        # chunking, order, and padding without a tokenizer download.
        return {"input_ids": [3 + sum(word.encode("utf-8")) % 50 for word in words]}


def fixture_dataset(tmp_path, split="train", max_length=32):
    (tmp_path / f"{split}.jsonl").write_text(json.dumps(gold_document()) + "\n", encoding="utf-8")
    return myDataset(FixtureTokenizer(), str(tmp_path), split, max_length=max_length)


def tiny_model():
    encoder = RobertaConfig(vocab_size=64, hidden_size=16, intermediate_size=24,
                            num_hidden_layers=1, num_attention_heads=2,
                            max_position_embeddings=66, hidden_dropout_prob=0.1,
                            attention_probs_dropout_prob=0.1)
    return MavenEreModel(MavenEreConfig(encoder_config=encoder.to_dict(), scorer_hidden_size=12))


def test_every_cluster_first_mention_has_dummy_and_no_future_gold():
    mask = fill_expand([[0, 2, 4], [1, 3], [5]]).bool()
    assert mask[0, 0] and mask[1, 1] and mask[5, 5]
    assert not mask[2, 2] and not mask[3, 3]
    assert mask[4, 0] and mask[4, 2]
    assert mask.any(dim=1).all()
    assert not torch.triu(mask, diagonal=1).any()
    assert fill_expand([]).shape == (0, 0)
    with pytest.raises(ValueError, match="partition"):
        fill_expand([[0], [0]])


def test_coref_loss_is_marginal_mean_over_mentions_then_documents():
    logits = torch.tensor([[0., -torch.inf, -torch.inf],
                           [1., 0., -torch.inf], [2., 1., 0.]], requires_grad=True)
    log_probs = torch.log_softmax(logits, dim=-1)
    actual = coreference_loss([log_probs], [[[0, 2], [1]]])
    expected = -(log_probs[0, 0] + log_probs[1, 1] + log_probs[2, 0]) / 3
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[1, 1] != 0  # A new singleton cluster learns its dummy.


def test_schema_direction_timex_mask_and_no_input_mutation():
    raw = gold_document()
    untouched = copy.deepcopy(raw)
    doc = Document(raw)
    assert raw == untouched
    assert doc.events_idx == [0, 1, 3, 4]
    ids = [event["id"] for event in doc.events]
    pairs = [(left, right) for left in ids for right in ids if left != right]
    lookup = {pair: index for index, pair in enumerate(pairs)}
    assert doc.subevent_labels[lookup[("m_a1", "m_b")]] == SUBEVENTREL2ID["subevent"]
    assert doc.subevent_labels[lookup[("m_b", "m_a1")]] == SUBEVENTREL2ID["NONE"]
    assert doc.causal_labels[lookup[("m_a2", "m_b")]] == CAUSALREL2ID["CAUSE"]
    assert doc.temporal_labels[lookup[("m_b", "m_c")]] == TEMPREL2ID["SIMULTANEOUS"]
    assert doc.temporal_labels[lookup[("m_c", "m_b")]] == TEMPREL2ID["SIMULTANEOUS"]
    for pair, index in lookup.items():
        if "TIME_MON" in pair:
            assert doc.causal_labels[index] == -100
            assert doc.subevent_labels[index] == -100
            assert doc.temporal_labels[index] >= 0


def test_padding_trim_preserves_event_spans_and_positive_joint_losses(tmp_path):
    dataset = fixture_dataset(tmp_path)
    batch = collator([dataset[0]])
    assert batch["input_ids"].shape[1] < 32
    assert batch["event_spans"] == [dataset[0]["event_spans"]]
    assert sum(map(len, batch["event_spans"][0])) == 5
    model = tiny_model()
    output = model(**batch)
    loss, parts = joint_loss(output, batch)
    assert all(torch.isfinite(value) and value >= 0 for value in parts.values())
    for task, none in (("temporal", 6), ("causal", 0), ("subevent", 0)):
        labels = batch[task + "_labels"]
        assert ((labels >= 0) & (labels != none)).any()
    loss.backward()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_all_ignored_relation_labels_have_finite_differentiable_zero():
    logits = torch.randn(2, 3, 2, requires_grad=True)
    loss = relation_loss(logits, torch.full((2, 3), -100))
    assert loss.item() == 0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_split_overlap_rejected_without_scoring_test(tmp_path):
    for split in ("train", "valid", "test"):
        record = gold_document(split)
        if split == "test":
            record["event_mentions"] = [mention for event in record.pop("events") for mention in event["mention"]]
            for key in ("temporal_relations", "causal_relations", "subevent_relations"):
                del record[key]
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert validate_splits(tmp_path)["test"]["documents"] == 1
    (tmp_path / "valid.jsonl").write_text(json.dumps(gold_document("train")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(tmp_path)


def test_full_resume_matches_next_dropout_update_and_restores_both_optimizers(tmp_path):
    set_seed(17)
    batch = collator([fixture_dataset(tmp_path)[0]])
    model = tiny_model()
    args = SimpleNamespace(encoder_lr=2e-5, head_lr=3e-4, warmup_updates=1)
    encoder_opt, head_opt, scheduler = build_optimizers(model, args, total_updates=4)

    def update(target, encoder, heads, schedule):
        target.train()
        joint_loss(target(**batch), batch)[0].backward()
        encoder.step()
        heads.step()
        schedule.step()
        encoder.zero_grad(set_to_none=True)
        heads.zero_grad(set_to_none=True)

    update(model, encoder_opt, head_opt, scheduler)
    progress = {"epoch": 0, "order": [1, 0, 2], "next_batch": 1,
                "update_step": 1, "micro_step": 1}
    path = tmp_path / "latest.pt"
    atomic_torch_save(path, checkpoint_state(model, encoder_opt, head_opt, scheduler, progress, {"fixture": True}))
    saved = load_checkpoint(path)
    assert saved["encoder_optimizer"]["state"] and saved["head_optimizer"]["state"]
    random_value, numpy_value = random.random(), np.random.random()
    update(model, encoder_opt, head_opt, scheduler)
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    expected_lr = encoder_opt.param_groups[0]["lr"]

    resumed = tiny_model()
    re_encoder, re_heads, re_schedule = build_optimizers(resumed, args, total_updates=4)
    restored = restore_training_state(saved, resumed, re_encoder, re_heads, re_schedule)
    assert restored == progress
    assert random.random() == random_value and np.random.random() == numpy_value
    update(resumed, re_encoder, re_heads, re_schedule)
    assert re_schedule.last_epoch == scheduler.last_epoch == 2
    assert re_encoder.param_groups[0]["lr"] == expected_lr
    for key, value in resumed.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


def test_timex_only_document_decodes_empty_coreference_and_finite_zero_loss(tmp_path):
    # 空事件文档仍可能有时间表达；推理必须返回空共指簇而非 argmax 异常。
    assert get_predicted_clusters(torch.empty(0, 0)) == ([], {})
    record = {"id": "timex-only", "tokens": [["Monday"]], "events": [],
              "TIMEX": [{"id": "TIME_ONLY", "sent_id": 0, "offset": [0, 1]}],
              "temporal_relations": {}, "causal_relations": {}, "subevent_relations": []}
    (tmp_path / "train.jsonl").write_text(json.dumps(record) + "\n")
    dataset = myDataset(FixtureTokenizer(), str(tmp_path), "train", max_length=32)
    batch = collator([dataset[0]])
    output = tiny_model()(**batch)
    loss, parts = joint_loss(output, batch)
    assert loss.item() == 0 and all(torch.isfinite(value) for value in parts.values())


def test_official_temporal_evaluator_preserves_raw_direction():
    # 真实官方评分函数只认可原始方向，额外反向预测须产生假正例。
    gold = gold_document("official-fixture")
    evaluator = load_official_evaluator()
    prediction = {"id": gold["id"], "temporal_relations": {
        "BEFORE": [["m_a1", "m_b"], ["m_a2", "m_b"]],
        "SIMULTANEOUS": [["m_b", "m_c"]],
    }}
    actual = evaluator.evaluate({gold["id"]: gold}, {gold["id"]: prediction}, "temporal")
    assert actual["temporal_f1"] == 100.0
    prediction["temporal_relations"]["SIMULTANEOUS"].append(["m_c", "m_b"])
    reversed_score = evaluator.evaluate({gold["id"]: gold}, {gold["id"]: prediction}, "temporal")
    assert reversed_score["temporal_recall"] == 100.0
    assert reversed_score["temporal_precision"] == 75.0
