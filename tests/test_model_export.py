"""仅使用随机 tiny RoBERTa；不下载模型或数据，预期在 LXC 901 执行。"""

import json

import pytest
import torch
from torch import nn
from transformers import AutoModel, RobertaConfig

from maven_ere import MavenEreConfig, MavenEreModel
from maven_ere.modeling_maven_ere import CorefPairScorer, PairScorer


def tiny_model():
    encoder = RobertaConfig(
        vocab_size=32,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=32,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        pad_token_id=1,
    )
    model = MavenEreModel(
        MavenEreConfig(encoder_config=encoder.to_dict(), scorer_hidden_size=6, dropout=0.0)
    )
    return model.eval()


def example_batch():
    return {
        "input_ids": torch.tensor([
            [0, 5, 6, 7, 8, 2],
            [0, 9, 10, 11, 12, 2],
            [0, 13, 14, 15, 16, 2],
        ]),
        "attention_mask": torch.ones(3, 6, dtype=torch.long),
        "event_spans": [[[(1, 2), (3, 5)], [(2, 3)]], [[]]],
        "splits": [0, 2, 3],
        "events_idx": [[0, 2], []],
        # driver 的附加字段应可直接随 batch 一起传给模型。
        "doc_id": ["three-mentions", "empty"],
    }


def test_safe_export_roundtrip_and_autoclass(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    torch.manual_seed(31)
    model = tiny_model()
    batch = example_batch()
    with torch.no_grad():
        expected = model(batch)
    model.save_pretrained(tmp_path, safe_serialization=True)
    assert (tmp_path / "model.safetensors").is_file()
    assert (tmp_path / "configuration_maven_ere.py").is_file()
    assert (tmp_path / "modeling_maven_ere.py").is_file()
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["encoder_config"]["hidden_size"] == 8
    assert config["scorer_hidden_size"] == 6
    assert config["encoder_attention_implementation"] == "eager"
    assert config["causal_label2id"] == {"NONE": 0, "PRECONDITION": 1, "CAUSE": 2}
    assert "AutoModel" in config["auto_map"]

    # 动态导入只依赖导出目录和 transformers，不依赖上游 joint 包。
    loaded = AutoModel.from_pretrained(
        tmp_path, trust_remote_code=True, local_files_only=True
    ).eval()
    assert set(loaded.state_dict()) == set(model.state_dict())
    with torch.no_grad():
        actual = loaded(**batch)
    for task in ("temporal_logits", "causal_logits", "subevent_logits"):
        torch.testing.assert_close(actual[task], expected[task], rtol=0, atol=0)
    for left, right in zip(actual.coref_log_probs, expected.coref_log_probs):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert actual.temporal_logits.shape == (2, 6, 7)
    assert actual.coref_log_probs[0].shape == (2, 2)
    assert actual.coref_log_probs[1].shape == (0, 0)
    assert torch.all(actual.temporal_logits[1] == -1000)


class DirectedPairScore(nn.Module):
    def forward(self, pair):
        return 100 * pair[:, :1] + pair[:, 2:3]


def test_relation_pair_order_keeps_direction():
    scorer = PairScorer(embed_dim=2, out_dim=1)
    scorer.score = DirectedPairScore()
    embed = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scores = scorer([embed])
    torch.testing.assert_close(
        scores[0, :, 0], torch.tensor([102.0, 103.0, 201.0, 203.0, 301.0, 302.0])
    )


class ProductPairScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_pairs = None

    def forward(self, pair):
        self.last_pairs = pair
        return pair[:, -2:].sum(dim=-1, keepdim=True)


def test_coreference_masks_future_mentions_and_uses_product_features():
    scorer = CorefPairScorer(embed_dim=2)
    classifier = ProductPairScore()
    scorer.score = classifier
    embed = torch.tensor([[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]], requires_grad=True)
    log_probs = scorer([embed], [[0, 1, 2]])[0]
    assert log_probs.dtype == torch.float32
    assert torch.isneginf(log_probs[torch.triu(torch.ones(3, 3), diagonal=1).bool()]).all()
    assert log_probs[0, 0] == 0
    torch.testing.assert_close(log_probs.exp().sum(dim=-1), torch.ones(3))
    torch.testing.assert_close(
        classifier.last_pairs[0], torch.tensor([5.0, 7.0, 2.0, 3.0, 10.0, 21.0])
    )
    # dummy 对角即使在真实先行词更强时也须保持有限、可用于稳定损失。
    assert torch.isfinite(log_probs.diagonal()).all()
    (-log_probs[1, 1]).backward()
    assert embed.grad is not None and torch.isfinite(embed.grad).all()
    assert embed.grad.abs().sum() > 0


def test_empty_and_single_mention_have_no_relation_pairs():
    model = tiny_model()
    batch = {
        "input_ids": torch.tensor([[0, 5, 6, 2], [0, 7, 8, 2]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "event_spans": [[[]], [[(1, 2)]]],
        "splits": [0, 1, 2],
        "events_idx": [[], [0]],
    }
    with torch.no_grad():
        output = model(**batch)
    assert output.temporal_logits.shape == (2, 0, 7)
    assert output.causal_logits.shape == (2, 0, 3)
    assert output.subevent_logits.shape == (2, 0, 2)
    assert output.coref_log_probs[0].shape == (0, 0)
    torch.testing.assert_close(output.coref_log_probs[1], torch.zeros(1, 1))


@pytest.mark.parametrize("indices", [[0, 0], [-1], [3]])
def test_rejects_invalid_coreference_mention_indices(indices):
    model = tiny_model()
    batch = example_batch()
    batch["events_idx"][0] = indices
    with pytest.raises(ValueError, match="events_idx"):
        model(**batch)


def test_rejects_span_outside_chunk():
    model = tiny_model()
    batch = example_batch()
    batch["event_spans"][0][0] = [(1, 7)]
    with pytest.raises(ValueError, match="within their chunk"):
        model(**batch)
