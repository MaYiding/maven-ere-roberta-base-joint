"""Exercise paper-width export and compatibility with the existing released head."""
import pytest
import torch
from torch import nn
from transformers import AutoModel, RobertaConfig

from maven_ere import MavenEreConfig, MavenEreModel


def tiny_config(**kwargs):
    encoder = RobertaConfig(vocab_size=32, hidden_size=12, num_hidden_layers=1,
                            num_attention_heads=3, intermediate_size=24,
                            max_position_embeddings=34, pad_token_id=1)
    return MavenEreConfig(encoder_config=encoder, dropout=0.0, **kwargs)


@pytest.mark.parametrize("num_layers,width", [(2, 150), (3, 200)])
def test_head_architecture_roundtrip(tmp_path, num_layers, width):
    torch.manual_seed(91)
    model = MavenEreModel(tiny_config(scorer_hidden_size=width,
                                    scorer_num_layers=num_layers)).eval()
    for name in ("temporal", "causal", "subevent", "coref"):
        head = getattr(model, name + "_scorer")
        assert len([m for m in head.modules() if isinstance(m, nn.Linear)]) == num_layers
    batch = dict(input_ids=torch.tensor([[0, 4, 5, 6, 2]]),
                 attention_mask=torch.ones(1, 5, dtype=torch.long),
                 event_spans=[[[[1, 2], [2, 3]]]], splits=[0, 1], events_idx=[[0, 1]])
    with torch.inference_mode():
        original = model(**batch)
    model.save_pretrained(tmp_path, safe_serialization=True)
    restored = AutoModel.from_pretrained(tmp_path, trust_remote_code=True,
                                        local_files_only=True).eval()
    assert restored.config.scorer_num_layers == num_layers
    assert restored.config.scorer_hidden_size == width
    with torch.inference_mode():
        actual = restored(**batch)
    for field in ("temporal_logits", "causal_logits", "subevent_logits"):
        torch.testing.assert_close(getattr(original, field), getattr(actual, field), rtol=0, atol=0)
    torch.testing.assert_close(original.coref_log_probs[0], actual.coref_log_probs[0], rtol=0, atol=0)


def test_existing_export_defaults_to_original_three_layer_layout():
    legacy = tiny_config().to_dict()
    legacy.pop("scorer_num_layers")
    config = MavenEreConfig.from_dict(legacy)
    assert config.scorer_num_layers == 3
    assert config.scorer_hidden_size == 200
    model = MavenEreModel(config)
    assert list(model.temporal_scorer.score.state_dict()) == [
        "score.0.weight", "score.0.bias", "score.3.weight", "score.3.bias",
        "score.6.weight", "score.6.bias"]
