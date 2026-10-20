"""基于 THU-KEG/MAVEN-ERE 官方 joint 模型的可独立发布实现。

保留三个有向事件对分类头和先行词共指头；文件不依赖上游 joint 包。
上游源码与许可证见 https://github.com/THU-KEG/MAVEN-ERE 。
"""

from dataclasses import dataclass
from operator import index
from typing import List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, PreTrainedModel, RobertaConfig
from transformers.utils import ModelOutput

from .configuration_maven_ere import MavenEreConfig


@dataclass
class MavenEreOutput(ModelOutput):
    """四任务输出；关系头按文档补齐，coref 保留各文档真实事件数。

    coref_log_probs[d] 为 [N, N]：第 i 行的列 0..i-1 是先行词，
    列 i 是分数固定为 0 的 dummy；列 j>i 恒为 -inf。概率计算固定
    使用 float32，训练时可直接在 log 空间聚合正确先行词。
    """

    coref_log_probs: Optional[List[torch.Tensor]] = None
    temporal_logits: Optional[torch.Tensor] = None
    causal_logits: Optional[torch.Tensor] = None
    subevent_logits: Optional[torch.Tensor] = None


class Score(nn.Module):
    """沿用官方三层 Linear、两层 ReLU/Dropout 的评分网络。"""

    def __init__(self, embeds_dim, out_dim, hidden_dim=200, dropout=0.2):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(embeds_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        for module in self.score:
            if isinstance(module, nn.Linear):
                module._maven_ere_head = True

    def forward(self, inputs):
        return self.score(inputs)


class PairScorer(nn.Module):
    def __init__(self, embed_dim=768, out_dim=7, hidden_dim=200, dropout=0.2):
        super().__init__()
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.score = Score(embed_dim * 2, out_dim, hidden_dim, dropout)

    def forward(self, event_embed):
        all_scores = []
        for embed in event_embed:
            n_mentions = embed.shape[0]
            if n_mentions < 2:
                all_scores.append(embed.new_empty((0, self.out_dim)))
                continue
            # 顺序必须与官方标签一致：i 为外循环、j 为内循环，排除 i==j。
            first, second = torch.where(
                ~torch.eye(n_mentions, dtype=torch.bool, device=embed.device)
            )
            pairs = torch.cat((embed[first], embed[second]), dim=-1)
            all_scores.append(self.score(pairs))
        if not all_scores:
            return next(self.parameters()).new_empty((0, 0, self.out_dim))
        max_pairs = max(scores.shape[0] for scores in all_scores)
        # 0/1 个 mention 没有事件对；不伪造一个需要 driver 特判的标签位置。
        return torch.stack(
            [F.pad(scores, (0, 0, 0, max_pairs - scores.shape[0]), value=-1000.0)
             for scores in all_scores]
        )


class CorefPairScorer(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=200, dropout=0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.score = Score(embed_dim * 3, 1, hidden_dim, dropout)

    def forward(self, event_embed, event_idx):
        all_log_probs = []
        for embed, selected in zip(event_embed, event_idx):
            selected = torch.as_tensor(selected, dtype=torch.long, device=embed.device)
            embed = embed.index_select(0, selected)
            n_events = embed.shape[0]
            if n_events == 0:
                all_log_probs.append(embed.new_empty((0, 0), dtype=torch.float32))
                continue
            scores = torch.full(
                (n_events, n_events), -torch.inf,
                dtype=torch.float32, device=embed.device,
            )
            scores.diagonal().fill_(0.0)
            if n_events > 1:
                mention, antecedent = torch.tril_indices(
                    n_events, n_events, offset=-1, device=embed.device
                )
                current, previous = embed[mention], embed[antecedent]
                pairs = torch.cat((current, previous, current * previous), dim=-1)
                scores[mention, antecedent] = self.score(pairs).squeeze(-1).float()
            all_log_probs.append(F.log_softmax(scores, dim=-1, dtype=torch.float32))
        return all_log_probs


class MavenEreModel(PreTrainedModel):
    config_class = MavenEreConfig
    base_model_prefix = "encoder"
    main_input_name = "input_ids"

    def __init__(self, config):
        super().__init__(config)
        encoder_config = RobertaConfig.from_dict(config.encoder_config)
        # 只按配置构造；基础权重由训练入口显式载入，导出模型不再次联网。
        self.encoder = AutoModel.from_config(
            encoder_config,
            attn_implementation=config.encoder_attention_implementation,
        )
        hidden_size = encoder_config.hidden_size
        scorer_args = dict(
            embed_dim=hidden_size,
            hidden_dim=config.scorer_hidden_size,
            dropout=config.dropout,
        )
        self.temporal_scorer = PairScorer(
            out_dim=len(config.temporal_label2id), **scorer_args
        )
        self.causal_scorer = PairScorer(
            out_dim=len(config.causal_label2id), **scorer_args
        )
        self.subevent_scorer = PairScorer(
            out_dim=len(config.subevent_label2id), **scorer_args
        )
        self.coref_scorer = CorefPairScorer(**scorer_args)
        self.post_init()

    def _init_weights(self, module):
        # 分类头沿用 PyTorch Linear 默认初始化，编码器使用 RoBERTa 的规则。
        if isinstance(module, nn.Linear):
            if getattr(module, "_maven_ere_head", False):
                module.reset_parameters()
            else:
                module.weight.data.normal_(
                    mean=0.0, std=self.config.encoder_config.get("initializer_range", 0.02)
                )
                if module.bias is not None:
                    module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.encoder_config.get("initializer_range", 0.02)
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, embeddings):
        self.encoder.set_input_embeddings(embeddings)
        self.encoder.config.vocab_size = embeddings.num_embeddings
        self.config.encoder_config["vocab_size"] = embeddings.num_embeddings

    @staticmethod
    def _indices(values, name):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().tolist()
        try:
            return [index(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain integer indices") from exc

    def _event_embeddings(self, hidden_states, event_spans, splits, events_idx):
        n_chunks, sequence_length, hidden_size = hidden_states.shape
        splits = self._indices(splits, "splits")
        if not splits or splits[0] != 0 or splits[-1] != n_chunks:
            raise ValueError("splits must start at 0 and end at the input chunk count")
        if any(left > right for left, right in zip(splits, splits[1:])):
            raise ValueError("splits must be nondecreasing")
        n_docs = len(splits) - 1
        if len(event_spans) != n_docs or len(events_idx) != n_docs:
            raise ValueError("event_spans and events_idx must have one entry per document")

        all_embeds, checked_indices = [], []
        for doc_index, doc_spans in enumerate(event_spans):
            chunk_start, chunk_end = splits[doc_index:doc_index + 2]
            if len(doc_spans) != chunk_end - chunk_start:
                raise ValueError("Each document needs one span list per input chunk")
            embeds = []
            for chunk_index, spans in enumerate(doc_spans):
                for span in spans:
                    bounds = self._indices(span, "event span")
                    if len(bounds) != 2:
                        raise ValueError("Each event span must contain start and end")
                    start, end = bounds
                    if not 0 <= start < end <= sequence_length:
                        raise ValueError("Event spans must be nonempty and within their chunk")
                    tokens = hidden_states[chunk_start + chunk_index, start:end]
                    pooled = (
                        tokens.mean(dim=0) if self.config.span_pooling == "mean"
                        else tokens.max(dim=0).values
                    )
                    embeds.append(pooled)
            embed = (
                torch.stack(embeds) if embeds
                else hidden_states.new_empty((0, hidden_size))
            )
            selected = self._indices(events_idx[doc_index], "events_idx")
            if len(set(selected)) != len(selected):
                raise ValueError("events_idx cannot contain duplicate mentions")
            if any(item < 0 or item >= embed.shape[0] for item in selected):
                raise ValueError("events_idx contains a mention outside the document")
            all_embeds.append(embed)
            checked_indices.append(selected)
        return all_embeds, checked_indices

    def forward(
        self,
        inputs=None,
        input_ids=None,
        attention_mask=None,
        event_spans=None,
        splits=None,
        events_idx=None,
        **kwargs,
    ):
        """接受官方 batch 字典或关键字；其余标签/文档元数据由 driver 使用。

        事件 span 是每文档、每 chunk 的半开区间；events_idx 仅选择事件，
        排除 TIMEX。三个关系头仍包含全部 mention 对，由标签的 -100
        掩掉因果/子事件中的 TIMEX 以及 batch padding。
        """
        batch = {} if inputs is None else dict(inputs)
        explicit = dict(
            input_ids=input_ids, attention_mask=attention_mask,
            event_spans=event_spans, splits=splits, events_idx=events_idx,
        )
        batch.update({key: value for key, value in explicit.items() if value is not None})
        for required in ("input_ids", "event_spans", "splits", "events_idx"):
            if required not in batch or batch[required] is None:
                raise ValueError(f"Missing required input: {required}")
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [chunks, sequence_length]")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if input_ids.shape[0]:
            hidden_states = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state
        else:
            hidden_states = self.get_input_embeddings().weight.new_empty(
                (0, input_ids.shape[1], self.config.encoder_config["hidden_size"]),
                device=input_ids.device,
            )
        event_embed, events_idx = self._event_embeddings(
            hidden_states, batch["event_spans"], batch["splits"], batch["events_idx"]
        )
        return MavenEreOutput(
            coref_log_probs=self.coref_scorer(event_embed, events_idx),
            temporal_logits=self.temporal_scorer(event_embed),
            causal_logits=self.causal_scorer(event_embed),
            subevent_logits=self.subevent_scorer(event_embed),
        )


MavenEreModel.register_for_auto_class("AutoModel")
