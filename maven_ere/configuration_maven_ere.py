"""MAVEN-ERE 四任务联合模型的独立 Hugging Face 配置。"""

from copy import deepcopy

from transformers import PretrainedConfig, RobertaConfig


TEMPORAL_LABEL2ID = {
    "BEFORE": 0,
    "OVERLAP": 1,
    "CONTAINS": 2,
    "SIMULTANEOUS": 3,
    "ENDS-ON": 4,
    "BEGINS-ON": 5,
    "NONE": 6,
}
CAUSAL_LABEL2ID = {"NONE": 0, "PRECONDITION": 1, "CAUSE": 2}
SUBEVENT_LABEL2ID = {"NONE": 0, "subevent": 1}
DEFAULT_LOSS_WEIGHTS = {
    "coreference": 0.4,
    "temporal": 2.0,
    "causal": 4.0,
    "subevent": 4.0,
}


class MavenEreConfig(PretrainedConfig):
    """保存编码器、四个任务头及文档预处理的完整构造信息。

    ``encoder_config`` 可以是 RobertaConfig 或其 ``to_dict()`` 结果；
    对外统一保留 JSON 可序列化的字典，不存放机器相关的模型绝对路径。
    """

    model_type = "maven_ere"

    def __init__(
        self,
        encoder_config=None,
        encoder_attention_implementation="eager",
        scorer_hidden_size=200,
        span_pooling="mean",
        dropout=0.2,
        loss_weights=None,
        temporal_label2id=None,
        causal_label2id=None,
        subevent_label2id=None,
        chunk_length=256,
        scorer_num_layers=3,
        **kwargs,
    ):
        if encoder_config is None:
            encoder_config = RobertaConfig().to_dict()
        elif isinstance(encoder_config, PretrainedConfig):
            encoder_config = encoder_config.to_dict()
        elif not isinstance(encoder_config, dict):
            raise TypeError("encoder_config must be a RobertaConfig or a dict")
        encoder_config = deepcopy(encoder_config)
        if encoder_config.get("model_type", "roberta") != "roberta":
            raise ValueError("This reproduction supports the RoBERTa encoder only")
        encoder_config["model_type"] = "roberta"
        encoder_config = RobertaConfig.from_dict(encoder_config).to_dict()
        if scorer_hidden_size <= 0:
            raise ValueError("scorer_hidden_size must be positive")
        if scorer_num_layers not in {2, 3}:
            raise ValueError("scorer_num_layers must be 2 or 3")
        if encoder_attention_implementation not in {"eager", "sdpa"}:
            raise ValueError("encoder_attention_implementation must be 'eager' or 'sdpa'")
        if span_pooling not in {"mean", "max"}:
            raise ValueError("span_pooling must be 'mean' or 'max'")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if chunk_length <= 0:
            raise ValueError("chunk_length must be positive")

        self.encoder_config = encoder_config
        self.encoder_attention_implementation = encoder_attention_implementation
        self.scorer_hidden_size = int(scorer_hidden_size)
        self.scorer_num_layers = int(scorer_num_layers)
        self.span_pooling = span_pooling
        self.dropout = float(dropout)
        self.loss_weights = deepcopy(
            DEFAULT_LOSS_WEIGHTS if loss_weights is None else loss_weights
        )
        if set(self.loss_weights) != set(DEFAULT_LOSS_WEIGHTS):
            raise ValueError("loss_weights must specify all four relation tasks")
        if any(weight < 0 for weight in self.loss_weights.values()):
            raise ValueError("loss weights must be nonnegative")
        self.temporal_label2id = self._label_map(
            temporal_label2id, TEMPORAL_LABEL2ID, "temporal"
        )
        self.causal_label2id = self._label_map(
            causal_label2id, CAUSAL_LABEL2ID, "causal"
        )
        self.subevent_label2id = self._label_map(
            subevent_label2id, SUBEVENT_LABEL2ID, "subevent"
        )
        # 不占用 PretrainedConfig 用于文本生成的 max_length 字段。
        self.chunk_length = int(chunk_length)
        super().__init__(**kwargs)

    @staticmethod
    def _label_map(supplied, default, name):
        result = deepcopy(default if supplied is None else supplied)
        if result != default:
            raise ValueError(f"{name}_label2id must preserve the official label IDs")
        return result


# save_pretrained 时将本文件一并复制到导出目录，供 Hub 独立加载。
MavenEreConfig.register_for_auto_class()
