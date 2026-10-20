"""改进候选训练契约；只在授权的 LXC 901 运行，无网络依赖。"""

import copy
import json
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from transformers import RobertaConfig, RobertaModel

import train_joint as training
from maven_ere import MavenEreModel


class StoredState:
    def __init__(self, value=None):
        self.value = copy.deepcopy(value or {})

    def state_dict(self):
        return copy.deepcopy(self.value)

    def load_state_dict(self, value):
        self.value = copy.deepcopy(value)


class CompleteCheckpoint(StoredState):
    """测试选择/事务，不用随机 logits 冒充不同权重快照。"""

    def __init__(self, marker):
        super().__init__({key: marker for key in ("encoder", *training.SELECTION_TASKS)})
        self.config = SimpleNamespace(to_dict=lambda: {"fixture": True})
        self.save_count = 0

    def save_pretrained(self, directory, **kwargs):
        directory.mkdir(parents=True)
        (directory / "complete_weights.json").write_text(json.dumps(self.value))
        self.save_count += 1


class TokenizerFixture:
    def save_pretrained(self, directory):
        (directory / "tokenizer_config.json").write_text("{}")


def metrics(coref=90.0, temporal=50.0, causal=20.0, subevent=10.0, coreference_metrics=None):
    coreference_metrics = coreference_metrics or {name: coref for name in training.COREFERENCE_METRICS}
    coref = sum(coreference_metrics.values()) / 4
    values = {"temporal": temporal, "causal": causal, "subevent": subevent}
    return {
        "split": "valid",
        "official_protocol": {
            "coreference": {name + "_f1": value for name, value in coreference_metrics.items()},
            "coreference_mean_f1": coref,
            "overall_f1": (coref + temporal + causal + subevent) / 4,
            **{task: {task + "_f1": value} for task, value in values.items()},
        },
        "training_augmented_labels_positive_micro": {
            task: {"f1": value} for task, value in values.items()
        },
    }


def predictions(marker):
    return {"doc": {"id": "doc", "coreference": [[marker]],
                    "temporal_relations": {"BEFORE": [[marker, "other"]]},
                    "causal_relations": {"CAUSE": [[marker, "other"]]},
                    "subevent_relations": [[marker, "other"]]}}


def progress(directory, selector="mean"):
    value = {"best_score": -1.0, "bad_evaluations": 0, "update_step": 1}
    training.restore_selection_artifacts(value, directory, selector)
    return value


def selection_args(monitor="any-task", selector="mean"):
    return SimpleNamespace(export_task_bests=True, early_stopping_monitor=monitor,
                           coreference_selection=selector)


def test_each_task_selects_one_complete_checkpoint_and_combination_is_explicit(tmp_path, monkeypatch):
    state = progress(tmp_path)
    tokenizer = TokenizerFixture()
    first = CompleteCheckpoint("A")
    training.record_validation(first, tokenizer, tmp_path, state, metrics(), predictions("A"), selection_args(), 0)
    assert first.save_count == 1  # 同一次验证的各项可共享完整模型。
    training.prune_exports(tmp_path)
    state["update_step"] = 2
    second = CompleteCheckpoint("B")
    training.record_validation(second, tokenizer, tmp_path, state, metrics(89, 55, 19, 9), predictions("B"), selection_args(), 1)
    training.prune_exports(tmp_path)
    assert json.loads((tmp_path / "best_joint" / "complete_weights.json").read_text())["encoder"] == "B"
    for task, expected in {"coreference": "A", "temporal": "B", "causal": "A", "subevent": "A"}.items():
        weights = json.loads((tmp_path / ("best_" + task) / "complete_weights.json").read_text())
        assert set(weights.values()) == {expected}  # 不允许只替换某个任务头。
        assert state["task_bests"][task]["exported"]
    seen = {}

    def score(gold, predicted):
        seen.update(predicted)
        return {"overall_f1": 0.0}

    monkeypatch.setattr(training, "score_official_predictions", score)
    combination = training.combine_task_best_predictions(tmp_path, state, {"doc": {"id": "doc"}})
    assert combination["is_single_model"] is False
    assert seen["doc"]["coreference"] == [["A"]]
    assert seen["doc"]["temporal_relations"]["BEFORE"] == [["B", "other"]]
    combined_metrics = json.loads((tmp_path / "task_best_combined_metrics.json").read_text())
    assert combined_metrics["artifact_type"] == "multiple_checkpoint_task_combination"
    assert len(list((tmp_path / "exports").iterdir())) == 2


def test_resume_rolls_back_uncommitted_history_and_links_without_losing_saved_best(tmp_path):
    state = progress(tmp_path)
    model = CompleteCheckpoint("A")
    training.record_validation(model, TokenizerFixture(), tmp_path, state, metrics(), predictions("A"), selection_args(), 0)
    training.prune_exports(tmp_path)
    optimizers = [StoredState({"step": 1}), StoredState({"step": 1}), StoredState({"step": 1})]
    checkpoint = training.checkpoint_state(model, *optimizers, state, {"fixture": True})
    committed_id = state["joint_best"]["checkpoint_id"]
    state["update_step"] = 2
    training.record_validation(CompleteCheckpoint("B"), TokenizerFixture(), tmp_path, state,
                               metrics(92, 60, 30, 20), predictions("B"), selection_args(), 1)
    assert len((tmp_path / "history.jsonl").read_text().splitlines()) == 2
    assert len(checkpoint["progress"]["validation_history"]) == 1  # 深拷贝，非活动引用。
    restored = training.restore_training_state(checkpoint, model, *optimizers)
    training.restore_selection_artifacts(restored, tmp_path)
    assert restored["joint_best"]["checkpoint_id"] == committed_id
    assert restored["coreference_selection"] == "mean"
    assert restored["coreference_metric_bests"] == {name: 90.0 for name in training.COREFERENCE_METRICS}
    assert len((tmp_path / "history.jsonl").read_text().splitlines()) == 1
    assert len(list((tmp_path / "exports").iterdir())) == 1
    for task in ("joint", *training.SELECTION_TASKS):
        assert set(json.loads((tmp_path / ("best_" + task) / "complete_weights.json").read_text()).values()) == {"A"}
    history = restored["validation_history"][0]
    assert history["epoch"] == 1 and history["update_step"] == 1
    assert history["metrics"]["official_protocol"] == metrics()["official_protocol"]
    assert history["metrics"]["training_augmented_labels_positive_micro"]


@pytest.mark.parametrize("monitor, expected_bad", [("overall", 1), ("any-task", 0)])
def test_any_task_stopping_recognizes_improvement_when_overall_declines(tmp_path, monitor, expected_bad):
    state = progress(tmp_path)
    args = selection_args(monitor)
    args.export_task_bests = False
    training.record_validation(CompleteCheckpoint("A"), TokenizerFixture(), tmp_path, state, metrics(), predictions("A"), args, 0)
    state["update_step"] = 2
    improved = training.record_validation(CompleteCheckpoint("B"), TokenizerFixture(), tmp_path, state,
                                          metrics(91, 49, 19, 8), predictions("B"), args, 1)
    assert not improved
    assert state["bad_evaluations"] == expected_bad
    assert state["task_bests"]["coreference"]["score"] == 91
    assert state["task_bests"]["coreference"]["exported"] is False


@pytest.mark.parametrize("task", ("temporal", "causal", "subevent"))
def test_positive_weighted_ce_uses_task_none_id_and_ignores_padding(task):
    none = training.TASK_MAPS[task]["NONE"]
    positive = next(index for index in training.TASK_MAPS[task].values() if index != none)
    logits = torch.arange(3 * len(training.TASK_MAPS[task]), dtype=torch.float32).reshape(3, -1)
    logits[1] = -logits[1]
    logits.requires_grad_()
    labels = torch.tensor([none, positive, -100])
    actual = training.relation_loss(logits, labels, positive_class_weight=2, none_id=none)
    nll = -torch.log_softmax(logits[:2], dim=-1)[torch.arange(2), labels[:2]]
    torch.testing.assert_close(actual, (nll[0] + 2 * nll[1]) / 3)
    torch.testing.assert_close(training.relation_loss(logits, labels), nll.mean())
    actual.backward()
    assert torch.equal(logits.grad[2], torch.zeros_like(logits.grad[2]))


def test_model_config_and_export_preserve_candidate_architecture(tmp_path):
    base = tmp_path / "base"
    RobertaModel(RobertaConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
                               num_hidden_layers=1, num_attention_heads=2,
                               max_position_embeddings=32)).save_pretrained(base)
    args = SimpleNamespace(base_model=str(base), scorer_hidden_size=7, scorer_num_layers=2,
                           positive_class_weight=2.0, max_length=16, gradient_checkpointing=False)
    model = training.build_model(args)
    output = tmp_path / "export"
    model.save_pretrained(output, safe_serialization=True)
    config = json.loads((output / "config.json").read_text())
    assert config["scorer_hidden_size"] == 7
    assert config["scorer_num_layers"] == 2
    assert config["positive_class_weight"] == 2.0
    assert config["chunk_length"] == 16
    loaded = MavenEreModel.from_pretrained(output, local_files_only=True)
    assert set(loaded.state_dict()) == set(model.state_dict())
    for key, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, loaded.state_dict()[key], rtol=0, atol=0)


def test_new_cli_defaults_and_selection_signature_are_compatible_with_old_cli():
    args = training.parse_args(["--base-model", "base", "--data-dir", "data", "--output-dir", "run"])
    assert (args.scorer_hidden_size, args.scorer_num_layers, args.positive_class_weight) == (200, 3, 1)
    assert not args.export_task_bests and not args.skip_test_predictions
    assert args.early_stopping_monitor == "overall"
    assert args.eval_epoch_end and args.coreference_selection == "mean"
    formal = training.parse_args(["--base-model", "base", "--data-dir", "data", "--output-dir", "run",
                                  "--no-eval-epoch-end", "--coreference-selection", "upstream-any"])
    assert not formal.eval_epoch_end and formal.coreference_selection == "upstream-any"
    signature = training.training_signature(args, {"train": "hash"}, ["doc"])
    legacy = {key: value for key, value in signature.items() if key not in training.NEW_SIGNATURE_DEFAULTS}
    assert training.resume_signature_matches(legacy, signature)
    for key, changed in {"scorer_num_layers": 2, "positive_class_weight": 2,
                         "early_stopping_monitor": "any-task", "export_task_bests": True,
                         "eval_epoch_end": False, "coreference_selection": "upstream-any"}.items():
        altered = {**signature, key: changed}
        assert not training.resume_signature_matches(signature, altered)


@pytest.mark.parametrize("selector, selected", [("mean", "A"), ("upstream-any", "B")])
def test_upstream_any_can_replace_coreference_with_lower_mean_and_preserves_actual_scores(tmp_path, selector, selected):
    state = progress(tmp_path, selector)
    args = selection_args(selector=selector)
    initial = dict(muc=80.0, b_cubed=96.0, ceaf=96.0, blanc=88.0)
    changed = dict(muc=81.0, b_cubed=92.0, ceaf=94.0, blanc=87.0)
    training.record_validation(CompleteCheckpoint("A"), TokenizerFixture(), tmp_path, state,
                               metrics(coreference_metrics=initial), predictions("A"), args, 0)
    state["update_step"] = 2
    training.record_validation(CompleteCheckpoint("B"), TokenizerFixture(), tmp_path, state,
                               metrics(coreference_metrics=changed), predictions("B"), args, 1)
    entry = state["task_bests"]["coreference"]
    actual = changed if selected == "B" else initial
    assert entry["score"] == sum(actual.values()) / 4
    assert entry["coreference_metric_values"] == actual
    assert entry["selection_rule"] == selector
    assert set(json.loads((tmp_path / "best_coreference" / "complete_weights.json").read_text()).values()) == {selected}
    assert set(json.loads((tmp_path / "best_joint" / "complete_weights.json").read_text()).values()) == {"A"}
    assert state["coreference_metric_bests"] == dict(muc=81.0, b_cubed=96.0, ceaf=96.0, blanc=88.0)
    stores = [StoredState(), StoredState(), StoredState()]
    restored_model = CompleteCheckpoint("B")
    saved = training.checkpoint_state(restored_model, *stores, state, {"fixture": True})
    state = training.restore_training_state(saved, restored_model, *stores)
    training.restore_selection_artifacts(state, tmp_path, selector)
    assert state["coreference_metric_bests"] == dict(muc=81.0, b_cubed=96.0, ceaf=96.0, blanc=88.0)
    assert state["task_bests"]["coreference"]["selection_rule"] == selector
    # mean 高于 B，但没有一项突破各自历史 max，upstream-any 不得换成 C。
    state["update_step"] = 3
    training.record_validation(CompleteCheckpoint("C"), TokenizerFixture(), tmp_path, state,
                               metrics(coreference_metrics=dict(muc=80.5, b_cubed=93, ceaf=95, blanc=87.5)),
                               predictions("C"), args, 2)
    assert set(json.loads((tmp_path / "best_coreference" / "complete_weights.json").read_text()).values()) == {selected}


@pytest.mark.parametrize("step, epoch_end, schedule_end, expected", [
    (92, True, False, False), (100, False, False, True),
    (73, False, True, True), (92, True, True, True),
])
def test_disabling_epoch_end_keeps_periodic_and_final_evaluation(step, epoch_end, schedule_end, expected):
    args = SimpleNamespace(eval_every_updates=50, eval_epoch_end=False)
    assert training.should_evaluate(args, step, epoch_end, schedule_end) is expected
    assert training.should_evaluate(SimpleNamespace(eval_every_updates=0), 92, True, False)


def test_numpy_metrics_produce_builtin_history_and_serializable_resume_progress(tmp_path):
    state = progress(tmp_path, "upstream-any")
    raw = metrics()
    raw["official_protocol"]["overall_f1"] = np.float64(raw["official_protocol"]["overall_f1"])
    raw["official_protocol"]["coreference_mean_f1"] = np.float32(90)
    for group in ("coreference", "temporal", "causal", "subevent"):
        raw["official_protocol"][group] = {
            key: np.float64(value) for key, value in raw["official_protocol"][group].items()}
    raw["training_augmented_labels_positive_micro"]["causal"]["f1"] = np.float32(20)
    raw["numpy_flag"] = np.bool_(True)
    model = CompleteCheckpoint("A")
    assert training.record_validation(model, TokenizerFixture(), tmp_path, state, raw, predictions("A"),
                                      selection_args(selector="upstream-any"), 0) is True
    history = json.loads((tmp_path / "history.jsonl").read_text())
    assert type(history["overall_improved"]) is bool
    assert all(type(flag) is bool for flag in state["validation_history"][0]["task_improvements"].values())
    assert history["metrics"]["numpy_flag"] is True
    state["last_metrics"] = raw
    stores = [StoredState(), StoredState(), StoredState()]
    checkpoint = training.checkpoint_state(model, *stores, state, {"fixture": True})
    json.dumps(checkpoint["progress"], allow_nan=False)
    restored = training.restore_training_state(checkpoint, model, *stores)
    training.restore_selection_artifacts(restored, tmp_path, "upstream-any")
    assert restored["coreference_metric_bests"] == state["coreference_metric_bests"]
    json.dumps(restored, allow_nan=False)


def test_official_evaluator_numpy_scores_are_converted_to_python_floats(monkeypatch):
    evaluator = SimpleNamespace(
        evaluate_coreference=lambda gold, preds: {name + "_f1": np.float64(90) for name in training.COREFERENCE_METRICS},
        evaluate=lambda gold, preds, task: {task + "_f1": np.float32(25)},
    )
    monkeypatch.setattr(training, "load_official_evaluator", lambda: evaluator)
    official = training.score_official_predictions({}, {})
    assert type(official["overall_f1"]) is float
    assert type(official["coreference"]["muc_f1"]) is float
    assert type(official["causal"]["causal_f1"]) is float
    json.dumps(official, allow_nan=False)
