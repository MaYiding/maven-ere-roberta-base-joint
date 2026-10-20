"""Reference-runner contracts; run only in the authorized remote environment."""

import ast
import importlib.util
import math
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("upstream_reference_runner", ROOT / "scripts/run_upstream_reference.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.fixture
def pinned_blobs(tmp_path):
    return RUNNER.extract_sources(ROOT, tmp_path / "original")


def test_extraction_uses_git_blobs_and_preserves_original_coref_bug(pinned_blobs):
    original = RUNNER.git_bytes(ROOT, "show", RUNNER.UPSTREAM_COMMIT + ":joint/src/utils.py")
    assert pinned_blobs["joint/src/utils.py"] == original
    # Current repaired workspace utils must not be substituted for this version.
    assert b"if len(gr) > 1:" in original
    assert b"filled_labels[gr[0]][gr[0]] = 1" in original
    assert b"sorted_gr[0], sorted_gr[0]" not in original
    assert "utils/model.py" in pinned_blobs and "utils/utils.py" in pinned_blobs
    assert "evaluate.py" in pinned_blobs and "LICENSE" in pinned_blobs


def test_patch_is_fail_closed_and_preserves_training_math(pinned_blobs):
    patched = RUNNER.patched_sources(pinned_blobs)
    assert set(patched) == {"joint/main.py", "joint/src/model.py", "joint/src/metrics.py", "joint/src/linear_assignment.py"}
    main = patched["joint/main.py"]
    for text in (
        "num_training_steps=len(train_dataloader) * args.epochs",
        "loss=loss / args.accumulation_steps",
        "if glb_step % args.accumulation_steps ==0:",
        "torch.sum(torch.clamp(torch.mul(prob, filled_labels), eps, 1-eps), dim=1)",
        "torch.mean(torch.log(prob_sum)) * -1 / float(len(coref_scores))",
        'res[k]["micro avg"]["f1-score"] > best_score[k]',
        'res["COREFERENCE"][k]["f1"] > best_score["COREFERENCE"][k]',
    ):
        assert text in main
    assert "clip_grad_norm" not in main and "autocast" not in main
    assert "max_length=256" in main and "max_length=512" not in main
    assert main.count("reference_loader(") == 3
    assert "@torch.no_grad()\ndef evaluate" in main
    assert "reference_should_stop(glb_step, args)" in main
    assert main.index("reference_before_update(model, glb_step)") < main.index("optimizer.step()")
    assert "/data/MODELS/roberta-base" not in main
    assert '"../data/MAVEN_ERE/test.jsonl"' not in main
    changed = dict(pinned_blobs)
    changed["joint/main.py"] = changed["joint/main.py"].replace(b"glb_step += 1", b"glb_step += 2")
    with pytest.raises(ValueError, match="patch mismatch"):
        RUNNER.patched_sources(changed)


def test_model_head_and_pair_scoring_ast_unchanged(pinned_blobs):
    original = ast.parse(pinned_blobs["joint/src/model.py"].decode())
    patched = ast.parse(RUNNER.patched_sources(pinned_blobs)["joint/src/model.py"])
    for name in ("Score", "PairScorer", "CorefPairScorer"):
        before = next(node for node in original.body if isinstance(node, ast.ClassDef) and node.name == name)
        after = next(node for node in patched.body if isinstance(node, ast.ClassDef) and node.name == name)
        assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)
    source = RUNNER.patched_sources(pinned_blobs)["joint/src/model.py"]
    assert "isinstance(config, RobertaConfig)" in source
    assert "self.model = nn.DataParallel(self.model)" in source
    assert "output_hidden_states=True" in source


def test_all_generated_and_patched_sources_are_valid_ast(pinned_blobs):
    for name, source in RUNNER.patched_sources(pinned_blobs).items():
        ast.parse(source, filename=name)
    ast.parse(RUNNER.LEGACY_ADAMW_SOURCE)
    ast.parse(RUNNER.REFERENCE_HOOKS_SOURCE)
    assert "Apache-2.0" in RUNNER.LEGACY_ADAMW_SOURCE
    assert "v4.24.0" in RUNNER.LEGACY_ADAMW_SOURCE


def test_cli_reference_defaults_and_canary_limit():
    common = ["--base-model", "/tmp/model", "--data-dir", "/tmp/data", "--output-dir", "/tmp/run"]
    args = RUNNER.parse_args(common)
    assert args.epochs == 100 and args.seed == 42
    assert args.max_steps is None and not args.gradient_checkpointing
    canary = RUNNER.parse_args(common + ["--canary", "--max-steps", "999"])
    assert canary.max_steps == 2 and canary.canary
    assert RUNNER.parse_args(common + ["--smoke"]).canary


def test_protected_paths_are_rejected():
    with pytest.raises(ValueError, match="Protected"):
        RUNNER.checked_path("/Users/mayiding/Library/Mobile Documents/com~apple~CloudDocs/example")


def test_gpu_lock_descriptor_helper_passes_only_open_valid_descriptors():
    assert RUNNER.inherited_gpu_lock_fds({}) == ()
    for value in ("invalid", "-1", "999999999999999999999999999999999"):
        assert RUNNER.inherited_gpu_lock_fds({"MAVEN_CAMPAIGN_GPU_LOCK_FD": value}) == ()
    read_fd, write_fd = os.pipe()
    try:
        assert RUNNER.inherited_gpu_lock_fds({"MAVEN_CAMPAIGN_GPU_LOCK_FD": str(read_fd)}) == (read_fd,)
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert RUNNER.inherited_gpu_lock_fds({"MAVEN_CAMPAIGN_GPU_LOCK_FD": str(read_fd)}) == ()


def test_legacy_adamw_retains_uncorrected_second_moment_epsilon_math():
    torch = pytest.importorskip("torch")
    namespace = {}
    exec(RUNNER.LEGACY_ADAMW_SOURCE, namespace)
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = namespace["AdamW"]([parameter], lr=0.1, eps=1e-6, weight_decay=0.03)
    expected, first, second = 1.0, 0.0, 0.0
    for step, gradient in enumerate((1e-8, -3e-8, 0.2), 1):
        first = 0.9 * first + 0.1 * gradient
        second = 0.999 * second + 0.001 * gradient * gradient
        step_size = 0.1 * math.sqrt(1 - 0.999 ** step) / (1 - 0.9 ** step)
        expected -= step_size * first / (math.sqrt(second) + 1e-6)
        expected *= 1 - 0.1 * 0.03
        parameter.grad = torch.tensor([gradient], dtype=torch.float64)
        optimizer.step()
        assert parameter.item() == pytest.approx(expected, rel=0, abs=2e-15)
    assert optimizer.state[parameter]["step"] == 3


def test_canary_selection_covers_longest_documents_and_actual_task_positives(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("MAVEN_REFERENCE_OUTPUT", str(tmp_path))
    namespace = {}
    exec(RUNNER.REFERENCE_HOOKS_SOURCE, namespace)
    dataset = []
    for index in range(80):
        dataset.append({
            "input_ids": torch.ones((10 if index == 79 else 1, 16), dtype=torch.long),
            "event_spans": [[(1, 2)] * (20 if index == 78 else 2)],
            "coref_labels": [[0, 1]] if index == 77 else [[0], [1]],
            "temporal_labels": torch.tensor([0 if index == 76 else 6]),
            "causal_labels": torch.tensor([1 if index == 75 else 0]),
            "subevent_labels": torch.tensor([1 if index == 74 else 0]),
            "doc_id": "document-" + str(index),
        })
    selected = namespace["_longest_and_positive_indices"](dataset)
    assert len(selected) == len(set(selected)) == 64
    assert {74, 75, 76, 77, 78, 79}.issubset(selected)
    assert namespace["_longest_and_positive_indices"](dataset) == selected


def test_diagnostics_reject_nonfinite_gradients_and_checkpoint_weights_without_changing_them(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv("MAVEN_REFERENCE_OUTPUT", str(tmp_path))
    namespace = {}
    exec(RUNNER.REFERENCE_HOOKS_SOURCE, namespace)
    model = torch.nn.Linear(2, 1)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    before = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    namespace["reference_before_update"](model, 4)
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, before[name], rtol=0, atol=0)
    namespace["reference_weight_check"](model, "COREFERENCE")
    assert namespace["FINITE_WEIGHT_CHECKS"]["COREFERENCE"] is True
    model.weight.grad[0, 0] = torch.inf
    with pytest.raises(FloatingPointError, match="gradients"):
        namespace["reference_before_update"](model, 8)
    with torch.no_grad():
        model.weight[0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="weights"):
        namespace["reference_weight_check"](model, "TEMPORAL")
