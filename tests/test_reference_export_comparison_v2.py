"""Standard-library contracts for v2 comparison helpers; no model imports.

Run with pytest in the authorized remote environment. Supports an explicit
MAVEN_EXPORT_EVALUATOR, remote tools/tests layout, public repository tests/
layout, and a local ops file beside the evaluator. No ML execution is needed.
"""

import copy
import importlib.util
import math
import os
from pathlib import Path
import unittest


def evaluator_path():
    configured = os.environ.get("MAVEN_EXPORT_EVALUATOR")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    here = Path(__file__).resolve()
    candidates = [here.parents[2] / "evaluate_reference_exports.py",
                  here.parents[1] / "scripts/evaluate_reference_exports.py",
                  here.parent / "evaluate_reference_exports.py"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Set MAVEN_EXPORT_EVALUATOR to the v2 evaluator")


SPEC = importlib.util.spec_from_file_location("reference_export_comparison_v2_helper", evaluator_path())
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


def predictions(before=False):
    temporal = {label: [] for label in HELPER.RELATIONS["temporal"] if label != "NONE"}
    if before:
        temporal["BEFORE"] = [["m1", "m2"]]
    return {"doc": {"id": "doc", "coreference": [["m1"], ["m2"]],
                    "temporal_relations": temporal, "causal_relations": {"CAUSE": [], "PRECONDITION": []},
                    "subevent_relations": []}}


def metrics(precision):
    return {"split": "valid", "documents": 1,
            "upstream_augmented_protocol_fraction": {"TEMPORAL": {"BEFORE": {"precision": precision}}},
            "official_protocol": {"temporal": {"temporal_precision": precision * 100,
                                                "temporal_recall": 50.0, "temporal_f1": 52.0}}}


class ReferenceExportComparisonV2Tests(unittest.TestCase):
    def test_import_is_standard_library_only(self):
        self.assertIsNone(HELPER.TORCH)
        self.assertEqual(HELPER.ASSESSMENT_VERSION, 2)

    def test_gpu_report_keeps_actual_difference_and_original_scores(self):
        gpu_precision = 0.5482913934333234
        cpu_precision = 0.5482882831778086
        gpu_metrics, cpu_metrics = metrics(gpu_precision), metrics(cpu_precision)
        original_gpu, original_cpu = copy.deepcopy(gpu_metrics), copy.deepcopy(cpu_metrics)
        report = HELPER.gpu_reference_report("TEMPORAL",
            gpu_metrics["upstream_augmented_protocol_fraction"], gpu_metrics["official_protocol"]["temporal"],
            cpu_metrics, predictions(False), predictions(True), {"doc"})
        self.assertEqual(gpu_metrics, original_gpu)
        self.assertEqual(cpu_metrics, original_cpu)
        difference = report["augmented_exact_metric_differences"][0]
        self.assertEqual(difference["gpu_reference"], gpu_precision)
        self.assertEqual(difference["cpu_actual"], cpu_precision)
        self.assertEqual(difference["cpu_minus_gpu"], cpu_precision - gpu_precision)
        selected = report["selected_task_prediction_differences"]
        self.assertEqual(selected["changed_document_count"], 1)
        self.assertEqual(selected["changed_relation_pair_count"], 1)
        self.assertEqual(selected["document_differences"][0]["differences"],
                         [{"pair": ["m1", "m2"], "gpu_label": "NONE", "cpu_label": "BEFORE"}])
        self.assertFalse(report["scores_modified"])
        self.assertFalse(report["tolerances_relaxed"])
        self.assertNotIn("matches", report)
        report["original_gpu_augmented_protocol_fraction"]["TEMPORAL"]["BEFORE"]["precision"] = 0
        self.assertEqual(gpu_metrics, original_gpu)

    def test_cpu_gate_rejects_one_prediction_change_even_with_equal_metrics(self):
        common = metrics(0.5)
        with self.assertRaisesRegex(AssertionError, "predictions differ"):
            HELPER.assert_cpu_equivalent({"doc"}, common, copy.deepcopy(common),
                                         predictions(False), predictions(True), "fixture")

    def test_cpu_gate_requires_all_expected_document_coverage(self):
        common = metrics(0.5)
        with self.assertRaisesRegex(ValueError, "coverage"):
            HELPER.assert_cpu_equivalent({"doc", "missing"}, common, common,
                                         predictions(), predictions(), "fixture")
        with self.assertRaisesRegex(ValueError, "coverage"):
            HELPER.prediction_field_differences("TEMPORAL", predictions(), {}, {"doc"})
        with self.assertRaisesRegex(ValueError, "coverage"):
            HELPER.assert_cpu_equivalent(set(), common, common, {}, {}, "fixture")

    def test_cpu_gate_does_not_relax_even_a_single_ulp_metric_difference(self):
        left, right = metrics(0.5), metrics(math.nextafter(0.5, 1.0))
        with self.assertRaisesRegex(AssertionError, "CPU metrics"):
            HELPER.assert_cpu_equivalent({"doc"}, left, right, predictions(), predictions(), "fixture")
        differences = HELPER.exact_metric_differences({"score": math.nextafter(1.0, 2.0)}, {"score": 1.0})
        self.assertEqual(len(differences), 1)
        self.assertGreater(differences[0]["cpu_minus_gpu"], 0)

    def test_exact_cpu_predictions_and_metrics_pass(self):
        common = metrics(0.5)
        result = HELPER.assert_cpu_equivalent({"doc"}, common, copy.deepcopy(common),
            predictions(True), predictions(True), "fixture")
        self.assertTrue(result["all_predictions_exact"])
        self.assertTrue(result["all_metrics_exact"])
        self.assertEqual(result["metric_tolerance"], 0.0)
        self.assertEqual(result["documents"], 1)

    def test_canary_never_compares_subset_scores_to_full_gpu_scores(self):
        gpu_metrics, cpu_metrics = metrics(0.9), metrics(0.1)
        report = HELPER.gpu_reference_report("TEMPORAL",
            gpu_metrics["upstream_augmented_protocol_fraction"], gpu_metrics["official_protocol"]["temporal"],
            cpu_metrics, predictions(), predictions(), {"doc"}, canary=True)
        self.assertEqual(report["comparison_scope"], "valid_subset")
        self.assertFalse(report["metric_comparison_performed"])
        self.assertIsNone(report["augmented_exact_metric_differences"])
        self.assertIsNone(report["selected_raw_exact_metric_differences"])


if __name__ == "__main__":
    unittest.main()
