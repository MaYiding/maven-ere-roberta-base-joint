"""Run the preregistered replication matrix serially; never publish repositories.

The upstream arm preserves released training semantics. Corrected candidates are
compared using the same raw-direction validation evaluator and task-best selection
protocol. Single shared-checkpoint results are retained as a separate result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time


TASKS = ("coreference", "temporal", "causal", "subevent")
STOP = False


def stop_requested(signum, frame):
    global STOP
    STOP = True


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
    tmp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def task_scores(report):
    report = report.get("official_protocol", report)
    scores = {"coreference": report["coreference_mean_f1"]}
    scores.update({name: report[name][name + "_f1"] for name in TASKS[1:]})
    if any(not math.isfinite(x) or not 0 <= x <= 100 for x in scores.values()):
        raise ValueError("Scores must be finite validation percentages")
    scores["overall"] = sum(scores.values()) / 4
    return scores


def choose_candidate(reference, candidates):
    """Select using one preregistered matching protocol, never test labels."""
    if not candidates:
        raise ValueError("No completed tuning candidates")
    ranking = []
    for index, (name, scores) in enumerate(candidates):
        deltas = {task: scores[task] - reference[task] for task in TASKS}
        ranking.append((min(deltas.values()), scores["overall"], -index, name, deltas))
    ranking.sort(reverse=True)
    best = ranking[0]
    return best[3], [{"candidate": x[3], "minimum_task_delta": x[0],
                      "overall_f1": x[1], "task_deltas": x[4]} for x in ranking]


def summarize_paired(reference, candidate):
    if set(reference) != set(candidate) or len(reference) != 5:
        raise ValueError("Final comparison requires the same five seeds")
    report = {}
    for task in (*TASKS, "overall"):
        seeds = sorted(reference)
        ref = [reference[seed][task] for seed in seeds]
        cand = [candidate[seed][task] for seed in seeds]
        differences = [a - b for a, b in zip(cand, ref)]
        delta = statistics.mean(differences)
        sem = statistics.stdev(differences) / math.sqrt(len(seeds))
        # Student t(4), two-sided 95%; reported descriptively, no test-set claim.
        margin = 2.7764451051977987 * sem
        report[task] = {
            "reference_mean": statistics.mean(ref), "reference_sample_std": statistics.stdev(ref),
            "candidate_mean": statistics.mean(cand), "candidate_sample_std": statistics.stdev(cand),
            "paired_mean_delta": delta, "paired_95pct_ci": [delta - margin, delta + margin],
            "per_seed_delta": dict(zip(map(str, seeds), differences)),
        }
    return report


class Campaign:
    def __init__(self, root, plan_path, output, python):
        self.root, self.output, self.python = root, output, python
        self.plan_path = plan_path
        self.plan = load_json(plan_path)
        self.state_path = output / "campaign_state.json"
        self.output.mkdir(parents=True, exist_ok=True)
        self.commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        cwd=root, text=True)
        if dirty:
            raise RuntimeError("Freeze tracked source before running a formal campaign")
        plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        if self.state_path.exists():
            self.state = load_json(self.state_path)
            if self.state["code_commit"] != self.commit or self.state["plan_sha256"] != plan_sha:
                raise ValueError("Cannot resume a campaign with changed source or plan")
        else:
            self.state = {"campaign_id": self.plan["campaign_id"], "code_commit": self.commit,
                          "plan_sha256": plan_sha, "runs": {}, "status": "ready",
                          "total_formal_runs": self.plan["total_formal_runs"],
                          "visibility": "private", "test_scores_available": False}
        self.save()

    def save(self):
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.state["completed_runs"] = sum(x.get("status") == "completed"
                                            for x in self.state["runs"].values())
        atomic_json(self.state_path, self.state)

    def read_result(self, kind, folder):
        if kind == "reference":
            folder = folder / "reference"
            summary = load_json(folder / "reference_summary.json")
            if (summary["status"] != "completed" or not summary.get("experiment_complete")
                    or not summary.get("leaderboard_eligible")
                    or summary["training"]["epochs_completed"] != self.plan["epochs"]):
                raise ValueError("Reference run did not complete the declared full protocol")
            return {"scores": task_scores(summary),
                    "protocol": "task_best_combined_official_valid",
                    "summary_path": str(folder / "reference_summary.json")}
        summary = load_json(folder / "training_summary.json")
        if (summary["status"] != "completed" or summary.get("smoke")
                or summary["epoch"] != self.plan["epochs"]):
            raise ValueError("Candidate run did not complete the declared full protocol")
        combined = load_json(folder / "task_best_combined_metrics.json")
        single = load_json(folder / "best_dev_metrics.json")
        return {"scores": task_scores(combined), "single_joint_scores": task_scores(single),
                "protocol": "task_best_combined_official_valid",
                "summary_path": str(folder / "training_summary.json"),
                "combined_metrics_path": str(folder / "task_best_combined_metrics.json")}

    def run(self, kind, seed, variant=None):
        name = "reference" if kind == "reference" else variant["id"]
        run_id = name + "-seed" + str(seed)
        folder = self.output / run_id
        record = self.state["runs"].get(run_id)
        if record and record.get("status") == "completed":
            result = self.read_result(kind, folder)
            if result != record["result"]:
                raise ValueError("Previously completed result changed: " + run_id)
            return result
        if STOP:
            raise InterruptedError("Campaign stop requested")
        if shutil.disk_usage(self.root).free < self.plan["minimum_free_gib"] * 1024**3:
            raise RuntimeError("Insufficient disk headroom before " + run_id)
        folder.mkdir(parents=True, exist_ok=True)
        base, data = self.plan["base_model"], self.plan["data_dir"]
        if kind == "reference":
            argv = [self.python, "scripts/run_upstream_reference.py", "--base-model", base,
                    "--data-dir", data, "--output-dir", str(folder / "reference"), "--seed", str(seed),
                    "--epochs", str(self.plan["epochs"])]
            if self.plan.get("reference_gradient_checkpointing", False):
                argv.append("--gradient-checkpointing")
            if record:
                # Original reference state is not a complete resume checkpoint.
                # Reassess an already-finished summary, otherwise require an explicit repair.
                if (folder / "reference" / "reference_summary.json").exists():
                    try:
                        result = self.read_result(kind, folder)
                    except (ValueError, KeyError):
                        pass
                    else:
                        record.update(status="completed", result=result)
                        self.save()
                        return result
                raise RuntimeError("Incomplete author reference requires preserved-evidence repair: " + run_id)
        else:
            argv = [self.python, "train_joint.py", "--base-model", base, "--data-dir", data,
                    "--output-dir", str(folder), "--cache-dir", self.plan["cache_dir"],
                    "--seed", str(seed), "--max_epochs", str(self.plan["epochs"]),
                    "--batch_size", str(self.plan["batch_size"]), "--grad_accum", str(self.plan["grad_accum"]),
                    "--eval_every_updates", "50", "--save_every_updates", "100",
                    "--early_stopping_patience", "0", "--no-bf16", "--export-task-bests",
                    "--skip-test-predictions"]
            for field in ("max_length", "scorer_hidden_size", "scorer_num_layers", "positive_class_weight"):
                argv.extend(["--" + field, str(variant[field])])
            if (folder / "latest.pt").exists():
                argv.append("--resume")
            elif record:
                raise RuntimeError("Incomplete candidate has no resumable checkpoint: " + run_id)
        self.state["status"] = "running"
        self.state["active_run"] = run_id
        record = {"kind": kind, "seed": seed, "variant": variant, "status": "running",
                  "output_dir": str(folder), "argv": argv,
                  "started_at": datetime.now(timezone.utc).isoformat()}
        self.state["runs"][run_id] = record
        self.save()
        atomic_json(folder / "campaign_command.json", {"code_commit": self.commit, "argv": argv})
        print("START " + run_id, flush=True)
        forwarded = False
        with (folder / "process.log").open("a") as log:
            process = subprocess.Popen(argv, cwd=self.root, stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            self.save()
            while process.poll() is None:
                if STOP and not forwarded:
                    process.send_signal(signal.SIGTERM)
                    forwarded = True
                record["heartbeat_at"] = datetime.now(timezone.utc).isoformat()
                progress = (folder / "reference" if kind == "reference" else folder) / "progress.json"
                if progress.exists():
                    try:
                        payload = load_json(progress)
                        record["progress"] = {k: payload[k] for k in
                                              ("status", "epoch", "update_step", "micro_step", "best_score")
                                              if k in payload}
                    except (OSError, ValueError):
                        pass
                self.save()
                time.sleep(10)
        record.pop("pid", None)
        record["exit_code"] = process.returncode
        record["ended_at"] = datetime.now(timezone.utc).isoformat()
        if process.returncode or STOP:
            record["status"] = "interrupted" if STOP else "failed"
            self.save()
            raise RuntimeError("Run exited unsuccessfully: " + run_id)
        try:
            result = self.read_result(kind, folder)
        except Exception:
            record["status"] = "invalid_result"
            self.save()
            raise
        record.update(status="completed", result=result)
        self.state.pop("active_run", None)
        self.save()
        print("COMPLETE " + run_id + " " + json.dumps(result["scores"]), flush=True)
        return result

    def execute(self):
        tune_seed = self.plan["tuning_seed"]
        reference = {tune_seed: self.run("reference", tune_seed)["scores"]}
        candidates = []
        for variant in self.plan["candidates"]:
            result = self.run("candidate", tune_seed, variant)
            candidates.append((variant["id"], result["scores"]))
        winner, ranking = choose_candidate(reference[tune_seed], candidates)
        self.state.update(selected_candidate=winner, selection_ranking=ranking)
        self.save()
        variant = next(x for x in self.plan["candidates"] if x["id"] == winner)
        candidate = {tune_seed: dict(candidates)[winner]}
        single = {tune_seed: self.run("candidate", tune_seed, variant)["single_joint_scores"]}
        for seed in self.plan["seeds"]:
            if seed != tune_seed:
                reference[seed] = self.run("reference", seed)["scores"]
        for seed in self.plan["seeds"]:
            if seed != tune_seed:
                result = self.run("candidate", seed, variant)
                candidate[seed], single[seed] = result["scores"], result["single_joint_scores"]
        paired = summarize_paired(reference, candidate)
        summary = {"status": "completed", "code_commit": self.commit,
                   "selected_candidate": winner, "selection_ranking": ranking,
                   "split": "valid", "validation_documents": 710,
                   "reference_seeds": reference, "candidate_seeds": candidate,
                   "candidate_single_joint_checkpoint_seeds": single,
                   "paired_comparison": paired, "standard_deviation_ddof": 1,
                   "task_combined_protocol": "separate full checkpoint selected for each task",
                   "paper_test_reference_only": self.plan["paper_test_f1"],
                   "strict_paper_test_reproduction_established": False,
                   "all_four_validation_mean_deltas_nonnegative":
                   all(paired[t]["paired_mean_delta"] >= 0 for t in TASKS),
                   "all_four_paired_ci_lower_bounds_nonnegative":
                   all(paired[t]["paired_95pct_ci"][0] >= 0 for t in TASKS),
                   "uploaded": False, "visibility": "private"}
        atomic_json(self.output / "campaign_summary.json", summary)
        lines = ["# Completed reproduction matrix", "",
                 "All scores below are five-seed validation results on 710 documents.",
                 "The paper's hidden-test results are a separate reference; no test score is claimed.", "",
                 "Task-selected combinations and single shared checkpoints are recorded separately in JSON.", "",
                 "| F1 | Author-code reference mean ± sample SD | Selected candidate mean ± sample SD | Paired delta [95% CI] |",
                 "|---|---:|---:|---:|"]
        for task, values in paired.items():
            lo, hi = values["paired_95pct_ci"]
            lines.append(f"| {task} | {values['reference_mean']:.3f} ± {values['reference_sample_std']:.3f} | "
                         f"{values['candidate_mean']:.3f} ± {values['candidate_sample_std']:.3f} | "
                         f"{values['paired_mean_delta']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        (self.output / "COMPARISON.md").write_text("\n".join(lines) + "\n")
        self.state.update(status="completed", summary_path=str(self.output / "campaign_summary.json"))
        self.state.pop("active_run", None)
        self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGINT, stop_requested)
    root = Path(__file__).resolve().parents[1]
    with args.gpu_lock.resolve().open("a+") as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = Campaign(root, args.plan_file.resolve(), args.output_dir.resolve(), sys.executable)
        try:
            campaign.execute()
        except Exception as exc:
            campaign.state.update(status="interrupted" if STOP else "failed", error=str(exc))
            campaign.save()
            raise


if __name__ == "__main__":
    main()
