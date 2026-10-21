# MAVEN-ERE: completed 15-run reproduction and improvement study

All 15 prescribed formal runs completed **100 actual epochs each (1,500 epochs total)**. The five-seed CPU task-selected combination comparison is **49.7728 ± 0.3367** for the author-code reference and **51.5469 ± 0.3519** for fixed winner positive2 (mean ± sample SD). The matched-seed overall difference is **+1.7741 F1 points**, paired 95% t interval **[+1.0901, +2.4581]**. These are scores on the same 710-document public validation set.

The separately reported single-checkpoint comparison is descriptive: the reference single is the best overall among four retained task-selected complete checkpoints, while the winner single was selected by overall validation F1 across its validation history. The selection scopes differ. Neither comparison establishes paper hidden-test reproduction or an isolated causal effect of class weighting.

## Data and evaluation

MAVEN-ERE 1.0 contains 2,913 train, 710 valid and 857 test documents. The original split JSONL bytes were retained, and split document IDs were checked disjoint. Inputs include gold event mentions, token locations and annotated temporal expressions. This system predicts coreference and temporal, causal and subevent relations; it does not discover event mentions from raw text.

The official raw-direction evaluator scores all validation documents. Training symmetrizes SIMULTANEOUS and BEGINS-ON labels; those augmented-label training metrics are recorded separately and never substituted for official results. Coreference is reported as MUC, B3, CEAF-e and BLANC. Coreference mean is their arithmetic mean; Overall is the arithmetic mean of coreference mean, temporal F1, causal F1 and subevent F1. All numbers below are percentages or percentage-point deltas.

No test score is available or claimed. The paper Tables 7/8 report five-run hidden-test results; these public-valid measurements use a different split and cannot verify equality with those paper results.

## Formal experiment matrix and implementation differences

The frozen training source is `8a49171cfe160d85877ffea5e7247c2141bcf3d4`; the extracted author source is `ac81a9711a69f43f55bfbc50b3bb573fd11c64b0`. All formal runs used RoBERTa-base, FP32, microbatch 8, accumulation 4 and 100 epochs with no early stopping. Independent experiments ran on three RTX4090D GPUs; each experiment used one GPU. Parallel execution did not introduce DDP, change batch size or shorten a run. Final reload comparisons used CPU FP32, batch 8 and four threads.

The author runner preserves original dummy-label/coreference-loss behavior, legacy AdamW semantics, the scheduler denominator based on microbatches, and accumulation across epoch boundaries. Compatibility patches address current libraries, local paths, eager attention and logging without silently replacing that training algorithm. The candidate driver corrects dummy-label loss, uses its documented scheduler, flushes accumulation at each epoch end, and implements its documented gradient handling and resume logic. These implementation changes belong to the tested treatment and prevent attribution of the entire gain to one class-weight change.

The reference has 36,500 microsteps, 9,125 optimizer updates, 182 training validations and four final selected-checkpoint validations per run. Candidates have 36,500 microsteps, 9,200 updates and 184 validations at updates 50, 100, …, 9200. Thus the epoch budget matches while update and selection-opportunity counts differ. The public code uses a 200-wide, three-Linear-layer scorer; the paper appendix states width 150 and a two-layer description. The width 150/3 and width 150/2 candidates probe these alternatives without claiming the latter is the author’s actual hidden implementation.

Reference prediction uses `--reference-padding`; candidates use default padding trimming. CPU and saved GPU results are retained separately. Public CLI and independent AutoModel CPU predictions and metrics match exactly within each checked package. Small CPU/GPU prediction differences are recorded, including unselected fields of retained complete models; GPU logits and margins were unavailable, so no exact numerical cause is asserted. No decoding threshold or prediction-comparison tolerance was changed.

See [the detailed protocol](REPRODUCTION_PROTOCOL.md) and [author runner compatibility changes](UPSTREAM_REFERENCE.md).

## Candidate selection

All six seed 42 candidates were predetermined. The primary selection rule maximizes the minimum of the four task F1 deltas relative to reference42, then Overall, then fixed candidate order. It selects **positive2-256** (minimum task delta +0.048999666), ahead of positive4-256 (minimum −0.382207493), although positive4 has higher seed 42 combination Overall. The fixed five-seed winner was never changed. Seed42 participated in selection, so the five seeds are not five untouched model-selection holdouts.

The following are original GPU validation results. They are separate from actual CPU package measurements. Every row represents a completed100-epoch run; “single” means one complete joint checkpoint and one forward, and “combination” means task fields from distinct complete models.

| Candidate | Chunk | Width | Linear layers | Positive weight | GPU single Overall | GPU combination Overall | Combination models |
| --- | --- | --- | --- | --- | --- | --- | --- |
| corrected-256 | 256 | 200 | 3 | 1.0 | 47.917987 | 49.433824 | 4 |
| corrected-512 | 512 | 200 | 3 | 1.0 | 49.469636 | 50.199116 | 4 |
| width 150-256 | 256 | 150 | 3 | 1.0 | 47.961230 | 49.727828 | 4 |
| appendix-two-layer-256 | 256 | 150 | 2 | 1.0 | 47.763558 | 49.296428 | 4 |
| positive2-256 | 256 | 200 | 3 | 2.0 | 51.499744 | 52.001530 | 3 |
| positive4-256 | 256 | 200 | 3 | 4.0 | 51.782188 | 52.145027 | 3 |

Four task selectors retain complete checkpoints: coreference uses upstream-any improvement among the four component metrics; temporal, causal and subevent use task F1. Coreference historical component maxima are not spliced into a fictitious model score. The joint selector uses Overall. Shared checkpoint identities are evaluated once and reused by their task fields. There is no head splicing, weight averaging or cross-seed voting.

## Matched five-seed CPU results

Seeds are 13, 21, 42, 87 and 100. Each cell is mean ± sample SD with denominator n−1=4. Differences are winner minus reference over matched seeds. Two-sided 95% Student t intervals use df4 and t0.975=2.7764451051977934. They describe training-seed variability on this fixed validation set and these selection procedures, not hidden-test/generalization uncertainty or document resampling. Intervals are per metric, without a familywise multiplicity adjustment.

### Task-selected complete-model combinations

| F1 % | Reference | Positive2 | Paired delta points | Paired95%CI |
| --- | --- | --- | --- | --- |
| MUC | 79.4678 ± 0.3506 | 79.4378 ± 0.4285 | -0.0300 | [-0.9286, +0.8686] |
| B3 | 97.7614 ± 0.0342 | 97.7993 ± 0.0618 | +0.0379 | [-0.0711, +0.1468] |
| CEAF-e | 97.4786 ± 0.0362 | 97.5155 ± 0.0770 | +0.0369 | [-0.0801, +0.1539] |
| BLANC | 88.1472 ± 0.0536 | 88.1881 ± 0.2967 | +0.0409 | [-0.3539, +0.4358] |
| Coreference mean | 90.7137 ± 0.0948 | 90.7352 ± 0.1437 | +0.0214 | [-0.2456, +0.2885] |
| Temporal | 53.0483 ± 0.1534 | 54.1355 ± 0.1951 | +1.0872 | [+0.8872, +1.2872] |
| Causal | 31.0940 ± 0.3902 | 32.7377 ± 0.1290 | +1.6438 | [+1.1121, +2.1754] |
| Subevent | 24.2354 ± 0.9987 | 28.5793 ± 1.1264 | +4.3439 | [+2.3036, +6.3841] |
| Overall | 49.7728 ± 0.3367 | 51.5469 ± 0.3519 | +1.7741 | [+1.0901, +2.4581] |

### One complete joint checkpoint, differing selection scopes

| F1 % | Reference | Positive2 | Paired delta points | Paired95%CI |
| --- | --- | --- | --- | --- |
| MUC | 78.0790 ± 0.9060 | 78.6405 ± 0.6910 | +0.5615 | [-0.1888, +1.3118] |
| B3 | 97.6029 ± 0.0939 | 97.6556 ± 0.0623 | +0.0527 | [-0.0513, +0.1567] |
| CEAF-e | 97.2604 ± 0.1523 | 97.3294 ± 0.0978 | +0.0690 | [-0.0936, +0.2317] |
| BLANC | 87.4495 ± 0.4842 | 87.6064 ± 0.3204 | +0.1569 | [-0.6672, +0.9811] |
| Coreference mean | 90.0979 ± 0.3897 | 90.3080 ± 0.2682 | +0.2100 | [-0.2268, +0.6469] |
| Temporal | 50.7717 ± 0.5739 | 53.6336 ± 0.5450 | +2.8619 | [+1.5815, +4.1423] |
| Causal | 30.3483 ± 1.7141 | 31.7229 ± 0.5089 | +1.3746 | [-1.1988, +3.9479] |
| Subevent | 21.3868 ± 2.3178 | 28.5793 ± 1.1264 | +7.1925 | [+3.4023, +10.9827] |
| Overall | 48.1512 ± 0.1583 | 51.0609 ± 0.3872 | +2.9098 | [+2.3237, +3.4959] |

Reference combinations use four complete models/four forwards per document in every seed. Positive2 uses three at seed 42 and four at each other seed. Reference singles are the best overall among the four retained task-selected complete models, not the best over all historical overall evaluations. Positive2 singles are joint-overall-selected checkpoints. This unequal single-selection scope must accompany the single table.

### Per-seed CPU Overall

| Seed | Reference retained single | Positive2 joint single | Reference combination | Positive2 combination | Positive2 combination models |
| --- | --- | --- | --- | --- | --- |
| 13 | 47.99896905 | 50.90612116 | 49.81578470 | 51.43221683 | 4 |
| 21 | 48.17106001 | 51.43998932 | 49.27563341 | 51.79947258 | 4 |
| 42 | 48.08566469 | 51.49974394 | 50.03618141 | 52.00153038 | 3 |
| 87 | 48.41259348 | 50.62988936 | 50.11003022 | 51.11593547 | 4 |
| 100 | 48.08758042 | 50.82891643 | 49.62657234 | 51.38539605 | 4 |

All four coreference metrics and three relation F1s for every CPU seed and both scopes, exact paired differences, selected identities and CPU/GPU metric differences are retained in the accompanying five_seed_cpu.json. Original GPU measurements and all 15 training-run identities/configurations/source, summary, history and selected-artifact hashes are in formal_runs.json. Raw-summary versus campaign arithmetic differences at floating-point rounding precision are retained as separate records.

## Reload verification and releases

Each reference seed passed 12 actual full 710-document passes: legacy CPU, converted CPU and independent AutoModel for each of four complete models; four actual public CLI passes followed, totaling 16 per seed. Positive2 seeds13,21,87,100 each passed 8 actual full 710-document passes, public CLI plus independent AutoModel for four distinct models. Seed42 passed 4 new passes plus 2 reused prior single-model passes for its three models. Reuse required exact checkpoint identity, all8 artifact hashes and complete saved parsed710-document predictions; no validated single inference was repeated. Task combinations were regenerated from complete-model predictions and scored by the official evaluator.

The verified positive2 seed 42 three-model CPU combination is52.00153037883534, versus reference42 four-model CPU50.036181407057285 (+1.965348971778056). It is an optional multi-model inference package, distinct from the positive2 single51.49974394139212. The strongest published single is positive4 CPU51.78221109033361 with one complete encoder/all heads/one forward.

The remaining positive4 seed 42 combination is a postprocessing/release-quality review of an already completed formal run. It does not replace positive2 as the fixed five-seed winner or add another training arm.

### Positive4 seed 42 combination: independently verified CPU result

Actual full 710-document CPU Overall is **52.14502736**, difference **+0.14349698 points** against the preceding positive2 CPU combination. It uses three independent complete encoders and three forwards. Four new full 710-document passes evaluated only two new complete models; the exact previously verified positive4 joint single supplied two reused passes. Full state/prediction/metric/canonical gates passed before this comparison.

| Metric | Preceding positive2 CPU | Positive4 CPU | Delta points |
| --- | --- | --- | --- |
| MUC | 79.56058589 | 78.52802161 | -1.03256428 |
| B3 | 97.78058903 | 97.73238598 | -0.04820306 |
| CEAF-e | 97.47092468 | 97.37842644 | -0.09249825 |
| BLANC | 88.50683704 | 87.95527399 | -0.55156305 |
| Coreference mean | 90.82973416 | 90.39852700 | -0.43120716 |
| Temporal | 54.42049630 | 53.13718065 | -1.28331565 |
| Causal | 32.81518762 | 33.85214008 | +1.03695246 |
| Subevent | 29.94070344 | 31.19226171 | +1.25155827 |
| Overall | 52.00153038 | 52.14502736 | +0.14349698 |

Release artifacts are packaged only after full 710-document model validation. Immutable published revisions, file hashes and anonymous readback receipts are recorded in the accompanying release manifest.

The repository root still loads the original single baseline48.7137937687; select a documented child directory or combination explicitly. Selected weights may come from an early epoch (for example, positive2/positive4 seed 42 epoch 9/update 750) even though the parent run completed100 epochs. This is validation checkpoint selection, not early-stopped training. Older release revisions remain available.

## Provenance, licenses and reproducibility limits

The original short baseline, failed/interrupted attempts, canaries and smoke tests are excluded from the 15 formal runs. Full counter/event/history audits establish every completed run; an inactive service alone was never treated as completion. Original source/data and failed-attempt evidence were preserved. Operational addresses, machine paths, credentials, private orchestration, optimizer/RNG states and caches are excluded from the release.

Training code retains GPL-3.0 and upstream notices, RoBERTa retains MIT attribution, and the legacy optimizer adapter retains its Apache-2.0 notice. The existing public dataset mirror preserves the MAVEN/Wikipedia CC BY-SA notices and all four,480 source attributions; it was not duplicated or reuploaded for this finalization.

The comparison supports improvement under the stated public-validation protocol and selection scopes. It does not prove equality to paper Tables 7/8, performance on hidden test labels, an equal-cost single-model gain for task combinations, or a five-seed positive4 result.

Code: https://github.com/MaYiding/maven-ere-roberta-base-joint  
Models: https://huggingface.co/MaYiding/maven-ere-roberta-base-joint  
Data: https://huggingface.co/datasets/MaYiding/MAVEN-ERE
