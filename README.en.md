# MAVEN-ERE: completed reproduction and improvement study

All15 prescribed formal runs completed100 actual epochs each (1500total). Fixed winner positive2 five-seed CPU combination Overall **51.5469 ±0.3519**, reference **49.7728 ±0.3367** (mean±sampleSD,n−1); paired delta **+1.7741**, 95%tCI **[+1.0901,+2.4581]**. Single and combination results, all4coreference and3relation F1s and selection/padding/device differences are reported separately. This is public validation, not paper Tables7/8 hidden-test reproduction.

[Final report](docs/FINAL_REPRODUCTION_REPORT.md) · [Five-seed CPU data](results/final-campaign/five_seed_cpu.json) · [All15 run metrics and checkpoint hashes](results/final-campaign/formal_runs.json) · [Formal training protocol](docs/REPRODUCTION_PROTOCOL.md) · [Author runner](docs/UPSTREAM_REFERENCE.md)

Positive4 seed42 CPU Overall **52.14502736**, +0.14349698 points versus preceding positive2 CPU52.00153038 on the same710-document valid protocol. Three independent complete joint encoders, three forwards per document. Coreference uses one new full model; temporal/causal share another; subevent reuses the existing positive4 joint single. No head splicing or averaging. Both combinations use CPUFP32, batch8, default trimming. All task declines are disclosed. The strongest single remains positive4 CPU51.78221109, one encoder/all heads/one forward. Fixed five-seed winner remains positive2; positive4 is a posthoc single-seed release review.

| F1 % | Positive2 CPU | Positive4 CPU | Delta points |
|---|---:|---:|---:|
| MUC | 79.56058589 | 78.52802161 | -1.03256428 |
| B3 | 97.78058903 | 97.73238598 | -0.04820306 |
| CEAF-e | 97.47092468 | 97.37842644 | -0.09249825 |
| BLANC | 88.50683704 | 87.95527399 | -0.55156305 |
| Coreference mean | 90.82973416 | 90.39852700 | -0.43120716 |
| Temporal | 54.42049630 | 53.13718065 | -1.28331565 |
| Causal | 32.81518762 | 33.85214008 | +1.03695246 |
| Subevent | 29.94070344 | 31.19226171 | +1.25155827 |
| Overall | 52.00153038 | 52.14502736 | +0.14349698 |

[Positive4 inference commands](docs/POSITIVE4_TASK_COMBINATION.md) · [Positive2 commands](docs/POSITIVE2_TASK_COMBINATION.md)

[Models](https://huggingface.co/MaYiding/maven-ere-roberta-base-joint) · [Existing licensed data mirror](https://huggingface.co/datasets/MaYiding/MAVEN-ERE). The root loads the old48.71379377 baseline; select a documented child or combination. Inputs require gold event mentions, TIMEX and token locations. GPL-3.0/RoBERTaMIT/legacy-optimizer notices and MAVEN/WikipediaCCBY-SA/all4480source attributions are retained. No dataset duplicate, test score or extra positive4 five-seed training is claimed. Historical revisions and checkpoints remain available.
