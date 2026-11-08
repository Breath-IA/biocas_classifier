# SPRSound respiratory-sound classification: models, results and comparison with the state of the art

Summary of experiments `experiments/gate1_normal_vs_abnormal/e01` and `experiments/gate2_event_type/e01–e08`, compared with [SPRSound_SOTA_summary.md](SPRSound_SOTA_summary.md).
Numbers are copied from the experiment reports (`results/report.md` of each experiment, `e06/cv_summary.csv`, `e07/` and `e08/` CSV files); nothing was re-run for this document.

## 1. Setting

- **Data:** SPRSound / BioCAS (pediatric lung sounds, 8 kHz), event level. Classes used: Normal (N), Wheeze (W), Fine crackle (FC).
- **Plan (`plan.md`):** two gates. *Gate 1*: normal vs abnormal. *Gate 2*: type of abnormal sound (W vs FC), evaluated in isolation (gate 1 assumed perfect).
- **Validation:** leave-one-year-out (2022–2025), patients disjoint between years. The held-out year is never augmented nor used for early stopping or thresholds. **2025 is the official test split and the hardest year** (annotation style changed, 86 % of its W/FC events are FC).
- **Metrics:** ROC AUC (threshold-free), balanced accuracy, macro-F1 and per-class recall.

## 2. Models and results

| Experiment | Model | How it works | Result |
|---|---|---|---|
| Gate 1 · e01 | Random Forest on hand-made spectral features | Describes each event with ~65 numbers (band energy, spikiness, tonality) and classifies them | **Normal vs abnormal: AUC 0.78 mean, only 0.63 in 2025.** Weak link of the project |
| Gate 2 · e01 | Random Forest, N / W / FC in one step, with audio augmentation | Same features, 3 classes | Balanced accuracy 0.46–0.50 in 2025. The bottleneck is N vs FC (AUC 0.60). Augmentation gave no gain; sub-sampling N did |
| Gate 2 · e02 | Random Forest / logistic regression, W vs FC | Same idea without N | **AUC 0.90 mean, 0.81 in 2025.** Logistic regression matches the forest in AUC and is better calibrated |
| Gate 2 · e03 | CNN, 4 blocks, 97 k parameters | Reads a log-mel image (time × frequency) and learns its own filters | **AUC 0.923 mean, 0.863 in 2025.** Better than every hand-made-feature model |
| Gate 2 · e03 | CNN + physical features | Adds 28 features (e.g. crackle decay time) to the CNN embedding | AUC 0.927: gain of +0.005, within noise |
| Gate 2 · e04 | Synthetic data generator | Inserts parametric crackles/wheezes into real normal recordings, in the dataset's own format | No model trained; realism check: real vs synthetic still distinguishable (probe AUC 0.84–0.92) |
| Gate 2 · e05 | CNN pretrained on synthetic data, then fine-tuned on real data | Learns "crackle vs tone" from synthetic events first | **AUC 0.931 mean, 0.878 in 2025** (+0.008, positive in 11 of 12 year × seed pairs). Mixing synthetic data into real batches *hurts* (0.905–0.911) |
| Gate 2 · e06 | Zoo of 8 CNNs (GPU) | Compares architectures and sizes | Table below |
| Gate 2 · e07, e08 | Top 3 models, shrunk from 1 M to ~0.07 M parameters | Tests how small the network can get; e08 repeats it with other seeds and tests the trend | AUC barely changes with size; no significant trend in any model or seed |

### e06: the eight CNNs (mean over held-out years 2023–2025, real data only)

| Network | What it is | Parameters | Test AUC | Macro-F1 |
|---|---|---|---|---|
| `cnn_deep` | Deeper plain CNN | 0.60 M | **0.937** | **0.839** |
| `resnet18` | Standard residual network (ImageNet design) | 11.2 M | 0.935 | 0.759 |
| `cnn_wide` | Wider plain CNN | 0.40 M | 0.929 | 0.691 |
| `cnn_baseline` | The e03 CNN | 0.11 M | 0.928 | 0.798 |
| `resnet_small` | Small residual CNN | 0.32 M | 0.926 | 0.829 |
| `crnn` | CNN followed by a GRU (recurrent layer) | 0.39 M | 0.924 | 0.834 |
| `cnn_se` | CNN with channel attention (squeeze-and-excitation) | 0.12 M | 0.924 | 0.757 |
| `efficientnet_b0` | Efficient mobile network | 4.0 M | 0.902 | 0.790 |

Ranking by macro-F1 on fold 2025 only (used to pick the models for e07/e08): `efficientnet_b0` 0.746, `crnn` 0.737, `cnn_deep` 0.735.

### e07 / e08: reducing the parameters (fold 2025, 7 sizes from ~1 M to ~0.07 M)

| Model | Test AUC range over sizes | Macro-F1 range over sizes | Trend test |
|---|---|---|---|
| `cnn_deep` | 0.85–0.88 | 0.54–0.77 | no significant trend |
| `crnn` | 0.84–0.87 | 0.65–0.77 | no significant trend |
| `efficientnet_b0` | 0.81–0.85 | 0.72–0.74 | no significant trend |

## 3. Analysis

1. **A CNN beats hand-made features for W vs FC.** AUC rises from 0.81 to 0.86–0.88 in 2025, and the CNN does not rely on event duration as a shortcut (checked by duration strata).
2. **Network size matters little.** The 0.11 M CNN is almost as good as the 11 M ResNet-18 in AUC, and `cnn_deep` / `crnn` keep AUC ≈ 0.85–0.88 down to ~75 k parameters. `efficientnet_b0` is the weakest and noisiest.
3. **Macro-F1 is unreliable in this problem.** For the same model it varies 0.54–0.77 across seeds and sizes while AUC barely moves. The cause is the fixed 0.5 threshold: in 2025, 86 % of events are FC, so a small threshold shift changes F1 a lot. Rank models by AUC and set the threshold on validation data. Synthetic pretraining also stabilised the threshold (balanced-accuracy spread across seeds 0.078 → 0.011).
4. **2025 is a shifted domain.** The spectral difference between FC and W disappears and annotation windows lengthened; every model loses 0.06–0.08 AUC there. This, not model capacity, is the main limit.
5. **Synthetic data helps only as pretraining.** Mixing it into real batches lowers AUC (a classifier tells synthetic from real easily, so the CNN can learn artefacts); pretraining then fine-tuning on real data gives a small but consistent gain.
6. **Physical features add almost nothing to the CNN** (+0.005 AUC); the decay-time features are the only ones that matter.
7. **Gate 1 is the open problem.** Normal vs FC is hard (AUC ≈ 0.60 with features), and no CNN has been trained for gate 1 yet, so the full cascade has not been evaluated.

### Limitations

- Effects of ≈ 0.01 AUC are the same size as seed noise (3 seeds in e03/e05, 2–3 in e07/e08).
- Configurations were compared on the same four folds used to choose them: there is no untouched test set.
- e07/e08 use a single fold (2025, a few hundred W events), so their p-values and the size-trend verdicts are rough.
- Only W vs FC was trained with CNNs; CC, WC, R and S were not used.

## 4. Comparison with the state of the art

**Not directly comparable.** The literature reports the BioCAS Score (mean of AS and HS from sensitivity and specificity) on whole tasks (1-1, 1-2, 2-1, 2-2) with different test sets; this project reports AUC and per-class metrics for isolated steps on the official 2025 year.

| | State of the art (from `SPRSound_SOTA_summary.md`) | This project |
|---|---|---|
| Normal vs abnormal (task 1-1) | Score 81.0 (spatio-temporal CNN) up to **89–91** (CNN + Transformer hybrids) | Only the feature-based forest was evaluated: AUC 0.63 in 2025. **Far behind** |
| 7-class event typing (task 1-2) | Score 66.7 up to **83–84** (hybrids) | Not attempted. Flat N/W/FC forest: balanced accuracy 0.46–0.50 in 2025 |
| W vs FC | Not published as a separate task. Closest: lightweight Mamba-KAN, 4 classes incl. Normal: F1 wheeze 0.42, crackle 0.58, macro 0.48 | AUC 0.87–0.89 in 2025; F1 for W ≈ 0.52–0.58 and FC ≈ 0.92 (no N class, 86 % FC prevalence) |
| Model size | Hybrids with Transformers are large; Mamba-KAN is the lightweight reference | **0.1–0.6 M parameters, works down to ~75 k.** Strongest point for edge / FPGA use |

**Conclusion.** The W-vs-FC step is at a good level, and a small CNN with synthetic pretraining is a sound recipe. To compare with the literature, the next steps are: (1) train a CNN for gate 1, where the project is furthest behind; (2) evaluate the whole cascade with the official Score (SE, SP, AS, HS); (3) choose the decision threshold on validation data, because the fixed 0.5 threshold makes macro-F1 unstable.

## 5. Where things are

| Item | Path |
|---|---|
| Run order and old → new experiment names | [experiments/README.md](experiments/README.md) |
| Reports | `experiments/*/*/results/report.md` |
| CNN zoo results | `experiments/gate2_event_type/e06_cnn_zoo_colab/cv_summary.csv`, `cv_per_fold.csv` |
| Parameter-reduction results | `experiments/gate2_event_type/e07_param_reduction_colab/`, `e08_param_reduction_new_seeds/` |
| GPU search pipeline (not yet run to final results) | [full_model/](full_model/) |
