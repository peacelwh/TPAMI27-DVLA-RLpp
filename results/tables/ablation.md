# Ablation and mechanism verification

5-way 1-shot accuracy (%) with matched semantic caches, training budgets, five training seeds and 2,000 evaluation
episodes per seed. *Clean* and *Shift* use matched queries before and after a context intervention that keeps the
foreground object and its label fixed while replacing the background or an incidental co-occurring object.

## Component controls (Table 4)

CSP: complementary semantic purification (`--no_cfg` keeps it alone). CFG: counterfactual gating (`--no_csp` keeps it alone).

| Variant | CSP | CFG | miniImageNet Clean | miniImageNet Shift | CUB Clean | CUB Shift |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| DVLA-RL | ✗ | ✗ | 81.7 | 75.8 | 91.9 | 85.0 |
| + CSP | ✓ | ✗ | 82.6 | 79.4 | 93.4 | 89.4 |
| + CFG | ✗ | ✓ | 82.4 | 78.4 | 93.0 | 88.1 |
| **DVLA-RL++** | ✓ | ✓ | **83.4** | **81.2** | **94.3** | **92.0** |

## Internal purification and policy controls (Table 5)

| Variant | Flag | miniImageNet Clean | miniImageNet Shift | CUB Clean | CUB Shift |
|:--|:--|:--:|:--:|:--:|:--:|
| DVLA-RL | | 81.7 | 75.8 | 91.9 | 85.0 |
| Uniform allocation | `--uniform_allocation` | 82.2 | 77.6 | 92.8 | 87.2 |
| Positive semantics only | `--positive_only` | 82.7 | 79.0 | 93.5 | 89.1 |
| No neutral option | `--no_neutral` | 83.0 | 79.7 | 93.8 | 90.0 |
| Static gate | `--gate static` | 82.8 | 79.8 | 93.6 | 90.0 |
| Fixed margin | `--fixed_margin` | 83.1 | 80.1 | 94.0 | 90.6 |
| No intrinsic fallback | `--no_intrinsic_fallback` | 82.9 | 79.6 | 93.7 | 89.8 |
| State baseline only | `--state_baseline_only` | 83.0 | 80.4 | 94.0 | 90.9 |
| No nuisance cost | `--gamma 0` | 83.1 | 80.1 | 94.0 | 90.5 |
| **DVLA-RL++** | | **83.4** | **81.2** | **94.3** | **92.0** |

## Parameter sensitivity (miniImageNet, 5-way 1-shot)

| τ_s | 0.02 | 0.05 | 0.1 | **0.2** | 0.5 | 1.0 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Accuracy | 79.8 | 81.6 | 83.1 | **83.4** | 83.0 | 82.2 |

Accuracy stays within 0.3 points of the default for λ in [0.5, 0.9] and within 0.2 points for γ in [0.1, 0.75].

## Policy diagnostics (miniImageNet)

| Estimator | Gradient-covariance trace (relative) |
|:--|:--:|
| Raw returns | 1.25 |
| State baseline | 1.00 |
| Paired returns + state baseline (ours) | 0.60 |

Frequency of C(a) > C(a⁰): 24% → 9%. Active PPO clipping after policy updates: 22% → 15%.
