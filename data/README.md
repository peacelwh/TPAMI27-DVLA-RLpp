# Data

Raw data, semantic banks and checkpoints are **not tracked** by git. Place them here following the layout below.

```
data/
├── datasets/<dataset>/
│   ├── base/<class>/*.jpg      training categories
│   ├── val/<class>/*.jpg       validation categories (model selection, hyper-parameters)
│   └── novel/<class>/*.jpg     test categories
├── semantic/<dataset>_banks.pth           intrinsic / nuisance banks (scripts/generate_semantics.py)
├── checkpoint/visformer-<dataset>.pth     pre-trained Visformer-Tiny (scripts/pretrain.py)
├── checkpoint/dvla_rl-<dataset>-<shot>shot.pth   optional DVLA-RL checkpoint for the frozen reference gate
└── prompts/intrinsic_nuisance_prompt.txt  generator prompt
```

## Datasets

| Name | Benchmark | Notes |
|:--|:--|:--|
| `miniImageNet`, `tieredImageNet`, `CIFAR-FS`, `FC100` | general FSL | ImageFolder splits of DVLA-RL / VT-FSL |
| `FG-CUB`, `FG-Cars`, `FG-Dogs` | fine-grained FSL | CUB folders `001.Black_footed_Albatross`, Dogs folders `n02085620-Chihuahua` are mapped to clean class names |
| `CD-CUB`, `Places`, `ChestX` | cross-domain FSL | only the `novel` split is used, with a model trained on miniImageNet |

Download links are given in the top-level README. Base, validation and test categories are disjoint; class splits
follow the standard protocols of each benchmark. Do not place personal or sensitive data in this directory.

## Semantic banks

`<dataset>_banks.pth` is a dictionary documented in `src/dvla_rlpp/data/semantic.py`. It stores, for every class,
unit-normalised CLIP text embeddings of intrinsic local phrases, an intrinsic global description, nuisance local
phrases, a nuisance global description (both nuisance fields may be empty) and the encoded class-name template used
as the intrinsic fallback. Optional per-support entries are keyed by the relative image path. The `meta` field
records the prompt hash, generator, text encoder and seed so that a cache entry is fully reproducible.

## Checkpoints

* `visformer-<dataset>.pth`: `{"state_dict": ...}` of the pre-trained backbone.
* `dvla_rl-<dataset>-<shot>shot.pth`: a DVLA-RL meta-tuned checkpoint. Only its gate parameters
  (`rl_gate1.*`, `rl_gate2.*`, `t2i`, `t2i2`) are copied into the frozen reference policy.
