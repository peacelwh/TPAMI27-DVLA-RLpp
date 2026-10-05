#!/usr/bin/env bash
# Full pipeline on miniImageNet: pre-training, semantic generation, meta-tuning and testing.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-0}
DATASET=${DATASET:-miniImageNet}

# 1) pre-train the Visformer-Tiny backbone on the base split (800 epochs; 300 for tieredImageNet)
python scripts/pretrain.py --dataset "$DATASET" --epoch 800 --batch_size 512 --gpu "$GPU"
cp "results/pretrain/$DATASET/visformer-t/best-1shot.pth" "data/checkpoint/visformer-$DATASET.pth"

# 2) generate intrinsic / nuisance semantic banks from labeled supports
python scripts/generate_semantics.py --dataset "$DATASET" --splits base,val,novel --gpu "$GPU"

# 3) episodic meta-tuning and testing
for SHOT in 1 5; do
  python scripts/train.py --dataset "$DATASET" --shot "$SHOT" --epoch 100 --gpu "$GPU"
  python scripts/test.py  --dataset "$DATASET" --shot "$SHOT" --episode 2000 --gpu "$GPU"
done
