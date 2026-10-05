#!/usr/bin/env bash
# Meta-tune and test on every in-domain benchmark, then run the cross-domain transfers.
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-0}

for DATASET in miniImageNet tieredImageNet CIFAR-FS FG-CUB FG-Cars FG-Dogs; do
  for SHOT in 1 5; do
    python scripts/train.py --dataset "$DATASET" --shot "$SHOT" --gpu "$GPU"
    python scripts/test.py  --dataset "$DATASET" --shot "$SHOT" --gpu "$GPU"
  done
done

# cross-domain: train on miniImageNet, test on CUB / Places / ChestX
for TARGET in CD-CUB Places ChestX; do
  for SHOT in 1 5; do
    python scripts/test.py --dataset miniImageNet --test_dataset "$TARGET" --shot "$SHOT" --gpu "$GPU"
  done
done

python scripts/collect_results.py
