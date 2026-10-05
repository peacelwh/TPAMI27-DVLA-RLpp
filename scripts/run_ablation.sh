#!/usr/bin/env bash
# Component controls of Table 4 and internal controls of Table 5 (5-way 1-shot).
set -euo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-0}
DATASET=${DATASET:-miniImageNet}
run() { NAME=$1; shift; python scripts/train.py --dataset "$DATASET" --shot 1 --name "$NAME" --gpu "$GPU" "$@"; \
        python scripts/test.py --dataset "$DATASET" --shot 1 --name "$NAME" --gpu "$GPU"; }

run csp_only            --no_cfg
run cfg_only            --no_csp
run uniform_allocation  --uniform_allocation
run positive_only       --positive_only
run no_neutral          --no_neutral
run static_gate         --gate static
run fixed_margin        --fixed_margin
run no_intrinsic_fallback --no_intrinsic_fallback
run state_baseline_only --state_baseline_only
run no_nuisance_cost    --gamma 0
