#!/bin/bash
#SBATCH --job-name=lf_guidance
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# Location finding, stage 2: encoder + guidance on the shared prior of lf_prior.sbatch. Submit from the repository root:
#   RUN=sigma1 FLOW_ARGS="--sigma 1.0" sbatch runs/sbatches/lf_guidance.sbatch
set -e

RUN=${RUN:-lf_attn}            # checkpoints go to checkpoints/runs/$RUN
FLOW_ARGS=${FLOW_ARGS:-}       # extra arguments, e.g. --sigma 1.0 --m 1e-3

OUT=checkpoints/runs/$RUN
mkdir -p $OUT

python runs/scripts/training/train_lf_flow.py --config runs/configs/lf/flow_attn.yaml \
    --init-prior checkpoints/runs/prior/prior.pt $FLOW_ARGS --out $OUT/guidance.pt
