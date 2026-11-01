#!/bin/bash
#SBATCH --job-name=lf_policy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
# Location finding, stage 3: a policy on the frozen guidance of lf_guidance.sbatch. Submit from the repository root:
#   RUN=sigma1 POLICY=detach POLICY_ARGS="--detach-policy-input" sbatch runs/sbatches/lf_policy.sbatch
set -e

RUN=${RUN:-lf_attn}            # the guidance is checkpoints/runs/$RUN/guidance.pt
POLICY=${POLICY:-policy}       # name of the policy checkpoint
POLICY_ARGS=${POLICY_ARGS:-}   # extra arguments, e.g. --detach-policy-input --seed 43

OUT=checkpoints/runs/$RUN

python runs/scripts/training/train_lf_policy.py --config runs/configs/lf/policy_div.yaml --flow-ckpt $OUT/guidance.pt \
    $POLICY_ARGS --out $OUT/$POLICY.pt --log-csv $OUT/$POLICY.csv
