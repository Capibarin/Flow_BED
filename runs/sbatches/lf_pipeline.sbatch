#!/bin/bash
#SBATCH --job-name=lf_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
# Location finding: prior -> encoder + guidance -> policy. Submit from the repository root (mkdir -p logs first):
#   sbatch runs/sbatches/lf_pipeline.sbatch
#   RUN=sigma1 FLOW_ARGS="--sigma 1.0" POLICY=detach POLICY_ARGS="--detach-policy-input" \
#       sbatch runs/sbatches/lf_pipeline.sbatch
set -e

RUN=${RUN:-lf_attn}            # checkpoints go to checkpoints/runs/$RUN
FLOW_ARGS=${FLOW_ARGS:-}       # extra arguments of the guidance stage, e.g. --sigma 1.0 --m 1e-3
POLICY=${POLICY:-policy}       # name of the policy checkpoint
POLICY_ARGS=${POLICY_ARGS:-}   # extra arguments of the policy, e.g. --detach-policy-input --seed 43

OUT=checkpoints/runs/$RUN
mkdir -p $OUT

python runs/scripts/training/train_lf_flow.py --config runs/configs/lf/prior.yaml --out $OUT/prior.pt
python runs/scripts/training/train_lf_flow.py --config runs/configs/lf/flow_attn.yaml --init-prior $OUT/prior.pt \
    $FLOW_ARGS --out $OUT/guidance.pt
python runs/scripts/training/train_lf_policy.py --config runs/configs/lf/policy_div.yaml --flow-ckpt $OUT/guidance.pt \
    $POLICY_ARGS --out $OUT/$POLICY.pt --log-csv $OUT/$POLICY.csv
