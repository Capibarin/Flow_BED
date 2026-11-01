#!/bin/bash
#SBATCH --job-name=lf_prior
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
# Location finding, stage 1: the prior flow (theta ~ N(0, I), so one prior serves every sigma and m).
# Submit from the repository root (mkdir -p logs first):  sbatch runs/sbatches/lf_prior.sbatch
set -e

OUT=checkpoints/runs/prior
mkdir -p $OUT

python runs/scripts/training/train_lf_flow.py --config runs/configs/lf/prior.yaml --out $OUT/prior.pt
