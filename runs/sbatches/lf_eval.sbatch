#!/bin/bash
#SBATCH --job-name=lf_eval
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
# Evaluation of a policy of lf_pipeline.sbatch: sPCE (T = 30, L = 5e5, 4096 runs; random designs as the baseline)
# and Figures 2 and 5. Submit from the repository root, or right after the training job:
#   RUN=sigma1 POLICY=detach sbatch runs/sbatches/lf_eval.sbatch
#   RUN=sigma1 POLICY=detach sbatch --dependency=afterok:<training job id> runs/sbatches/lf_eval.sbatch
set -eo pipefail

RUN=${RUN:-lf_attn}
POLICY=${POLICY:-policy}

OUT=checkpoints/runs/$RUN
mkdir -p $OUT/figures

python runs/scripts/evaluation/eval_lf_spce.py --flow-ckpt $OUT/guidance.pt --policy $POLICY=$OUT/$POLICY.pt \
    | tee $OUT/spce_$POLICY.txt
python runs/scripts/figures/lf_rollout_figures.py --flow-ckpt $OUT/guidance.pt --policy $POLICY=$OUT/$POLICY.pt \
    --out $OUT/figures
