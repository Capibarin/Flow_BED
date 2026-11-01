#!/bin/bash
# Grid: m = 1e-2, sigma in {0.5, 1, 2} x 4 policy variants, each policy evaluated. Submits every job at once; each
# starts when its input exists (one prior -> a guidance per sigma -> its policies -> their evaluations).
# From the repository root:  bash runs/sbatches/lf_grid.sh
set -e

M=1e-2
SIGMAS="0.5 1 2"
declare -A VARIANTS=(
  [sum_detach]="--detach-policy-input"
  [sum]=""
  [finalrandq]="--policy-term final_random"
  [greedy]="--greedy"
)

mkdir -p logs
prior=$(sbatch --parsable runs/sbatches/lf_prior.sbatch)
for s in $SIGMAS; do
  RUN=m${M}_sigma$s
  guidance=$(RUN=$RUN FLOW_ARGS="--sigma $s --m $M" \
    sbatch --parsable --dependency=afterok:$prior runs/sbatches/lf_guidance.sbatch)
  for v in "${!VARIANTS[@]}"; do
    policy=$(RUN=$RUN POLICY=$v POLICY_ARGS="${VARIANTS[$v]}" \
      sbatch --parsable --dependency=afterok:$guidance runs/sbatches/lf_policy.sbatch)
    RUN=$RUN POLICY=$v sbatch --dependency=afterok:$policy runs/sbatches/lf_eval.sbatch
  done
done
