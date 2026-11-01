#!/bin/bash
#SBATCH --job-name=lf_pipeline
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
##SBATCH --partition=gpu         # set for the cluster (remove one '#')
# Location finding end to end on one GPU, one stage after the other:
#   1) prior flow                              runs/scripts/training/train_lf_flow.py, PRIOR_CONFIG
#   2) history encoder + guidance on the prior train_lf_flow.py --init-prior,       GUIDANCE_CONFIG
#   3) decoupled policy on the frozen flow     train_lf_policy.py,                   POLICY_CONFIG
# A stage that finishes leaves <stage>.done next to its checkpoint; resubmitting the job (e.g. after the time limit)
# skips the finished stages, and an interrupted stage starts over. Several policies on one guidance: submit the first
# job, then the others with --dependency=afterok:<its job id> (or once guidance.done exists), each with its own
# POLICY_NAME; jobs that train the same prior or guidance at the same time would overwrite each other.
#
# Submit from the repository root, with the Python environment active (the job inherits it):
#   mkdir -p logs && sbatch runs/sbatches/lf_pipeline.sbatch
#   RUN=sigma1 GUIDANCE_ARGS="--sigma 1.0" POLICY_NAME=finalrandq POLICY_ARGS="--policy-term final_random" sbatch ...
#   RUN=sigma1 POLICY_NAME=greedy SEED=43 POLICY_ARGS="--greedy" sbatch runs/sbatches/lf_pipeline.sbatch
# Settings (environment variables, [default]):
#   RUN              run name; checkpoints in $OUT_ROOT/$RUN: prior.pt, guidance.pt, <policy>.pt + .csv  [lf_attn]
#   OUT_ROOT         [checkpoints/runs]
#   PRIOR_CONFIG     [runs/configs/lf/prior.yaml]
#   GUIDANCE_CONFIG  [runs/configs/lf/flow_attn.yaml]  (sigma and m of the outcome model are set here)
#   POLICY_CONFIG    [runs/configs/lf/policy_div.yaml]
#   POLICY_NAME      [policy_div]; <policy> = POLICY_NAME, plus _s$SEED if SEED is set
#   SEED             policy seed [the config's]
#   PRIOR_ARGS, GUIDANCE_ARGS, POLICY_ARGS   extra arguments of a stage, they override its config file
#                    (policy variants: --policy-term final_random, --detach-policy-input, --greedy)
#   PYTHON           [python]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"

RUN=${RUN:-lf_attn}
D=${OUT_ROOT:-checkpoints/runs}/$RUN
T=runs/scripts/training
PRIOR_CONFIG=${PRIOR_CONFIG:-runs/configs/lf/prior.yaml}
GUIDANCE_CONFIG=${GUIDANCE_CONFIG:-runs/configs/lf/flow_attn.yaml}
POLICY_CONFIG=${POLICY_CONFIG:-runs/configs/lf/policy_div.yaml}
SEED=${SEED:-}
PRIOR_ARGS=${PRIOR_ARGS:-} GUIDANCE_ARGS=${GUIDANCE_ARGS:-} POLICY_ARGS=${POLICY_ARGS:-}
PY=${PYTHON:-python}
POLICY=${POLICY_NAME:-policy_div}${SEED:+_s$SEED}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "$D"
echo "run $D, commit $(git rev-parse --short HEAD 2>/dev/null || echo '?'), host $(hostname), $(date -u)"

stage() {  # name, command ...
  local name=$1; shift
  if [ -f "$D/$name.done" ]; then echo "== $name: finished earlier, skipped"; return; fi
  echo "== $name ($(date -u)): $*"
  "$@"
  touch "$D/$name.done"
  echo "== $name finished $(date -u)"
}

stage prior "$PY" -W ignore $T/train_lf_flow.py --config "$PRIOR_CONFIG" $PRIOR_ARGS --out "$D/prior.pt"
stage guidance "$PY" -W ignore $T/train_lf_flow.py --config "$GUIDANCE_CONFIG" --init-prior "$D/prior.pt" \
  $GUIDANCE_ARGS --out "$D/guidance.pt"
stage "$POLICY" "$PY" -W ignore $T/train_lf_policy.py --config "$POLICY_CONFIG" --flow-ckpt "$D/guidance.pt" \
  ${SEED:+--seed $SEED} $POLICY_ARGS --out "$D/$POLICY.pt" --log-csv "$D/$POLICY.csv"
echo "all stages done $(date -u)"
