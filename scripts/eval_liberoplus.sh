#!/bin/bash
# Zero-shot LIBERO-Plus evaluation: one process per suite (RLinf reads
# LIBERO_TYPE once per process), all perturbation categories.
# Usage: bash scripts/eval_liberoplus.sh [hydra overrides...]
#   e.g. runner.ckpt_path=<actor ckpt>   or   actor.model.model_path=<.pt>
set -euo pipefail
: "${IMAGO_ROOT:?}" "${RLINF_ROOT:?}" "${IMAGO_CKPT_DIR:?}"
SUITES=${SUITES:-"libero_spatial libero_object libero_goal libero_10"}
STAMP=$(date +%Y%m%d-%H%M%S)
for SUITE in $SUITES; do
  LIBERO_TYPE=plus LIBERO_SUFFIX=${LIBERO_SUFFIX:-all} \
    bash "$IMAGO_ROOT/scripts/run_imago.sh" eval_liberoplus_zeroshot \
    env.eval.task_suite_name="$SUITE" \
    runner.logger.experiment_name="eval_liberoplus_${SUITE}_${STAMP}" "$@"
done
echo "[eval] per-category success rates are in the eval metrics of each run under $IMAGO_ROOT/logs/"
