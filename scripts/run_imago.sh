#!/bin/bash
# Launch an IMAGO experiment through RLinf's (patched) embodied entry point.
# Usage: bash scripts/run_imago.sh <config_name> [hydra overrides...]
#   config_name: liberoplus_imago_fastwam | c1_action_only | c5_first_frame |
#                imago_beta_real_1 | imago_beta_real_0p1
set -euo pipefail
: "${IMAGO_ROOT:?}" "${RLINF_ROOT:?}" "${IMAGO_CKPT_DIR:?}"
CONFIG_NAME=${1:-liberoplus_imago_fastwam}
shift || true

export EMBODIED_PATH=$RLINF_ROOT/examples/embodiment
export PYTHONPATH=$IMAGO_ROOT:$RLINF_ROOT:${PYTHONPATH:-}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=${LIBERO_TYPE:-plus}
export LIBERO_SUFFIX=${LIBERO_SUFFIX:-all}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-$IMAGO_CKPT_DIR/diffsynth}

LOG_DIR=$IMAGO_ROOT/logs/$(date +%Y%m%d-%H%M%S)-$CONFIG_NAME
mkdir -p "$LOG_DIR"
python "$EMBODIED_PATH/train_embodied_agent.py" \
  --config-path "$IMAGO_ROOT/configs" --config-name "$CONFIG_NAME" \
  runner.logger.log_path="$LOG_DIR" "$@" 2>&1 | tee "$LOG_DIR/run.log"
