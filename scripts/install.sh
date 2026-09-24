#!/bin/bash
# IMAGO environment on 2-3 x RTX PRO 6000 Blackwell (sm_120, driver >= 570).
# Usage: bash scripts/install.sh [/path/to/workdir]
set -euo pipefail

WORK=${1:-$HOME/imago_work}
IMAGO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RLINF_REF=807e5fdd836f0c3671fd351f65faf3bce5ddaf35      # RLinf main, Sep 2026
FASTWAM_REF=7faa71108368fbb3b6885649f112af607427a2d4    # has FastWAMOptionalIDM
mkdir -p "$WORK" && cd "$WORK"

# 1) RLinf at the pinned commit + the IMAGO registration patch.
if [ ! -d RLinf ]; then
  git clone https://github.com/RLinf/RLinf.git
fi
git -C RLinf checkout -q "$RLINF_REF"
git -C RLinf apply --check "$IMAGO_ROOT/third_party/rlinf_imago.patch" 2>/dev/null \
  && git -C RLinf apply "$IMAGO_ROOT/third_party/rlinf_imago.patch" \
  || echo "[imago] patch already applied (or conflicts) -- check 'git -C RLinf diff'"

# 2) Embodied stack: FastWAM (newer ref than RLinf's pin) + LIBERO-Plus.
#    Blackwell: cu128 wheels. FastWAM's own attention is torch SDPA
#    (wan_video_dit.flash_attention -> F.scaled_dot_product_attention), so
#    flash-attn is optional for IMAGO. FA2 builds for sm_120 from source with
#    CUDA >= 12.8 (setup.py emits sm_120 gencode); if the build fails (nvcc
#    segfaults have been reported, Dao-AILab/flash-attention#2361), rerun with
#    IMAGO_SKIP_FLASH_ATTN=1.
cd RLinf
export UV_TORCH_BACKEND=${UV_TORCH_BACKEND:-cu128}
export FASTWAM_GIT_REF=$FASTWAM_REF
FA_FLAG=""
if [ "${IMAGO_SKIP_FLASH_ATTN:-0}" = "1" ]; then FA_FLAG="--no-flash-attn"; fi
export FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS:-120}
export MAX_JOBS=${MAX_JOBS:-8}
bash requirements/install.sh embodied --model fastwam --env liberoplus $FA_FLAG
cd ..

# 3) LIBERO-Plus assets (see RLinf docs, libero.rst#liberopro-plus-benchmark).
echo "[imago] Download LIBERO-Plus assets.zip into the liberoplus package dir:"
echo '  D=$(python -c "import pathlib,liberoplus.liberoplus as l;print(pathlib.Path(l.__file__).parent)")'
echo '  hf download --repo-type dataset Sylvest/LIBERO-plus assets.zip --local-dir "$D" && unzip -o "$D/assets.zip" -d "$D"'

# 4) FastWAM Optional-IDM checkpoint + stats.
mkdir -p ckpt
echo "[imago] hf download yuanty/fastwam libero_optional_idm_2cam224.pt libero_optional_idm_2cam224_dataset_stats.json --local-dir $WORK/ckpt"

cat <<MSG
[imago] Done. Before running:
  export IMAGO_ROOT=$IMAGO_ROOT
  export RLINF_ROOT=$WORK/RLinf
  export IMAGO_CKPT_DIR=$WORK/ckpt
  python \$IMAGO_ROOT/scripts/build_prompt_bank.py --out \$IMAGO_CKPT_DIR/prompt_bank_liberoplus.pt
  python \$IMAGO_ROOT/scripts/make_liberoplus_splits.py --out \$IMAGO_ROOT/configs/splits
MSG
