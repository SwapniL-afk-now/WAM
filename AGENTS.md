# AGENTS.md: working on IMAGO on the GPU server

Target machine: 2–3 × NVIDIA RTX PRO 6000 Blackwell (96 GB, sm_120), driver ≥ 570, CUDA 12.8+.

What IMAGO is and how it works: `imago/README.md`. Research plan: `research_plan/imago_design_plan.md`.

## Rules for agents

- Do **not** install packages or run tests/training unless the user explicitly asks in that session.
- Work only on branch `claude/world-action-model-paper-ltruk5`. Never push elsewhere.
- Never commit checkpoints, datasets, prompt banks, logs or videos. They live under `$IMAGO_CKPT_DIR`, `logs/` and `results/`.
- The IMAGO code has **not been run yet**. Expect API mismatches on the first run and fix them in `imago/`. Do not change RLinf or FastWAM, except through `third_party/rlinf_imago.patch`.

## Pinned versions (important)

| Component | Pin | Why |
|---|---|---|
| RLinf | `807e5fdd836f0c3671fd351f65faf3bce5ddaf35` + `third_party/rlinf_imago.patch` | The patch registers `model_type: fastwam_imago` and `loss_type: embodied_joint_nft` (4 files, 23 lines) |
| FastWAM | `7faa71108368fbb3b6885649f112af607427a2d4`, installed via `FASTWAM_GIT_REF` (`scripts/install.sh` sets it) | RLinf's own pin (`45d8e14`) **predates `FastWAMOptionalIDM`**, the imagine-then-act model IMAGO needs |

**Known side effect:** the newer FastWAM commit may break RLinf's *own* FastWAM wrapper (`rlinf/models/embodiment/fastwam/fastwam_policy.py`, `model_type: fastwam`). That wrapper calls the old API (`video_expert.pre_dit`, `mot.prefill_video_cache`). IMAGO does not use that wrapper; it only borrows a few pure helpers from it. If you need RLinf's stock `fastwam` model, use a separate environment with the old pin.

## Installs you will need

1. **Base stack:** `bash scripts/install.sh ~/imago_work`. It installs RLinf with the patch, FastWAM `7faa711` and LIBERO-Plus, using cu128 wheels.
2. **Transformer-Engine.** Needed **only** for 4-bit rollouts (`imago.rollout_precision: nvfp4`, i.e. `configs/imago_fast.yaml`; Sol-RL's NVFP4 explore / BF16 train):
   ```bash
   pip install --no-build-isolation "transformer-engine[pytorch]"
   ```
   Without it, keep `rollout_precision: bf16`. Everything else still works.
3. **flash-attn: optional.** FastWAM's attention is `torch.nn.functional.scaled_dot_product_attention`. FA2 builds for sm_120 with CUDA ≥ 12.8 (`FLASH_ATTN_CUDA_ARCHS=120`); `install.sh` tries it. If nvcc crashes (Dao-AILab/flash-attention#2361), rerun with `IMAGO_SKIP_FLASH_ATTN=1`.
4. **LIBERO-Plus assets:** follow the `hf download ... Sylvest/LIBERO-plus assets.zip` hint printed by `install.sh`.

## Setup order

```bash
export IMAGO_ROOT=$PWD RLINF_ROOT=~/imago_work/RLinf IMAGO_CKPT_DIR=~/imago_work/ckpt
hf download yuanty/fastwam libero_optional_idm_2cam224.pt \
    libero_optional_idm_2cam224_dataset_stats.json --local-dir $IMAGO_CKPT_DIR
python scripts/build_prompt_bank.py --out $IMAGO_CKPT_DIR/prompt_bank_liberoplus.pt
python scripts/make_liberoplus_splits.py --out configs/splits    # check the printed categories
python scripts/profile_rollout.py --batch-sizes 8 16 32         # throughput and memory per GPU
```

Then, in this order (details in `imago/README.md`, section "Checks to run first"):

1. **Parity.** At step 0, success should be within ±2 points of FastWAM's README (97.75 first-frame / 98.55 idm on standard LIBERO).
2. **Environment layout.** All task descriptions are in the prompt bank, and GRPO groups are contiguous env slots.
3. **NFT sanity.** On the first update, `delta_v_norm` is about 0. With a constant reward, the weights don't drift.
4. **Smoke run.** One task, one perturbation, 20 updates. Success must rise and each GPU must stay under 90 GB.
5. **Kill test.** `bash scripts/run_imago.sh liberoplus_imago_fastwam` vs `bash scripts/run_imago.sh c1_action_only`, at equal rollouts.

With 3 GPUs set `env.train.total_num_envs=48`. The batch arithmetic is explained in the header of `configs/liberoplus_imago_fastwam.yaml`.

## DIDO one-step imagination (implemented, optional pipeline)

DIDO (arXiv 2609.15570) distils the video branch to **one denoising step**. The official code is not released, so we implemented the parts IMAGO needs, following the paper's recipe (full transcription in `research_plan/dido_implementation_guide.md`).

| Part | File | Status |
|---|---|---|
| Stage I: DMD2 one-step distillation (teacher = FastWAM video expert on its 4-step grid, fake score every iteration / generator every 5th, normalised DMD gradient) | `imago/dido/distill.py`, `configs/dido/stage1_libero.yaml` | implemented |
| Stage II: adapt the action expert to one-step imagination (λ_video 0.5, λ_act 1.0, lr 1e-4, cosine) | `imago/dido/adapt.py`, `configs/dido/stage2_libero.yaml` | implemented |
| Dynamics-based token refinement (top ~15% of 2×2 regions kept, rest pooled) | `imago/dido/token_refine.py`, `imago.token_refine` | implemented; used in rollouts, the NFT action loss and Stage II |
| Interaction tokens + box / DINOv3 supervision | not implemented | accuracy add-on; needs a detector/tracker label pipeline |

Deviations from the paper:

- **No teacher-preparation phase.** FastWAM Optional-IDM is already robot-domain; its video expert is the teacher.
- **Teacher guidance scale 1.0 instead of 7.** FastWAM is trained without prompt dropout, so it has no unconditional branch.
- **LoRA instead of full fine-tuning.** The student and the fake score are two LoRA adapters on one frozen base (one 5B copy per GPU, fits 2–3 × 96 GB). Learning rates are about 10× the paper's (tune on the server).
- **Stage II trains the video expert through LoRA,** merged at export, instead of full fine-tuning.
- **Token grid adapted to FastWAM.** FastWAM has a 7×14 token grid per frame, so the edge regions are always pooled. The paper leaves the reference layer and future frame unspecified; both are config fields.

Run order (after the IMAGO setup above):

```bash
export LIBERO_DATA_ROOT=/path/to/libero_mujoco3.3.2    # HF yuanty/LIBERO-fastwam, extracted
# FastWAM text-embedding cache for the LIBERO data (FastWAM repo script):
#   python scripts/precompute_text_embeds.py task=libero_optional_idm_2cam224_1e-4 \
#     data.train.text_embedding_cache_dir=$IMAGO_CKPT_DIR/text_embeds_cache/libero
torchrun --nproc_per_node 3 -m imago.dido.distill configs/dido/stage1_libero.yaml   # Stage I
torchrun --nproc_per_node 3 -m imago.dido.adapt   configs/dido/stage2_libero.yaml   # Stage II
bash scripts/run_imago.sh imago_dido      # IMAGO RL on the 1-step imagination
```

Checks before trusting it:

- One-step imagination quality vs the 10-step teacher: compare Stage I samples with `analysis/drift.py`-style PSNR.
- LIBERO parity of the Stage II export (expect about 98–99; DIDO reports 98.3 for one-step distillation only).
- `scripts/profile_rollout.py` with `imago.train_video_steps=1` vs 4.

Expected gain, from the paper: −32% end-to-end latency (384 vs 562 ms on an H100); the video share of rollout time drops about 4×.
