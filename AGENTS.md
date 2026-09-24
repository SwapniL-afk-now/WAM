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

## Deferred: one-step WAM distillation (DIDO)

DIDO (arXiv 2609.15570) distils the video branch of a FastWAM-style world action model to **one denoising step**. Its official code is **not released**: `github.com/LoveJu1y/DIDO-WAM` has only a README. The paper does contain a full implementation guide (Appendix B/C, Table 4). It is transcribed, with every hyperparameter and the remaining gaps, in **`research_plan/dido_implementation_guide.md`**. Follow that file if DIDO is implemented.

Key facts:

- **Same design family as FastWAM.** Wan2.2-TI2V-5B video model, MoT action expert, and Fast-WAM's video loss in Stage II.
- **Pipeline.** Start from a public 4-step Wan2.2-TI2V-5B checkpoint, adapt it into a robot teacher, then:
  - Stage I: DMD2-style one-step distillation (teacher CFG 7, fake score updated every iteration, generator every 5th), plus 64 interaction tokens with box and DINOv3 supervision.
  - Stage II: joint policy training.
- **Speed.** 384 ms vs 562 ms for the 4-step teacher on an H100: **−32% end to end, not 4×**, because action denoising remains. Fast-WAM takes 356 ms.
- **LIBERO-Plus.** Fast-WAM 51.5 → DIDO 76.6 (Faster-WAM 75.0). Our IMAGO base (FastWAM) therefore has large headroom; report DIDO as the baseline to beat.
- **Cost on this server.** Stage I trains two of three 5B copies. Full fine-tuning needs FSDP full-shard on all 3 GPUs; LoRA would be a deviation from the paper. Box labels need a detector + tracker pipeline (the gripper comes from simulator state).
- **Check-back trigger.** When the DIDO code or checkpoints appear, evaluate replacing IMAGO's 4-step video imagination (`imago.train_video_steps`) with a one-step expert and re-run `scripts/profile_rollout.py`.
