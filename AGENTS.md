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
| RLinf | `807e5fdd836f0c3671fd351f65faf3bce5ddaf35` + `third_party/rlinf_imago.patch` | The patch registers `model_type: fastwam_imago` and `loss_type: embodied_joint_nft`, and lets `env.*.imago_randomization.enabled` select IMAGO's randomized LIBERO env (`get_env_cls` + the LIBERO worker's `reconfigure`) |
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
5. **DIDO annotation pipeline** (run once, before DIDO Stage I): `pip install transformers av pyarrow` and SAM 2 (`pip install "git+https://github.com/facebookresearch/sam2.git"`, then download `sam2.1_hiera_large.pt` into `checkpoints/`). Grounding DINO and DINOv3 come from Hugging Face; DINOv3 is gated, so accept its license and `hf auth login` first.
6. **C3 baseline only:** `pip install lpips`.

## Protocol (read first)

We compare against DIDO (76.6) and Faster-WAM (75.0) on **LIBERO-Plus, zero-shot**:

- **Training** (DIDO Stage I/II and IMAGO RL) uses only the standard LIBERO four suites (Spatial, Object, Goal, 10). RL adds *generic* domain randomization (joint noise, camera jitter, light scaling; `imago/envs/libero_randomized.py`). It uses no LIBERO-Plus assets, splits or perturbation generators. Report a no-randomization run as well (`env.train.imago_randomization.enabled=False`).
- **LIBERO-Plus is evaluation only**, in separate runs (`scripts/eval_liberoplus.sh`). RLinf reads `LIBERO_TYPE` once per process, so training and LIBERO-Plus evaluation cannot share a run. `scripts/run_imago.sh` defaults to `LIBERO_TYPE=standard`, and the randomized env refuses to start under any other value.
- **Accuracy first.** Every stage is full-parameter (FSDP2), as in FastWAM and DIDO. LoRA and the efficiency tricks that might cost accuracy (NVFP4 rollouts, MixGRPO window, TempFlow weighting) are ablations only. Turn one on in the main run only after it is within 0.5 points on LIBERO-Plus zero-shot.

## GPU layouts (2 or 3 × 96 GB)

| Stage | 3 GPUs | 2 GPUs |
|---|---|---|
| DIDO Stage I (teacher, generator + tokens, fake score; each an FSDP2 unit) | about 57 GB/GPU | generator CPU-offloaded (`fsdp.offload_generator_when_2gpu`; it steps every 5th iteration) |
| DIDO Stage II (whole MoT + tokens + proprio, one FSDP2 unit) | about 35–45 GB/GPU | about 65 GB/GPU; set `fsdp.offload_when_2gpu=true` if it OOMs |
| IMAGO RL (`libero_rl_zeroshot`) | FSDP2 full shard, EMA reference sharded | `libero_rl_zeroshot_2gpu` (actor CPU offload; reference swap buffer on CPU) |

Every trainer prints its memory estimate at start-up (`imago/fsdp.py`). Stage I's global batch is 128 on 2 GPUs and 120 on 3 (micro 8 × accumulation 5 × 3), because 128 is not divisible by 24.

## Setup order

```bash
export IMAGO_ROOT=$PWD RLINF_ROOT=~/imago_work/RLinf IMAGO_CKPT_DIR=~/imago_work/ckpt
hf download yuanty/fastwam libero_optional_idm_2cam224.pt \
    libero_optional_idm_2cam224_dataset_stats.json --local-dir $IMAGO_CKPT_DIR
LIBERO_TYPE=standard python scripts/build_prompt_bank.py --libero-type standard --out $IMAGO_CKPT_DIR/prompt_bank_libero.pt
LIBERO_TYPE=plus     python scripts/build_prompt_bank.py --libero-type plus     --out $IMAGO_CKPT_DIR/prompt_bank_liberoplus.pt
LIBERO_TYPE=standard python scripts/make_libero4_split.py --out configs/splits/libero4_train.yaml   # must print 0–29, 120–129
python scripts/profile_rollout.py --batch-sizes 8 16 32         # throughput and memory per GPU
```

If `make_libero4_split.py` prints different ids, copy them into `env.train.task_id_filter` of `configs/libero_rl_zeroshot.yaml`.

### 1. DIDO interaction annotations (once)

```bash
export LIBERO_DATA_ROOT=/path/to/libero_mujoco3.3.2    # HF yuanty/LIBERO-fastwam, extracted
for S in libero_spatial libero_object libero_goal libero_10; do
  for i in 0 1 2; do
    CUDA_VISIBLE_DEVICES=$i python scripts/annotate_libero.py --suite $S \
      --dataset-dir $LIBERO_DATA_ROOT/${S}_no_noops_lerobot --out $IMAGO_CKPT_DIR/annotations/libero --shard $i/3 &
  done; wait
done
python scripts/audit_annotations.py --dataset-dir $LIBERO_DATA_ROOT/libero_spatial_no_noops_lerobot \
  --ann $IMAGO_CKPT_DIR/annotations/libero --out logs/audit --num 100
```

Open `logs/audit/` and count. DIDO's audit found 95/100 object boxes correct. We need at least that, and 100/100 gripper boxes. If the gripper boxes are mirrored, rerun with another `--image-transform`.

### 2. DIDO Stage I and II (full-parameter)

```bash
# FastWAM text-embedding cache for the LIBERO data (FastWAM repo script):
#   python scripts/precompute_text_embeds.py task=libero_optional_idm_2cam224_1e-4 \
#     data.train.text_embedding_cache_dir=$IMAGO_CKPT_DIR/text_embeds_cache/libero
torchrun --nproc_per_node 3 -m imago.dido.distill configs/dido/stage1_libero.yaml   # -> dido_stage1/fastwam_optional_idm_onestep_video.pt
torchrun --nproc_per_node 3 -m imago.dido.adapt   configs/dido/stage2_libero.yaml   # -> dido_stage2/fastwam_optional_idm_dido.pt
```

### 3. Checks before RL (details in `imago/README.md`, "Checks to run first")

1. **FastWAM parity.** Success at step 0 of `imago_fastwam_multistep` should be within ±2 points of FastWAM's README (98.55 idm on standard LIBERO).
2. **Stage II parity.** Standard LIBERO ≥ 98.9 (DIDO with tokens), and LIBERO-Plus zero-shot around 76 (`scripts/eval_liberoplus.sh`, no `runner.ckpt_path`). This is the "SFT" row of every table.
3. **Environment.** Every task description is in the prompt bank, and GRPO groups are contiguous env slots. Save a few randomized episodes (`env.train.video_cfg.save_video=True`) and check the jitter is mild.
4. **NFT sanity.** On the first update `delta_v_norm` is about 0. With a constant reward the weights don't drift.
5. **Smoke run.** Four tasks, 20 updates. Success must rise and each GPU must stay under 90 GB.

### 4. Runs (3 seeds each; equal rollouts)

```bash
bash scripts/run_imago.sh libero_rl_zeroshot        # IMAGO main (use libero_rl_zeroshot_2gpu on 2 GPUs)
bash scripts/run_imago.sh c1_action_only            # C1 action-only NFT, imagination frozen
bash scripts/run_imago.sh c2_flow_grpo              # C2 Flow-SDE GRPO (likelihood RL), action only
bash scripts/run_imago.sh c3_realism_reward         # C3 success + realism (-LPIPS) reward
bash scripts/run_imago.sh c5_first_frame            # C5 Fast-WAM mode, no imagination
bash scripts/run_imago.sh imago_beta_real_0p1       # realism-anchor sweep
bash scripts/run_imago.sh imago_fastwam_multistep   # ablation: no DIDO
bash scripts/run_imago.sh imago_fast                # efficiency ablation (NVFP4 needs Transformer-Engine)
bash scripts/eval_liberoplus.sh runner.ckpt_path=<actor checkpoint>   # zero-shot LIBERO-Plus, per suite
```

Report alongside: Fast-WAM 51.5, Faster-WAM 75.0, DIDO 76.6 (LIBERO-Plus), and our Stage II checkpoint without RL.

## DIDO one-step imagination (implemented, full-parameter)

DIDO (arXiv 2609.15570) distils the video branch to **one denoising step** and adds interaction tokens. The official code is not released. We implemented the paper's recipe; the full transcription is in `research_plan/dido_implementation_guide.md`.

| Part | File | Status |
|---|---|---|
| Stage I: DMD2 one-step distillation. Three full-parameter video experts (teacher, generator, fake score); fake score every iteration, generator every 5th; normalised DMD gradient; Table 4 hyperparameters | `imago/dido/distill.py`, `configs/dido/stage1_libero.yaml` | implemented |
| Stage II: `0.5·L_video + L_act + L_inter + 0.02·L_align` over the whole MoT, lr 1e-4, cosine, 14,480 steps at batch 384 | `imago/dido/adapt.py`, `configs/dido/stage2_libero.yaml` | implemented |
| Interaction tokens (64 = object / interaction / gripper / alignment × 16), box heads (smooth-L1 + GIoU), DINOv3 alignment | `imago/dido/interaction.py`, `imago/dido/data.py` | implemented |
| Automatic annotation (simulator-projected gripper boxes, Grounding DINO + SAM 2 object boxes, DINOv3 crop features) and audit | `scripts/annotate_libero.py`, `scripts/audit_annotations.py` | implemented |
| Dynamics-based token refinement (top ~15% of 2×2 regions kept, rest pooled) | `imago/dido/token_refine.py` | implemented; used in Stage II, rollouts and the NFT action loss |

Deviations from the paper (each one is a config field where possible):

- **No teacher-preparation phase.** FastWAM Optional-IDM is already robot-domain; its video expert is the teacher.
- **Teacher guidance 1.0 instead of 7.** FastWAM is trained without prompt dropout, so it has no unconditional branch.
- **Continuous teacher timesteps.** DIDO samples its 4-step-distilled teacher's grid {1000, 750, 500, 250}. FastWAM is a full continuous-time model, so we sample its training distribution clipped to [0.02, 0.98]. `dmd.teacher_timesteps: grid` reproduces the paper.
- **Unspecified details we chose:** align layers {10, 20, 29}; box heads on mean-pooled `[T_obj; T_int]` / `[T_grip; T_int]`; tokens take the noisy video's timestep modulation and identity RoPE; frame 0 does not attend to them (FastWAM's first-frame-causal mask); object target from the BDDL `obj_of_interest`; DINOv3 ViT-L/16 pooled to 4×4.
- **Token grid adapted to FastWAM.** FastWAM has a 7×14 token grid per frame, so edge regions are always pooled. The reference layer and future frame are config fields.

Checks before trusting it:

- One-step imagination quality vs the 10-step teacher: PSNR of Stage I samples on held-out clips (`analysis/drift.py`-style).
- LIBERO parity of the Stage II export (≥ 98.9 with tokens; DIDO reports 98.3 for one-step distillation alone).
- `scripts/profile_rollout.py` with `imago.train_video_steps=1` vs 4.

Expected gain, from the paper: −32% end-to-end latency (384 vs 562 ms on an H100), and +3.6 LIBERO-Plus from the interaction tokens.
