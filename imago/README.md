# IMAGO: outcome-driven RL on a world action model's imagination

IMAGO trains **both** branches of an imagine-then-act world action model (FastWAM Optional-IDM, a Wan2.2-TI2V-5B video expert plus an action expert) with a single **task-success-only** advantage, using **Diffusion-NFT**. The imagined video is treated as the robot's chain of thought: it is rewarded for being *useful for acting*, not for being realistic. An optional realism anchor, `imago_beta_real`, controls how far the imagined video may drift from what the pretrained model would predict.

It is built on **RLinf** (the `embodied_nft` pipeline, LIBERO-Plus env, FSDP workers) and the upstream **FastWAM** model code. All of this is untested so far: it needs GPUs, so the verification steps below are planned for the server.

## Layout

| Path | What |
|---|---|
| `imago/policy.py` | `FastWAMImaginePolicy`. Batched imagine-then-act rollout (video ODE → video K/V cache → action ODE), per-branch velocity for NFT, prompt bank, eval recording |
| `imago/builder.py` | `get_model` for `model_type: fastwam_imago`. Composes the FastWAM config, loads the Optional-IDM checkpoint, injects LoRA, freezes everything else |
| `imago/lora.py` | Dependency-free LoRA on DiT block linears; pretrained reference via `lora_disabled` |
| `imago/nft_math.py` | NFT `mse` objective per branch, realism anchor, masks (pure torch) |
| `imago/worker.py` | `EmbodiedJointNFTFSDPPolicy` (`loss_type: embodied_joint_nft`). EMA and pretrained references come from swapping the trainable tensors into the live model, so no full second model is kept |
| `imago/efficiency.py` | Iso-temporal groups and gradient rectification (Flash-GRPO), noise-aware weights (TempFlow-GRPO), sliding noise window (MixGRPO) |
| `imago/lowprec.py` | NVFP4 rollouts (Sol-RL) using Transformer-Engine shadow linears with LoRA merged in |
| `third_party/rlinf_imago.patch` | 23-line patch that registers the model type and loss type in RLinf `807e5fd` |
| `configs/` | FastWAM config for the Optional-IDM checkpoint, RLinf model group, main run and controls |
| `scripts/` | `install.sh`, `run_imago.sh`, `build_prompt_bank.py`, `make_liberoplus_splits.py`, `profile_rollout.py` |
| `analysis/` | `drift.py` (how realistic the imagination is), `swap_test.py` (do actions follow the imagination?), `attention_maps.py` (where action tokens look), `probe_subgoals.py` (does imagination jump ahead?) |

## Experiments

| Config | What it tests |
|---|---|
| `liberoplus_imago_fastwam` | **IMAGO**: joint video + action NFT, outcome-only, `beta_real = 0` |
| `c1_action_only` | C1: same rollouts, imagination frozen, action-only NFT |
| `c5_first_frame` | C5: Fast-WAM mode (no test-time imagination), action-only NFT |
| `imago_beta_real_1`, `imago_beta_real_0p1` | Realism-anchor sweep: "imagination must stay true" vs "free" |
| `imago_fast` | IMAGO plus all efficiency techniques (NVFP4 rollouts, iso-temporal, TempFlow weights, MixGRPO window) |

Not implemented yet:
- C2 (action-only Flow-SDE GRPO): port `fastwam_rl.py` from `Yutenji-Nyamu/rlinf_fastwam`.
- C3 (VAMPO / WAM-RL-style reconstruction reward): currently approximated by `imago_beta_real_1`.

## Efficiency techniques and where they come from

NFT already trains on one re-noised step per sample. That is the main saving that one-step and sliding-window GRPO methods get (no backprop through the sampling chain), so it is built in. The ported pieces act on *which* noise level is trained, *how it is weighted*, and *how fast rollouts run*:

| Technique | Paper / code | Where | Knob |
|---|---|---|---|
| Iso-temporal grouping | Flash-GRPO, ICML'26 ([code](https://github.com/Shredded-Pork/Flash-GRPO)) | `efficiency.sample_grouped_timesteps` | `imago_iso_temporal` |
| Temporal gradient rectification | Flash-GRPO | `efficiency.time_weights("rectify")` | `imago_time_weight: rectify` (use with `nft_weight_mode: constant`) |
| Noise-aware weighting | TempFlow-GRPO, ICLR'26 ([code](https://github.com/Shredded-Pork/TempFlow-GRPO)) | `efficiency.time_weights("tempflow")` | `imago_time_weight: tempflow` |
| Sliding noise window | MixGRPO, ECCV'26 ([code](https://github.com/Tencent-Hunyuan/MixGRPO)) | `efficiency.sliding_window` | `imago_noise_window`, `imago_window_interval` |
| NVFP4 rollout, BF16 train | Sol-RL ([code](https://github.com/NVlabs/Sana), `train_scripts/sol_rl/`) | `lowprec.NVFP4Rollout` | `imago.rollout_precision: nvfp4` |
| Forward-process RL (the base) | DiffusionNFT, ICLR'26 ([code](https://github.com/NVlabs/DiffusionNFT)), via RLinf | `worker.py` | `loss_type: embodied_joint_nft` |

Deliberately not ported:
- **Sol-RL's candidate selection** (score 96 FP4 samples, keep the most contrastive 24) needs a reward that can score a sample without executing it. Robot rewards need the episode to be run.
- **Sparse or sliding-tile video attention** gives nothing here: a 224×448, 9-frame clip is only 294 video tokens.
- **One-step WAM distillation (DIDO, arXiv 2609.15570).** It uses DMD to go from a 4-step Wan2.2-TI2V-5B teacher to 1 step, with bounding-box and DINOv3 supervision. That saves 4→1 steps on the *video branch only*; the action denoising and cache prefill remain, so it is not a 4× faster rollout. Code is not released (the repo is README-only). Its LIBERO-Plus result, 76.6, is worth citing as a baseline. Flash-WAM targets LingBot-VA. Details and a check-back trigger are in `AGENTS.md`.

## Setup (on the GPU server)

```bash
bash scripts/install.sh ~/imago_work        # RLinf@807e5fd + patch, FastWAM@7faa711, LIBERO-Plus
export IMAGO_ROOT=$PWD RLINF_ROOT=~/imago_work/RLinf IMAGO_CKPT_DIR=~/imago_work/ckpt
hf download yuanty/fastwam libero_optional_idm_2cam224.pt libero_optional_idm_2cam224_dataset_stats.json --local-dir $IMAGO_CKPT_DIR
python scripts/build_prompt_bank.py --out $IMAGO_CKPT_DIR/prompt_bank_liberoplus.pt
python scripts/make_liberoplus_splits.py --out configs/splits
python scripts/profile_rollout.py --batch-sizes 8 16 32          # Week-1 throughput / memory
bash scripts/run_imago.sh liberoplus_imago_fastwam                # main run
bash scripts/run_imago.sh c1_action_only                          # control
```

Notes:
- **Blackwell (sm_120).** FastWAM's attention is `F.scaled_dot_product_attention`, so flash-attn is optional for IMAGO. FA2 compiles for sm_120 with CUDA ≥ 12.8 (`FLASH_ATTN_CUDA_ARCHS=120`), and `install.sh` tries it by default. If nvcc crashes (Dao-AILab/flash-attention#2361), rerun with `IMAGO_SKIP_FLASH_ATTN=1`.
- **NVFP4** additionally needs `pip install --no-build-isolation "transformer-engine[pytorch]"`.
- **3 GPUs:** use `env.train.total_num_envs=48`, because envs must split evenly across GPUs and groups. `global_batch_size` must divide `(max_episode_steps / num_action_chunks) * total_num_envs * rollout_epoch` and be divisible by `micro_batch_size * num_gpus`.

## Checks to run first (server, Week 1–2)

1. **Parity.** Eval `c5_first_frame` and IMAGO at step 0 on LIBERO (standard). Success should be within ±2 points of the FastWAM README (97.75 first-frame / 98.55 idm).
2. **Environment layout.** Confirm that LIBERO-Plus task descriptions all appear in the prompt bank (the policy raises `KeyError` otherwise), and that groups are contiguous env slots (`LiberoEnv.reset_state_ids.repeat(group_size)`).
3. **NFT sanity.** With `nft_tau: 1.0` and the first update, `delta_v_norm` should be about 0 and E_pos ≈ E_neg. With the reward forced constant, LoRA weights should not drift.
4. **Smoke run.** One task, one perturbation, 20 updates. Success on seeds it has seen must go up, and each GPU must stay under 90 GB.
5. **Kill test** (see `research_plan/`). IMAGO vs C1 at equal rollouts on four LIBERO-Plus categories.
