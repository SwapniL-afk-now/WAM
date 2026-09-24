# IMAGO: outcome-driven RL on a world action model's imagination

IMAGO trains **both** branches of an imagine-then-act world action model (FastWAM Optional-IDM, a Wan2.2-TI2V-5B video expert plus an action expert) with a single **task-success-only** advantage, using **Diffusion-NFT**. The imagined video is treated as the robot's chain of thought: it is rewarded for being *useful for acting*, not for being realistic. An optional realism anchor, `imago_beta_real`, controls how far the imagined video may drift from what the pretrained model would predict.

It is built on **RLinf** (the `embodied_nft` pipeline, LIBERO envs, FSDP2 workers) and the upstream **FastWAM** model code, starting from our full-parameter DIDO reproduction (one-step imagination + interaction tokens). Protocol: RL on standard LIBERO only (with generic domain randomization), LIBERO-Plus zero-shot, so results are comparable with DIDO (76.6) and Faster-WAM (75.0). Every stage is **full-parameter**; LoRA is an ablation. All of this is untested so far: it needs GPUs, so the verification steps below are planned for the server.

## Layout

| Path | What |
|---|---|
| `imago/policy.py` | `FastWAMImaginePolicy`. Batched imagine-then-act rollout (video ODE → video K/V cache → action ODE), per-branch velocity for NFT, prompt bank, eval recording |
| `imago/builder.py` | `get_model` for `model_type: fastwam_imago`. Composes the FastWAM config, attaches DIDO tokens, loads the checkpoint, opens branches per `imago_trainable` (`full` default; `full_tail`, `lora`, `frozen`) with per-branch `lr_mult`; `imago.policy: flow_grpo` selects the C2 policy |
| `imago/lora.py` | LoRA on DiT block linears (ablation only) |
| `imago/fsdp.py` | FSDP2 root-unit sharding, fp32 master / bf16 compute, CPU offload, memory estimate (DIDO trainers) |
| `imago/envs/libero_randomized.py` | Standard LIBERO with generic randomization (joint noise, camera jitter, light scale) for zero-shot RL |
| `imago/baselines/` | C2 `flow_grpo.py` (Flow-SDE GRPO, ported from `rlinf_fastwam`), C3 `realism.py` (−LPIPS realism reward) |
| `imago/nft_math.py` | NFT `mse` objective per branch, realism anchor, masks (pure torch) |
| `imago/worker.py` | `EmbodiedJointNFTFSDPPolicy` (`loss_type: embodied_joint_nft`). EMA and pretrained references are sharded copies of the trainable parameters, swapped into the live model, so no second full model is kept; also adds the C3 reward when enabled |
| `imago/efficiency.py` | Iso-temporal groups and gradient rectification (Flash-GRPO), noise-aware weights (TempFlow-GRPO), sliding noise window (MixGRPO) |
| `imago/lowprec.py` | NVFP4 rollouts (Sol-RL) using Transformer-Engine shadow linears (ablation) |
| `imago/dido/` | DIDO (arXiv 2609.15570), full-parameter: Stage I one-step DMD distillation (`distill.py`), Stage II joint adaptation (`adapt.py`), interaction tokens + box / DINOv3 losses (`interaction.py`), annotated dataset (`data.py`), token refinement (`token_refine.py`) |
| `third_party/rlinf_imago.patch` | Patch for RLinf `807e5fd`: registers the model type and loss type, and routes `imago_randomization` to the randomized LIBERO env |
| `configs/` | FastWAM config for the Optional-IDM checkpoint, RLinf model group, main run and controls |
| `scripts/` | `install.sh`, `run_imago.sh`, `eval_liberoplus.sh`, `build_prompt_bank.py`, `make_libero4_split.py`, `annotate_libero.py`, `audit_annotations.py`, `make_liberoplus_splits.py`, `profile_rollout.py` |
| `analysis/` | `drift.py` (how realistic the imagination is), `swap_test.py` (do actions follow the imagination?), `attention_maps.py` (where action tokens look), `probe_subgoals.py` (does imagination jump ahead?) |

## Experiments

All runs start from the DIDO Stage II export, train on standard LIBERO (four suites, generic randomization) and are evaluated zero-shot on LIBERO-Plus with `scripts/eval_liberoplus.sh`.

| Config | What it tests |
|---|---|
| `libero_rl_zeroshot` (`_2gpu`) | **IMAGO**: full-parameter joint video + action NFT, outcome-only, `beta_real = 0` |
| `c1_action_only` | C1: same rollouts, imagination frozen, action-only NFT |
| `c2_flow_grpo` | C2: action-only Flow-SDE GRPO (likelihood RL, πRL style) |
| `c3_realism_reward` | C3: success + dense −LPIPS(imagined, next real frame) reward (VAMPO / WAM-RL style) |
| `c5_first_frame` | C5: Fast-WAM mode (no test-time imagination), action-only NFT |
| `imago_beta_real_0p1` | Realism-anchor sweep point (main run is 0) |
| `imago_fastwam_multistep` | Ablation: released multi-step FastWAM, no DIDO |
| `imago_fast` | Efficiency ablation: NVFP4 rollouts, TempFlow weights, MixGRPO window on the action branch |
| `eval_liberoplus_zeroshot` | Evaluation only (one process per suite) |

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

Defaults are accuracy-first: only iso-temporal grouping is on. The others are ablations until they pass a ≤ 0.5-point parity check.

Deliberately not ported:
- **Sol-RL's candidate selection** (score 96 FP4 samples, keep the most contrastive 24) needs a reward that can score a sample without executing it. Robot rewards need the episode to be run.
- **Sparse or sliding-tile video attention** gives nothing here: a 224×448, 9-frame clip is only 294 video tokens.

## Setup (on the GPU server)

Follow `AGENTS.md` ("Setup order"): install, prompt banks, four-suite split, annotations and audit, DIDO Stage I/II, checks, then the runs. It also lists the GPU layouts for 2 and 3 GPUs.

Notes:
- **Blackwell (sm_120).** FastWAM's attention is `F.scaled_dot_product_attention`, so flash-attn is optional. FA2 compiles for sm_120 with CUDA ≥ 12.8 (`FLASH_ATTN_CUDA_ARCHS=120`); `install.sh` tries it. If nvcc crashes (Dao-AILab/flash-attention#2361), rerun with `IMAGO_SKIP_FLASH_ATTN=1`.
- **Batch arithmetic** (header of `configs/libero_rl_zeroshot.yaml`): `global_batch_size` must divide `(max_episode_steps / num_action_chunks) * total_num_envs * rollout_epoch` and be divisible by `micro_batch_size * num_gpus`. The defaults work for 2 and 3 GPUs.

## Checks to run first (server)

1. **Parity.** At step 0, `imago_fastwam_multistep` should be within ±2 points of the FastWAM README on standard LIBERO (98.55 idm). The DIDO Stage II export should reach ≥ 98.9 on LIBERO and about 76 on LIBERO-Plus zero-shot.
2. **Environment.** All task descriptions appear in the prompt bank (the policy raises `KeyError` otherwise); groups are contiguous env slots (`LiberoEnv.reset_state_ids.repeat(group_size)`); randomized episodes look plausible.
3. **NFT sanity.** With `nft_tau: 1.0` and the first update, `delta_v_norm` should be about 0 and E_pos ≈ E_neg. With the reward forced constant, weights should not drift.
4. **Smoke run.** Four tasks, 20 updates. Success must go up, and each GPU must stay under 90 GB.
5. **Kill test.** IMAGO vs C1 and C2 at equal rollouts, LIBERO-Plus zero-shot, 3 seeds.
