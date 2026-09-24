# IMAGO: Outcome-Driven RL on a World Action Model's Imagination — Implementation Plan

## Context

**Thesis.** Today every WAM trains its imagined future to *match reality*: supervised video loss, VAMPO's expert-dynamics reward, WAM-RL's reconstruction reward. Two recent findings suggest that objective doesn't serve action well:
- Fast-WAM found test-time imagination barely helps.
- BadWAM reports WAMs that "dream right but act wrong".

We treat the imagined video as the robot's *chain of thought*. We train video and action **jointly with RL on task success only**, with a realism anchor whose weight goes to about 0, and study what imagination becomes. This is the WAM analogue of R1-Zero; no prior WAM paper does outcome-only RL on the imagination branch.

**Target:** CVPR 2027. **Hardware:** 2–3 × RTX PRO 6000 Blackwell (96 GB, sm_120). Everything is built on existing, framework-supported code.

## Stack decisions (from code inspection)

| Piece | Choice | Why |
|---|---|---|
| Framework | **RLinf** (`github.com/RLinf/RLinf`, pin commit `807e5fdd…`) | Has FastWAM as a registered model, **Diffusion-NFT on Wan2.2** inside the embodied pipeline, GRPO, LIBERO-Plus and RoboTwin envs, collocated placement with offload |
| Model | **FastWAM Optional-IDM** (`yuanty/fastwam: libero_optional_idm_2cam224.pt`) | Wan2.2-TI2V-5B video expert + ActionDiT (about 6.5B total; my estimate from the configs, not from the paper). In `idm` mode it **imagines video, then acts on it**, the mode we need. It scores 98.55% on LIBERO |
| RL algorithm base | **Diffusion-NFT** (`rlinf/workers/actor/fsdp_nft_policy_worker.py::EmbodiedNFTFSDPPolicy`) | Likelihood-free, so it avoids noisy SDE log-probs over 5B-dim video latents. It already runs on Wan2.2 and on a π0 action expert with a task-success reward |
| Action-only baseline code | Flow-SDE core from the fork `Yutenji-Nyamu/rlinf_fastwam` (`rlinf/models/embodiment/fastwam/fastwam_rl.py`) plus the πRL sampler (`rlinf/models/embodiment/openpi/sampling/rl_sampler.py`) | Batched Flow-SDE rollout and log-probs for FastWAM already exist there |
| Primary benchmark | **LIBERO-Plus** (in RLinf via `LIBERO_TYPE=plus`) | Unsaturated (π0 53.6 total; camera 13.8, robot-init 6.0). CVPR-2026 benchmark. MuJoCo EGL is less risky on Blackwell than SAPIEN. The imagination checkpoint already exists |
| Second benchmark | **RoboTwin 2.0 randomized ("hard")**, 6–8 tasks | FastWAM trained on clean demos only gets 1.9% on randomized, so there is huge headroom. Needs our own Optional-IDM SFT (Phase 3) |

Upstream RLinf's FastWAM wrapper is **SFT/eval only**:
- It only accepts the `create_fastwam` target (no Optional-IDM variant).
- `default_forward` raises `NotImplementedError`.
- Upstream `infer_action` accepts batch size 1 only.

Closing these gaps is the core engineering work.

## Algorithm: joint-branch NFT with outcome-only reward

For each group of G=8 rollouts from the same LIBERO-Plus init state:

1. **Rollout (per action chunk):**
   - Sample the video noise ε_v and the action noise ε_a. Different seeds give different imagined futures, and hence different actions; this is where exploration comes from.
   - Run an ODE for video with 4 steps, then an ODE for actions with 4 steps (idm mode). Actions attend to the imagined video.
   - Store x0_video (latents), x0_action, and the conditioning (first-frame latent, cached text embedding, proprio).
2. **Reward:** binary episode success from the env, broadcast to every chunk (chunk-level, as in πRL). Filter out groups whose success rate is outside [0.1, 0.9].
3. **Advantage:** GRPO normalisation within the group (`rlinf/algorithms/advantages.py::grpo`), mapped to [0,1] as NFT requires.
4. **Loss per branch b ∈ {video, action}:** reuse NFT's `_compute_nft_target_and_pred` / `_compute_nft_clipped_energies` / `_compute_nft_loss`:
   `L = w_v·L_NFT(video) + w_a·L_NFT(action) + β_real·‖v_θ^video − v_ref^video‖²`
   - **β_real is the key knob**: the pull back toward the pretrained video prediction. Sweep it over {1, 0.1, 0}.
   - `v_old` is the EMA-LoRA reference (`soft_update_rollout_model`, `nft_tau` schedule). No second full model copy is needed.
5. **Trainable parameters:**
   - LoRA (r=32) on the video expert's last 12 blocks and on all 30 action-expert blocks. Alternatively, fully train the last 12 action blocks, as PFD's "12×12" recipe does.
   - T5, the VAE, and early blocks stay frozen. Text embeddings are precomputed, so T5 is off the GPU.

Controls, all using the same code, budget and seeds:
- **(C1)** Action-only NFT, video frozen.
- **(C2)** Action-only Flow-SDE GRPO (fork core).
- **(C3)** Realism-reward RL: reward = −latent L1 between the imagined and real future (VAMPO/WAM-RL style).
- **(C4)** SFT continuation.
- **(C5)** FastWAM first-frame mode (no imagination).

## Code changes (fork RLinf → `third_party/RLinf`, branch `imago`)

Changes in the RLinf fork:
- `rlinf/models/embodiment/fastwam/__init__.py`
  - Accept `create_fastwam_optional_idm`.
  - Inject LoRA via peft on the target blocks (use `mot.get_expert_tail_blocks` as PFD does).
  - Load precomputed text embeddings.
- `rlinf/models/embodiment/fastwam/fastwam_policy.py`
  - **Batched** `predict_action_batch` for idm mode. Port the batched video-cache and ODE logic from FastWAM's `infer_joint` / `_denoise_action_with_video_cache` (`src/fastwam/models/wan22/fastwam.py` L703/L771).
  - Return NFT fields: `nft_x0_video`, `nft_x0_action`, noise, conditioning.
  - `forward(ForwardType.NFT)` predicts the velocity for both branches at re-sampled t.
  - `default_forward` (Flow-SDE log-probs) for C2, ported from the fork's `fastwam_rl.py`.
- **New** `rlinf/workers/actor/fsdp_joint_nft_worker.py`: `EmbodiedJointNFTFSDPPolicy(EmbodiedNFTFSDPPolicy)`. It loops the NFT energies over both branches, adds the realism anchor, and logs losses per branch.
- `examples/embodiment/train_embodied_agent.py`: add `loss_type: embodied_joint_nft` → the new worker.
- **New configs:**
  - `examples/embodiment/config/model/fastwam_imago.yaml`
  - `examples/embodiment/config/liberoplus_imago_fastwam.yaml`, plus baseline variants C1–C5
  - `robotwin_imago_fastwam.yaml`
  - Model settings: placement `actor,env,rollout: all`, `enable_offload: True`, `group_size: 8`, `total_num_envs: 32` (start there; tune after profiling), `num_steps: 4` for video and for action, NFT `nft_beta 0.1`, `nft_tau [1.0,0.01,0,70]`, LoRA lr 1e-4.

Changes in this WAM repo:
- `analysis/drift.py`: FVD and latent L1 between imagined and real frames over training. This tests whether success keeps rising as realism falls.
- `analysis/swap_test.py`: counterfactual future swap. Measures how sensitive actions are to the imagined video (the BadWAM test).
- `analysis/attention_maps.py`: action→video attention (explicit weights at 1–3 layers, action queries only).
- `analysis/probe_subgoals.py`: whether imagined frames become keyframes or subgoals (nearest-neighbour to future real frames at different time offsets).
- `scripts/profile_rollout.py`: throughput and memory of the rollout per env.

## Phases (7 weeks to about Nov 12)

1. **Week 1: environment and parity.**
   - Install: `bash requirements/install.sh embodied --model fastwam --env liberoplus` with `UV_TORCH_BACKEND=cu128`. FastWAM attention is torch SDPA, so flash-attn is optional; FA2 builds for sm_120 with CUDA ≥ 12.8 (skip with `IMAGO_SKIP_FLASH_ATTN=1` if the build fails).
   - Reproduce the Optional-IDM checkpoint's LIBERO idm score through RLinf eval (±2 points of 98.55 on a subset), then its LIBERO-Plus baseline per perturbation category.
   - Profile rollout throughput.
2. **Week 2: model plumbing.**
   - Batched idm rollout, NFT fields, the joint-NFT worker.
   - Sanity checks: zero advantage gives no parameter change; an overfit test on one task and one perturbation succeeds.
   - **Kill test by about Oct 10:** on 4 LIBERO-Plus categories (camera, robot-init, light, layout), compare IMAGO with C1 at equal rollouts.
   - **Go** if IMAGO beats C1 clearly, *or* the swap test shows actions now follow imagination.
   - **No-go:** keep the negative result as an analysis paper ("imagination matters only as a training signal").
3. **Weeks 3–4: main runs.**
   - LIBERO-Plus full run: train on a subset of perturbation instances, evaluate on held-out instances plus all 7 dimensions.
   - Run the baselines C1–C5 and the β_real sweep. The third GPU runs baselines and evaluation in parallel.
   - In parallel on the third GPU, the RoboTwin Optional-IDM SFT:
     - Start from `robotwin_uncond_3cam_384.pt` with the `fastwam_optional_idm` config on **clean-only** data from `yuanty/robotwin2.0-fastwam`.
     - Use DeepSpeed ZeRO-1 with tail-12×12 partial training for about 1–2 days.
     - For RoboTwin on Blackwell, add `12.0` to `TORCH_CUDA_ARCH_LIST` in the RoboTwin install script, and run bare-metal with `NVIDIA_DRIVER_CAPABILITIES=all` (Vulkan).
4. **Week 5: RoboTwin randomized RL.**
   - 6–8 tasks, chosen from the leaderboard hard set: stamp_seal, turn_switch, blocks_ranking_size, place_mouse_pad, put_bottles_dustbin, handover_block, place_dual_shoes, scan_object.
   - Check each against the RLinf-supported list (place_fan and put_object_cabinet are unsupported).
   - Run IMAGO vs C1 vs C3.
5. **Week 6: analysis.** Drift vs success curves, the swap test, attention maps, subgoal probes, whether imagination changes on hard states, and transfer to held-out tasks.
6. **Week 7:** writing, video supplement, code release.

**Fallbacks:**
- If Week 1 slips by more than 5 days, drop RoboTwin and add LIBERO-Pro instead.
- If throughput is below about 1k episodes/hour on 2 GPUs, cut video frames or steps and train on fewer categories.

## Verification

- **Parity:** the RLinf-wrapped batched idm rollout matches the official FastWAM `run_libero_manager` success on the same seeds (±2 points).
- **Unit checks:**
  - NFT loss is about 0 when v_θ = v_old.
  - A constant advantage causes no drift.
  - LoRA-EMA reference updates follow `nft_tau`.
  - Chunk-level reward broadcasting is correct.
- **Smoke run:** 1 task × 1 perturbation, 20 updates. Success must increase on seeds it has seen, and memory must stay under 90 GB per GPU.
- **Full eval:** RLinf `evaluations/run_eval.sh libero` with `LIBERO_TYPE=plus`, all 7 dimensions, 3 seeds, error bars. RoboTwin: 100 trials per task in the randomized setting.
- **Every run logs:** success, rollouts, GPU-hours, drift (FVD), swap-sensitivity, and inference latency (which is unchanged from base idm mode).

## Key risks

- **Rollout cost:** video denoising on every chunk dominates. Mitigations are 4 denoising steps, the video KV cache, 32 envs, and offload.
- **Sparse reward on a 5B video branch:** mitigated by LoRA on the last blocks only and the group filter. An optional progress reward is added only as an ablation, to keep the "outcome-only" claim clean.
- **Blackwell:** FastWAM uses SDPA, so flash-attn is not on the critical path (FA2 does build for sm_120). SAPIEN/Vulkan in containers is fragile, so run RoboTwin bare-metal.
- **Scoop:** post on arXiv on the day of submission.

## Addendum: efficiency techniques (implemented in `imago/efficiency.py`, `imago/lowprec.py`)

- Flash-GRPO: iso-temporal grouping and temporal gradient rectification.
- TempFlow-GRPO: noise-aware weighting.
- MixGRPO: sliding noise window.
- Sol-RL: NVFP4 rollouts with BF16 training; uses the FP4 tensor cores of the RTX PRO 6000.

Config: `configs/imago_fast.yaml`. Details and what was deliberately not ported: `imago/README.md`.
