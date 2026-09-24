# LLM-RL (2026) → World Action Models: Gap Analysis

Goal: find 2026 LLM-RL ideas that improve **training efficiency and accuracy together** and have not yet been carried over to World Action Models (WAMs).

## Method

1. Collected every arXiv ID from 2026 (`26MM.xxxxx`, Jan–Sep 2026) in community-maintained paper lists:
   - LLM side: [smiles724/Awesome-LLM-RLVR](https://github.com/smiles724/Awesome-LLM-RLVR) (699), [chrisliu298/awesome-on-policy-distillation](https://github.com/chrisliu298/awesome-on-policy-distillation) (464), [nick7nlp/Awesome-LLM-On-Policy-Distillation](https://github.com/nick7nlp/Awesome-LLM-On-Policy-Distillation) (245), [benjaminzwhite/reasoning-models](https://github.com/benjaminzwhite/reasoning-models) (277, RL-filtered), [VoltAgent/awesome-ai-agent-papers](https://github.com/VoltAgent/awesome-ai-agent-papers) (386, RL-filtered), plus TsinghuaC3I and other lists that had only a few 2026 entries.
   - WAM side: [OpenMOSS/Awesome-WAM](https://github.com/OpenMOSS/Awesome-WAM), [XuejiFang/awesome-vla-wam](https://github.com/XuejiFang/awesome-vla-wam), [LMD0311/Awesome-World-Model](https://github.com/LMD0311/Awesome-World-Model), [Denghaoyuan123/Awesome-RL-VLA](https://github.com/Denghaoyuan123/Awesome-RL-VLA), plus 20 WAM RL papers found by web search that the lists missed.
2. Removed duplicates: **1,235 LLM-RL papers** ([llm_rl_2026.md](llm_rl_2026.md)) and **525 world-model / WAM / VLA papers** ([wam_world_model_2026.md](wam_world_model_2026.md)).
3. Sorted the titles into technique families by keyword, then read every title in the training-method families by hand.
4. Checked each family against every embodied paper (WAM, VLA, world model, robot, driving) in the combined 2,248-paper set, and web-searched the closest WAM work for each gap.

Caveats:
- The lists are community-curated and lag arXiv by days to weeks.
- Classification uses titles only.
- arXiv itself was not reachable from this environment, so abstracts come from search snippets.
- "Not found" means not found in these sources and searches, not proof that no such paper exists.

## Maturity verdict

WAM RL has **not** reached the level of LLM RL. What has transferred:
- Plain GRPO/DAPO training inside a world model (WMPO, VAMPO)
- Hand-designed and self-supervised rewards (WAM-RL, PAVXploreRL, WorldCycle, WorldReward)
- Adaptive imagination at inference time (RISE-adaptive, When to Trust Imagination, GlanceWAM)
- One on-policy distillation paper with an external teacher (WAM-OPD)

What has not: the algorithm and efficiency work that makes up most of the 2026 LLM-RL literature.

## Coverage table

| Technique family | LLM-RL 2026 (primary category) | Representative LLM papers | Closest WAM / embodied 2026 work | WAM status |
|---|---|---|---|---|
| On-policy distillation & self-distillation | **439** | SDPO 2601.20802, OPSD 2601.18734, π-Distill 2602.04942, GHD 2608.06065, CRAFT 2606.29476 | WAM-OPD 2608.22364 (external teacher), PFD 2604.25859 (offline), ROAD-VLA 2606.25800, VLA-OPD 2603.26666 | **Mostly open**: no on-policy *self*-distillation for WAMs |
| Policy-optimization objectives & stability | 158 | GDPO 2601.05242, Your Group-Relative Advantage Is Biased 2601.08521, Rethinking the Trust Region 2602.04879, Target PO 2604.06159 | VAMPO 2603.19370 (GRPO), WMPO (GRPO+DAPO, 2025) | Vanilla GRPO only |
| Rewards, verifiers & label-free RL | 157 | Rubric/verifier/TTRL lines | WAM-RL, PAVXploreRL, WorldCycle, WorldReward, ReWorld, WAV, WCM | **Active**: the one area WAMs are keeping up |
| Rollout, data & sample efficiency | 63 | AERO 2602.14338, Prune as You Generate 2603.24840, VIGOR 2607.22002, Early Verdicts 2607.26253 | WISE 2609.03681, Prioritized Rollouts 2609.22879 (both VLA-in-world-model) | Partial: nothing for RL of the WAM itself |
| Entropy, exploration & diversity | 62 | Parameter-space noise 2602.02555, Bidirectional entropy modulation 2604.04894, Pass@k inversion 2607.20543 | none | **Open** |
| Analysis, theory, scaling & sparsity | 54 | Multiple Ticket Hypothesis 2602.01599, Is One Layer Enough 2607.01232, Compute–Supervision Tradeoffs 2605.25252 | Do WAMs generalize better than VLAs? 2603.22078 | **Open** (no RL-scaling or update-sparsity study) |
| Credit assignment, token/step-level | 50 | GMTS 2608.30632, DelTA 2605.21467, STARE 2606.19236, Relative Surprisal Index 2606.31575 | VAMPO (first-step-only noise) | **Open** |
| Off-policy reuse, async & systems | 43 | APER 2606.04560, Group Prioritized Off-Policy 2606.01281, RolloutPipe 2606.26997, QuRL 2602.13953 | AcceRL, RL-VLA³, D-VLA (VLA only) | Open for WAMs |
| Efficient reasoning / adaptive compute | 23 | Length control line | RISE-adaptive 2608.20430, When to Trust Imagination 2605.06222, GlanceWAM 2608.23927, Adaptive-WAM 2608.06008 | Covered (inference-time) |
| SFT–RL interplay | 16 | Sequential Beats Joint 2609.04108, Good SFT Prepares for RL 2602.01058 | WAM-RL (online video SFT), VLA-OPD | Partial |

Keyword counts in the combined 2,248-paper set make the imbalance plain:

| | LLM side | Embodied side |
|---|---|---|
| Privileged / hindsight / self-distillation | ~170 | 4 (none on-policy for WAMs) |
| Entropy and exploration | 62 | 0 |
| Token-level credit | 50 | 0 |
| RL scaling laws | several | 0 |

## Ranked directions (efficiency + accuracy)

### D1. On-policy privileged-foresight self-distillation for WAMs (recommended core)

**Idea.**
1. Run GRPO-style groups from the same start state.
2. For a failed rollout, build the teacher from the **same WAM**, with its video tokens conditioned on the observed future of a **successful sibling** rollout (fall back to the demonstration future when no sibling succeeds). The masking trick is the one PFD uses.
3. The teacher re-denoises the student's action chunks at the states the student actually visited.
4. Distill only where the student failed and the teacher recovers the successful behavior (the GHD-style gate).
5. Successful rollouts keep the ordinary GRPO advantage.

**Why it should be efficient.**
- Dense per-chunk targets from rollouts GRPO already generates: sparse success signals become dense supervision at no extra rollout cost.
- No external teacher model, no critic.
- The deployed student can stay current-only (as in PFD), so inference cost does not change.

**Why it should be accurate.**
- It trains on the policy's own state distribution, which targets compounding error (the motivation behind WAM-OPD).
- PFD already shows that the residual from a privileged future carries useful action information, even offline.

**LLM evidence.**
- SDPO 2601.20802
- OPSD 2601.18734
- π-Distill 2602.04942
- GATES 2602.20574
- HDPO 2603.23871
- RLSD 2604.03128
- Self-Distillation Zero 2604.12002
- Multi-Rollout OPD 2605.12652
- CRAFT (sibling rollouts) 2606.29476
- Distill Where You Fail 2608.00782
- GHD / The Next Screenshot Knows 2608.06065
- Sequential Beats Joint 2609.04108

**Known pitfalls to design against (all from 2026 LLM papers).**
- Biased privileged-information teachers: 2608.04794, 2608.18271, 2608.01589
- Diversity loss: 2606.26091
- Degraded reasoning or collapse: 2603.24472, 2607.10805

**Closest WAM work and the new part.**

| Paper | What it does | What this direction adds |
|---|---|---|
| PFD 2604.25859 | Offline, demo futures, adapter | On-policy data |
| WAM-OPD 2608.22364 | On-policy, frozen external teacher | Teacher is the model itself |
| ROAD-VLA 2606.25800 | VLA, self-teacher built from advantage-perturbed action logits | Future-video context |
| WAM-RL 2606.17906 | Imagined-vs-executed gap used as a *reward* | Future used as *teacher context* |

**Minimal experiment.**
- Setup: a Fast-WAM / Efficient-WAM-scale model on LIBERO + RoboTwin (the same benchmarks PFD reports).
- Baselines: GRPO (VAMPO-style), PFD, and an external-teacher OPD (WAM-OPD-style).
- Report: success rate vs. number of rollouts and vs. GPU-hours.

### D2. Sparse-update post-training for WAMs (efficiency lever for D1)

- **LLM evidence** that RL and on-policy distillation updates are sparse and low-rank:
  - Multiple Ticket Hypothesis 2602.01599
  - Is One Layer Enough 2607.01232
  - Dense Supervision, Sparse Updates 2606.13657
  - Rank-1 Trajectories 2605.21468
  - Do We Need Adam 2602.07729
  - Sparse but Critical 2603.22446
  - Low-rank dynamics 2605.06523
  - GeoRA 2601.09361
  - Orthonormal LoRA init 2606.31813
- **Forgetting risk:** Correct-Set Turnover 2606.03087.
- **WAM gap:** no study of *which* WAM module (action expert, video expert, shared layers) carries the RL/OPD update.
- **Hypothesis:** the action expert plus a few shared layers suffice. That would cut memory and compute on multi-billion-parameter video models and protect the video prior from forgetting.
- **Cost:** a cheap analysis study; it also serves as an ablation for D1.

### D3. Rollout-budget control for WAM RL

- **LLM evidence:**
  - Allocation: Adaptive Rollout Allocation 2602.01601, CoBA-RL 2602.03048, AERO 2602.14338, Hit-Utility 2605.07114, SALT 2606.05800, VIGOR 2607.22002
  - Pruning and early stopping: Prune as You Generate 2603.24840, Early Verdicts 2607.26253
  - Reuse and replay: When to Stop Reusing 2605.19425, Prompt Replay 2603.21177, APER 2606.04560, Group Prioritized Off-Policy 2606.01281
- **Closest:** WISE 2609.03681 (cuts world-model compute by 80%) and Prioritized Rollouts 2609.22879. Both post-train a *VLA* using a world model; neither applies to RL of the WAM itself.
- **WAM-specific twist:** a WAM predicts its own future, so it can issue an "early verdict" and stop hopeless or already-decided rollouts before generating the full video.
- **Novelty:** moderate; the area is getting crowded.

### D4. Exploration control for flow/diffusion WAM RL

- **LLM evidence:**
  - Parameter-space noise 2602.02555
  - Temperature policy learned from internal states 2602.13035
  - First-token diversification 2605.28295
  - Latent-space rollout diversification 2608.21595
  - Bidirectional entropy modulation 2604.04894
  - Diversity collapse via overtraining 2606.15455
  - Pass@k inversion 2607.20543
- **WAM analog:** the stochastic sampler's noise schedule and the noise in the first denoising step are the exploration knobs. Nobody has studied entropy collapse in WAM RL.

### D5. Token- and modality-level credit inside WAM outputs

- **LLM evidence:**
  - GMTS 2608.30632
  - DelTA 2605.21467
  - STARE 2606.19236
  - Relative Surprisal Index 2606.31575
  - Vision-anchored token selection 2606.03937
  - Perception-grounded PO 2604.01840
  - Non-uniform token trust region 2606.10968
- **WAM question:** a WAM outputs thousands of mostly background video latents plus a few action values. Which of them should carry RL gradient?
- **Closest:** VAMPO's first-step-only noise.

### D6. Recipe order and scaling

- **LLM evidence:**
  - Recipe order: Sequential Beats Joint 2609.04108, OPSD Compresses What RLVR Teaches 2605.06188, Nemotron-Cascade 2 2603.19220
  - Scaling and recipes: Compute–Supervision Tradeoffs 2605.25252, Demystifying RL Post-Training 2608.24949
- **WAM gap:** WAM-OPD and WAM-RL exist separately. Nobody has studied the order to apply them in, or how WAM RL scales.

## Recommendation

**Core contribution: D1.**
- It has the largest gap by volume: about 170 LLM self-distillation papers vs. zero on-policy WAM versions.
- It tells the clearest efficiency-plus-accuracy story.
- It reuses existing WAM benchmarks and baselines (PFD, WAM-OPD, VAMPO on LIBERO/RoboTwin).
- The 2026 LLM literature already lists the failure modes to design against.

**Supporting pieces:**
- D2 as the efficiency lever: update only the action expert plus a few shared layers.
- D3's skipping of zero-variance groups as a cheap add-on.

**Main risk:** PFD (Apr 2026) and WAM-OPD (Aug 2026) are close neighbors. The paper has to show that being on-policy, self-taught and outcome-gated beats both, and it should cite them prominently.
