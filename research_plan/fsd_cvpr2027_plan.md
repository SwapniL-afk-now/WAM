# Research Plan: On-Policy Foresight Self-Distillation for World Action Models

Target: CVPR 2027 (paper deadline historically mid-November; confirm the official date). Written 2026-09-24.

---

## 0. Novelty check: has on-policy self-distillation (OPSD) been done for WAMs?

**Verdict: no paper found (as of 2026-09-24) in which a WAM is post-trained on its own rollouts with itself as the teacher and a real observed future as privileged input.**
The gap is narrow, though. PFD already uses the exact same kind of teacher, just offline.

How the check was done: web searches (arXiv pages are blocked in this environment, so they were read via search-engine summaries and the papers' GitHub READMEs), plus the 525-paper 2026 WAM/world-model list in `literature/wam_world_model_2026.md`.

| Paper | What it does | Same model as teacher? | Privileged future? | On-policy (own rollouts)? | Uses reward / RL? | Gap left for us |
|---|---|---|---|---|---|---|
| **PFD** 2604.25859 (Apr) | Same backbone with two attention masks: the teacher sees the true future video, the student sees only the current frame; the difference is distilled into an adapter | Yes | Yes (demo future) | **No** (offline demos) | No | On-policy states, failure rescue, reward |
| **WAM-OPD** 2608.22364 (Aug, UCL) | A frozen teacher WAM labels the histories of a fast student (Flash-WAM) | **No** (separate frozen teacher) | No | Yes | No | Self-teacher, privileged future, reward |
| **WMSD** 2606.12072 (Jun) | Solution-text-conditioned "Demonstrator" distilled into an instruction-conditioned "Executor", plus RL | Yes | Text solution | Yes | Yes (VLM reward) | Video-generation tasks only; **no actions, no robots** |
| **ROAD-VLA** 2606.25800 (Jun) | VLA self-distillation; the teacher perturbs logits with advantages. Reports that **text-based privileged teachers fail for VLAs** | Yes | No (advantage-based) | Yes | Yes | Not a WAM; no visual privileged information |
| S-VAM 2603.16195 | Distills the model's own multi-step foresight into one step | Yes | Own generated video | No | No | Speed only |
| WAM-RL 2606.17906 | RL with a reconstruction reward, plus online video SFT on successful rollouts | – | – | Yes | Yes | No teacher / distillation |
| Motus2 2608.30237 | Policy + simulator + value model loop | – | – | Yes | Yes | No self-distillation |
| SelfWAM 2608.00725 | Self-mask prediction to ground the future on the robot | – | – | No | No | Unrelated ("self" = the robot's body) |
| Vid2WAM, Flash-WAM, DIDO, ForeTime-VLA | Distillation for speed or priors (offline) | No | – | No | No | Unrelated to OPSD |

**What is new in this project:** we move PFD's privileged-future teacher **on-policy**, use it to **rescue failed rollouts** with futures taken from successful rollouts, and **combine it with RL** as one objective.

**Scoop risk: high.** PFD's on-policy version is the obvious next paper for the PFD authors (Fang, Chen, Cai). WAM-OPD's authors could also add a self-teacher. Check arXiv weekly (search "self-distillation" + "world action") and post to arXiv on the submission day.

Two facts that shape the plan:
- The PFD README reports **98.1% on LIBERO**, so LIBERO is saturated. The headline numbers must come from **RoboTwin 2.0** and **LIBERO-plus**; LIBERO serves only as a sanity check.
- ROAD-VLA's finding that text-based privileged teachers fail for VLAs supports our choice: **the privileged information has to be visual, in the model's own modality.**

---

## 1. The claim (one sentence)

> A world action model is its own best teacher: conditioned on a *real* future instead of its *imagined* one, the same network produces better actions, and distilling this on the policy's own rollouts gives dense, critic-free supervision that makes WAM RL more sample-efficient and more accurate, at no inference cost.

Working titles:
- "World Action Models Are Their Own Teachers"
- "Foresight Self-Distillation: On-Policy Post-Training for World Action Models"

Method name: **FSD** (Foresight Self-Distillation).

---

## 2. Scientific hypotheses

- **H1 (distribution shift, not capacity).** PFD found that adding capacity does not close the gap between the teacher that sees the true future and the student that sees only the current frame. We hypothesize that the gap comes from *state distribution*: the student acts in states the demonstrations never visit. The teacher/student gap should therefore be *larger on the policy's own states than on demo states*, and on-policy distillation should close it where offline PFD cannot.
- **H2 (failure rescue).** In a state where a rollout failed, giving the teacher the real future of a *successful* rollout from a nearby state yields actions that more often recover success than the student's own actions.
- **H3 (signal where GRPO has none).** A group where every rollout succeeded, or every one failed, gives GRPO zero advantage. FSD still gets a signal from it: the own real future when all succeed, and success-buffer futures when all fail. This improves sample efficiency.
- **H4 (the imagination gets better too).** Training actions under the model's own imagined future (as at deployment), with a teacher that sees the real future, also improves video prediction and action–video consistency.

---

## 3. Method

### 3.1 Setup

- The WAM jointly denoises future video latents `v` and an action chunk `a` with flow matching, conditioned on observations `o_{≤t}` and the instruction `ℓ`.
- The action velocity field is `u_θ(a_τ, τ | o_{≤t}, ℓ, c)`, where `c` is the video context the action tokens attend to:
  - **Student / deployment:** `c = v̂`, the model's own imagined future (or current frame only, for current-only variants such as Fast-WAM or PFD).
  - **Teacher:** `c = F`, clean real future frames. Implemented exactly as in PFD: same weights, different attention mask or inputs.
- Teacher weights `θ̄` are a stop-gradient copy: either the current weights or an EMA (ablated).

### 3.2 One training iteration

1. **Rollouts.** Sample `M` start states. From each, run `G` episodes (G = 8) with the stochastic (SDE) sampler, recording per-chunk observations, the student's actions, its imagined futures, and the episode reward `R_i ∈ {0,1}` (optionally with a dense progress term).
2. **Choose the privileged future `F_{i,k}` for each chunk k of episode i:**
   - **Successful episode (R_i = 1):** its **own** real future `o_{t_k+1 : t_k+H}` (hindsight).
   - **Failed episode (R_i = 0):** a **success transplant**. Take the chunk `k'` of a successful episode `j` (a sibling in the same group, else the success buffer `B` for this task) that minimizes `d = ‖φ(o_{i,t_k}) − φ(o_{j,t_k'})‖`, where `φ` is a frozen DINOv2/SigLIP embedding of the frame. Use `j`'s real future after `k'` if `d < δ`.
   - **Fallback:** the demonstration's future at its nearest state (this is PFD's case). If nothing is within `δ`, the chunk gets no distillation.
3. **Gate `w_{i,k}`:**
   - 1 for hindsight chunks.
   - `exp(−d/σ)` for transplanted chunks.
   - Optional confidence gate: drop the chunk if the teacher's actions vary a lot across noise seeds.
4. **Losses.**
   - Self-distillation (velocity matching on the student's own noised action path; the student sees its own imagined future, as at deployment):
     `L_SD = Σ w_{i,k} · E_τ ‖ u_θ(a_τ, τ | o, ℓ, v̂) − sg[u_θ̄(a_τ, τ | o, ℓ, F_{i,k})] ‖²`
   - RL on actions: Flow-GRPO-style with group-normalized advantages `Â_i = (R_i − mean_G R)/std_G R`, or advantage-weighted flow matching, which is simpler and avoids SDE log-probabilities (pick whichever the codebase supports in week 1).
   - Optional grounding of the video branch: flow-matching loss on the observed videos of successful rollouts (as in WAM-RL's online video SFT).
   - **Total:** `L = L_RL + λ·L_SD + β·L_vid`, with a KL / regression anchor to the SFT model if training becomes unstable.
5. **Update only a sparse subset** (the D2 idea): the last N action and video layers, as in PFD's released "12×12 partial" config, or LoRA. This cuts memory and compute and is itself an ablation.
6. **Success buffer:** add successful episodes to `B` (FIFO, per task).

### 3.3 One objective that covers the prior methods

| Setting of FSD | What it reduces to |
|---|---|
| λ = 0 | Flow-GRPO / VAMPO-style WAM RL |
| No RL; `F` = demo future; offline data | **PFD** |
| No RL; teacher = a different frozen model; `F` = none | **WAM-OPD** |
| Keep successful rollouts, plain regression | Rejection-sampling fine-tuning / WAM-RL online SFT |
| **On-policy; `F` = own or transplanted real future; gated; + RL** | **FSD** |

This table is the "field map" figure. It makes the paper the reference point for the existing methods.

### 3.4 Inference

Unchanged. The student interface, latency and parameters are the same as the base model's.

---

## 4. Experimental setup

### 4.1 Codebase and backbones (all open)

| Role | Codebase | Why |
|---|---|---|
| **Primary** | **PFD / FastWAM code** (`github.com/PengchengFang-cs/PFD`; Wan2.2 backbone; LIBERO + RoboTwin loaders; released checkpoint) | The teacher/student masking is already implemented, and it gives a direct comparison with PFD on the same code |
| **Second backbone** | **OpenWAM** (`github.com/OpenWAM-Official/OpenWAM`; Wan2.2-5B, Wan2.1-1.3B or Cosmos-Predict2.5-2B video backbones; RoboTwin (50 tasks), LIBERO, LIBERO-plus, RoboCasa, VLABench evaluation; weights on HF) | Shows the method works across architectures. The **1.3B / 2B backbones** make RL affordable |
| Optional third | Efficient-WAM (1B) or Fast-WAM, if code is out | Small-model point |

### 4.2 Benchmarks

- **RoboTwin 2.0**, the main benchmark: choose 10–16 tasks where the base model is at 20–70% success (room to improve), including randomized-clutter settings.
- **LIBERO-plus** (perturbation robustness), a strong CVPR angle: on-policy training should improve robustness.
- **LIBERO** (4 suites): sanity check, since it is saturated at about 98%.
- Optional: RoboCasa or VLABench through OpenWAM, and one real-robot task.

### 4.3 Baselines (all on the same backbone and budget)

1. Base SFT checkpoint
2. SFT for the same extra number of steps (compute-matched)
3. **PFD** (offline privileged foresight)
4. **RL only** (FSD with λ = 0), plus a variant that drops zero-signal groups
5. **Rejection-sampling fine-tuning** (regress on successful rollouts only)
6. **WAM-OPD-style**: frozen larger teacher, e.g. OpenWAM 5B teaching 1.3B, if compute allows
7. PFD followed by RL (to show that the combination inside one objective beats running them one after the other)

### 4.4 Metrics

- Success rate: mean ± std over 3 seeds; the RoboTwin and LIBERO-plus protocols.
- **Efficiency:** success vs. number of environment rollouts, and vs. GPU-hours. Headline: "reaches RL-only's final success with N× fewer rollouts."
- Inference latency (to show it is unchanged).
- Video: FVD / PSNR / LPIPS of imagined vs. real futures on held-out rollouts; **action–video consistency** (an inverse-dynamics model applied to the imagined video vs. the executed actions).
- Diversity: action variance within a group across training (to catch collapse).

### 4.5 Ablations

| # | Ablation | Question |
|---|---|---|
| A1 | Source of `F`: own-only / transplant-only / success buffer / demo-only / all | Which privileged future matters? |
| A2 | Gate on/off; the δ threshold; confidence gate | Does gating prevent teaching from bad matches? |
| A3 | Teacher = current weights vs. EMA | Stability |
| A4 | λ sweep; RL on/off | Complementarity |
| A5 | Student sees imagined `v̂` vs. current frame only | Does training under the model's own imagination matter (H4)? |
| A6 | Dense future (H frames) vs. goal-frame only | How much privileged information is needed? |
| A7 | Sparse update: last 6 / 12 / all layers; LoRA | Efficiency lever (D2) |
| A8 | Group size G = 4 / 8 / 16 | Rollout budget |
| A9 | Handling of all-fail / all-success groups: drop vs. FSD signal | H3 |

### 4.6 Analyses (the figures people will reuse)

1. **Teacher/student gap on demo states vs. on-policy states** (H1): the key motivation figure.
2. **Rescue rate:** when a failed state gets the transplanted-success teacher's action chunk and the student then continues, how much does success rise? Plotted over training (H2).
3. Where in the episode distillation helps most (per chunk position), with qualitative videos of imagined vs. real vs. teacher-corrected rollouts.
4. Fraction of zero-advantage groups over training, and FSD's gain on exactly those groups (H3).
5. Video quality and action–video consistency before and after (H4).

---

## 5. Pilot and go / no-go (week 2)

Run on 3–4 RoboTwin tasks with the released FastWAM/PFD checkpoint. No training, just measurements:

- **P1 (H1):** compare teacher-with-real-future and student action errors on demo states vs. on states from the model's own rollouts. Also try the executable version: replace the student's chunk with the teacher's and measure success.
- **P2 (H2):** rescue rate with transplanted success futures, sweeping δ.

**Go** if the teacher beats the student by **≥ 10 success points** on the model's own states in P1 or P2.
**No-go:** switch the thesis to the goal-frame variant (A6), or to a pure systematic study of RL for WAMs (the fallback paper type below). Decide this by **October 8**.

---

## 6. Compute (estimate; measure precisely in week 1)

- Rollouts dominate cost. Each episode is 20–40 policy calls with multi-step denoising. Use the **1.3B / 2B backbone** for the main sweeps and the **5B** one for the headline table.
- Train only the last N layers (as PFD does) with ZeRO-1.
- Plan for **≥ 8×80 GB GPUs for 5 weeks**; a second 8-GPU node makes the ablations feasible.
- Cost savers:
  - vectorized simulator environments (many parallel env instances)
  - fewer denoising steps during rollouts
  - teacher forward passes only on gated chunks
  - reusing rollouts for 2 update epochs, with ratio clipping

---

## 7. Timeline (7 weeks; deadline assumed to be about November 12, 2026)

| Week | Dates | Deliverable | Gate |
|---|---|---|---|
| 1 | Sep 25 – Oct 1 | Set up the PFD/FastWAM and OpenWAM (1.3B) repos; reproduce the base model and PFD on 4 RoboTwin tasks and LIBERO; parallel rollout collection; RL-only (λ = 0) running | Reproduced numbers within ±3 points |
| 2 | Oct 2 – Oct 8 | **Pilot P1/P2**; success buffer + state matching; FSD v0 on 4 tasks | **Go / no-go** |
| 3 | Oct 9 – Oct 15 | FSD full loop, gating, sparse update; first efficiency curve vs. RL-only | FSD > RL-only on ≥ 3 of 4 tasks |
| 4 | Oct 16 – Oct 22 | Scale to 10–16 RoboTwin tasks + LIBERO-plus; all baselines launched | Main table draft |
| 5 | Oct 23 – Oct 29 | Ablations A1–A9 (on the 1.3B backbone, 4 tasks); analyses 1–4 | Figures drafted |
| 6 | Oct 30 – Nov 5 | Second backbone (5B headline); video metrics; optional real robot; **full draft by Nov 3** | Internal review |
| 7 | Nov 6 – Nov 12 | Polish, supplementary video, code cleanup, **arXiv on the day of submission** | Submit |

If week 1 slips by more than 5 days, drop the second backbone and the real robot; don't drop the ablations. If week 2 is a no-go, retarget to ICML 2027 (late January) with the fallback paper.

---

## 8. Paper outline

1. **Intro.** WAMs imagine the future, then act. At training time we *have* the real future. Teaser figure: imagined vs. real-future teacher on the same failed state, plus the rescue.
2. **Related work:**
   - WAMs
   - RL for WAMs/VLAs (VAMPO, WAM-RL, World-Gymnast, Flow-GRPO)
   - privileged / self-distillation (PFD, WAM-OPD, OPSD/SDPO in LLMs, ROAD-VLA)
3. **Method:** the unifying objective (Table §3.3), choice of privileged future, gating, sparse update.
4. **Why on-policy:** the gap-on-own-states analysis (H1).
5. **Experiments:** main table (RoboTwin, LIBERO-plus, LIBERO, 2 backbones), efficiency curves, ablations, video analysis.
6. **Limitations:**
   - needs real rollouts (simulation, or a real robot with a success detector)
   - state matching may fail in highly diverse scenes
7. **Supplement:** hyperparameters, per-task numbers, videos, pseudocode.

## 9. Anticipated reviewer objections

| Objection | Answer prepared in advance |
|---|---|
| "It's PFD + RL." | Compute-matched: FSD > PFD, FSD > RL-only, and FSD > PFD→RL run in sequence. H1 analysis shows *why* on-policy matters. The transplant/rescue part has no counterpart in PFD. |
| "Only simulation." | Two simulation suites + robustness suite + (ideally) one real-robot task. Inference is unchanged, so deployment claims are safe. |
| "Teacher bias / collapse (known in LLM OPSD)." | Gating, EMA teacher, the RL term as a corrective, and a diversity plot (A3, A9). |
| "Unfair compute." | All curves plotted against rollouts *and* GPU-hours; same number of trainable layers. |
| "Why not a value model?" | Critic-free by design; compare the cost with a critic baseline if time allows. |

## 10. Fallback paper (if the week-2 pilot is a no-go)

**"What Matters in RL for World Action Models: A Large-Scale Empirical Study,"** an analysis-style paper like the well-known "What Matters in On-Policy RL" study for continuous control. It uses the same infrastructure to compare:
- RL objectives (Flow-GRPO, advantage-weighted matching, rejection-sampling fine-tuning, PFD)
- rollout allocation
- sparse updates
- group size
- whether the video branch is trained

It is less flashy, but it is heavily cited and would target ICML 2027.

## 11. Immediate next actions

1. Confirm the CVPR 2027 deadline and the compute budget.
2. Clone PFD and OpenWAM; download the FastWAM RoboTwin data and the PFD checkpoint.
3. Read PFD, WAM-OPD and WMSD in full (the PDFs could not be reached from this environment) to confirm the novelty table above, especially whether PFD has an on-policy section or a stated follow-up.
4. Set up a weekly arXiv alert for "world action model" + "self-distillation" / "on-policy".

## Sources

- PFD: https://arxiv.org/abs/2604.25859 and https://github.com/PengchengFang-cs/PFD
- WAM-OPD: https://arxiv.org/abs/2608.22364
- World Model Self-Distillation: https://arxiv.org/abs/2606.12072 and https://github.com/sebastian-stapf/world-model-self-distillation
- ROAD-VLA: https://arxiv.org/abs/2606.25800
- S-VAM: https://arxiv.org/abs/2603.16195
- WAM-RL: https://arxiv.org/abs/2606.17906
- Motus2: https://arxiv.org/abs/2608.30237
- SelfWAM: https://arxiv.org/abs/2608.00725
- OpenWAM: https://arxiv.org/abs/2609.07398 and https://github.com/OpenWAM-Official/OpenWAM
- CRPO (exposure bias in OPSD): https://arxiv.org/abs/2607.28026
- Reinforcement Learning via Self-Distillation: https://arxiv.org/abs/2601.20802
