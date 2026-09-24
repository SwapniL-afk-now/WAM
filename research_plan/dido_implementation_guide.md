# DIDO: implementation guide extracted from the paper

Source: *DIDO: Distilling Interaction-Centric Dynamics into One-Step Denoising for World Action Models*, arXiv 2609.15570, preprint PDF supplied by the user (§3, App. A–C, Table 4). The official code is not released (`github.com/LoveJu1y/DIDO-WAM` has only a README). Everything below is taken from the paper; items the paper does not specify are marked **UNSPECIFIED**.

## 1. Architecture

- **Video model:** Wan2.2-TI2V-5B, initialised from its *public 4-step distilled checkpoint* (the paper cites Li et al. 2025b; **UNSPECIFIED** which repo; candidates are Wan2.2-TI2V-5B-Turbo and LightX2V).
- **Action expert:** coupled via **mixture-of-transformers** (MoT) with **shared attention** at every layer.
  - Same depth as the video model, **1/4 of its hidden width**.
  - Output dimension 7 (LIBERO) or 14 (RoboTwin). Chunk length H_a = 32.
  - Inference integrates **10 Euler steps**; the first **10** actions are executed on LIBERO, **24** on RoboTwin.
  - The action loss follows Fast-WAM (Eq. 8): `a_τ = (1-τ)ε + τa`, target `u = a - ε`, τ ~ U[0,1].
- **Attention mask** (Fig. 6):
  - Stage I: interaction tokens, current-frame and future-frame tokens attend within the world-model stream.
  - Stage II: action tokens attend to **all** world-model tokens and to each other. World-model tokens **cannot** attend to action tokens.
- **Interaction-centric tokens:** 64 learned tokens inserted into the video transformer, in 4 groups of 16: `T_obj`, `T_int`, `T_grip`, `T_align`.
  - **Box heads:** `[T_obj; T_int]` → object boxes and `[T_grip; T_int]` → gripper boxes. Each head is a 2-layer MLP (GELU, hidden width = video width) predicting **normalised boxes for 32 future steps**.
  - **Alignment:** `T_align` at the layers in set S → projection `P_l` → DINOv3 token space. S is **UNSPECIFIED**, as are the P_l architecture and the DINOv3 variant.
- **Dynamics-based token refinement** (applied to future-video tokens only, before the action expert sees them):
  - `V_t0` and `V_tk` are the value features of the current frame and the k-th future frame; k is **UNSPECIFIED**.
  - Cell score `s(r) = ||V_tk(r) − V_t0(r)||_2`.
  - The feature map is split into a **4×5 grid of regions, each 2×2 cells**. A region's score is the sum of its 4 cell scores.
  - Keep full resolution in the **top-3** regions; apply **2×2 average pooling** to the other 17 regions. The same pooling is applied to the K and V features seen by the action expert.
  - Effect: context shrinks from 323 to 191 tokens and latency drops by about 30 ms.
  - The 4×5×(2×2) layout implies an 8×10 spatial token grid per frame, which depends on DIDO's resolution. **Adapt the grid to FastWAM's grid** (7×14 tokens per frame at 224×448, patch 2).

## 2. Losses

- **Box loss:** `L_box = (1/H_a) Σ_h [ smoothL1(b̂_h, b_h; β=1.0) + λ_giou · GIoU(b̂_h, b_h) ]`
- **Interaction loss:** `L_inter = λ_obj · L_box(obj) + λ_grip · L_box(grip)`
- **Alignment loss:** `L_align = (1/|S|) Σ_{l∈S} || P_l(T_align^l) − F_DINO ||_1` (L1 against frozen DINOv3 tokens of the object crop)
- **Stage I:** `L = L_DMD + L_inter + λ_align · L_align`
- **Stage II:** `L = λ_video · L_video + λ_act · L_act + L_inter + λ_align · L_align`, where L_video is Fast-WAM's video flow-matching loss.

## 3. DMD details (Stage I; DMD2-style, App. B.1 and C.2)

- **Networks:** generator `G_θ`, frozen teacher `f_T` (the real score) and online fake score `f_F`. All three start from the robot-domain 4-step teacher.
- **Generator step:** one forward pass at **timestep 1000**, `x_g = G_θ(z, c)`.
- **Score timesteps:** `τ ~ U{1000, 750, 500, 250}` (the teacher's grid), mapped to scheduler t. One timestep is sampled **per example** and shared across its latent frames.
- **Noising:** `x_t = (1−σ_t) x_g + σ_t ε`. Convert Wan flow predictions to x̂0 for both `f_T` and `f_F`.
- **Teacher CFG:** `ε_cond + 6(ε_cond − ε_uncond)`, i.e. guidance scale 7. The fake score uses **no CFG**.
- **Normalised gradient:** `g = (x̂0_F − x̂0_T) / mean|x_g − x̂0_T|`, injected through a stop-gradient surrogate: `L_DMD = ½‖x_g − sg(x_g − g)‖²`.
- **Fake-score loss:** `L_fake = ‖f_F(x_t,t,c) − sg(x_g)‖²` (x0 regression) at an independently sampled t.
- **Update schedule:** the fake score updates every iteration, the generator **every 5th** iteration.
- **Other:** no GAN / discriminator. "Backward simulation" is used; with a one-step generator this is simply the generator's output.

## 4. Data annotation (App. A)

- **Gripper boxes** (LIBERO / RoboTwin): project the simulator gripper pose into the image.
- **Object boxes:** follow the LaRA-VLA pipeline. Parse the target from the instruction, detect it in a reference frame with an open-vocabulary detector, then track it through the video.
  - Real robots: Grounding DINO + SAM 3, tracked both directions, highest-confidence box kept, gaps filled by linear interpolation.
  - Reported quality: LIBERO objects correct in 95/100 trajectories, grippers in 100/100.
- **DINOv3 targets:** crop the object box, enlarge by a fixed margin (**UNSPECIFIED**), resize to the DINOv3 input size, encode to tokens, and cache them.

## 5. Hyperparameters (Table 4; all phases AdamW, weight decay 0.01, bf16)

| | LIBERO | RoboTwin |
|---|---|---|
| **Teacher prep:** LR / betas | 3e-6 / (0.0, 0.999) | same |
| schedule / steps / global batch | 500 warm-up, const / 3K / 384 | 500 warm-up, const / 4K / 2304 |
| λ_obj / λ_grip / λ_giou / λ_align | 0.05 / 0.05 / 0.5 / 0.02 | same |
| **Stage I:** generator LR / fake-score LR | 1e-6 / 1e-7 | same |
| betas / schedule | (0.0, 0.999) / 500 warm-up, const | same |
| generator update interval / steps / global batch | 5 / 3K / 128 | 5 / 3K / 1536 |
| λ_obj / λ_grip / λ_giou / λ_align | 0.03 / 0.03 / 0.5 / 0.01 | 0.045 / 0.045 / 0.5 / 0.01 |
| **Stage II:** LR / betas | 1e-4 / (0.9, 0.95) | same |
| schedule / steps / global batch | 5% warm-up, cosine / 14,480 / 384 | 5% warm-up, cosine / 90K / 384 |
| λ_obj / λ_grip / λ_giou / λ_align | 0.03 / 0.03 / 1.0 / 0.02 | same |
| λ_video / λ_act | 0.5 / 1.0 | same |
| **Compute** (H100s, count **UNSPECIFIED**) | 15 h + 5 h + 7 h | 25 h + 12 h + 100 h |

Interaction supervision is already on during teacher preparation. LIBERO trains **one model on all 4 suites**, with no extra data.

## 6. Reported results

- **LIBERO-Plus (Table 2):**

| Method | Camera | Robot | Language | Light | Background | Noise | Layout | Average |
|---|---|---|---|---|---|---|---|---|
| **Fast-WAM** | 53.7 | 16.4 | 68.9 | 78.2 | 60.7 | 44.5 | 37.7 | **51.5** |
| **DIDO** | 45.3 | 74.3 | 94.4 | 96.6 | 72.7 | 75.0 | 83.3 | **76.6** |

  Faster-WAM averages 75.0 and ST-WAM 72.8.
- **Ablation (LIBERO / LIBERO-Plus):**
  - 1-step truncation: 97.7 / 71.4
  - + DMD: 98.3 / 72.3
  - + interaction reasoning: 98.9 / 75.9
  - + token refinement: 99.0 / 76.6
  - The 4-step teacher reaches 98.4 on LIBERO.
- **Latency** (H100, batch 1): DIDO 384 ms vs 562 ms for the 4-step teacher (−32%) and 356 ms for Fast-WAM.

## 7. What this means for IMAGO

- DIDO is the same design family as FastWAM: a Wan2.2-TI2V-5B video expert, a narrow MoT action expert, and Fast-WAM's L_video.
- **Base FastWAM scores only 51.5 on LIBERO-Plus.** That leaves large headroom for IMAGO's outcome-only RL. DIDO (76.6) and Faster-WAM (75.0) are the numbers to beat.
- **Speed:** a DIDO-style 1-step video expert would cut IMAGO's rollout cost by roughly the 4→1 video-step share, about 32% end to end in DIDO's measurement. It is not 4×.
- **Mapping onto our codebase if we implement it:**
  - teacher = FastWAM video expert fine-tuned at 4 steps;
  - `G_θ` / `f_F` = copies of the video expert;
  - tokens and heads added in the video-expert prepare/blocks;
  - token refinement adapted to FastWAM's 7×14 per-frame grid.
- **Memory on 2–3 × 96 GB:** Stage I holds three 5B video models, two of them trained. Full fine-tuning with AdamW needs about 200 GB plus activations. That means FSDP full-shard across 3 GPUs, or LoRA on `G_θ` / `f_F`, which would be a deviation from the paper.
