# Experiment log

One entry per experiment: what changed, why, result, takeaway. Newest last.
Metrics are IoU on the **test** split (nuScenes `mini_val`: scene-0103, scene-0916, 81 frames) unless stated.
mIoU = mean of vehicle and pedestrian IoU. Model selection uses the held-out **val** scene (scene-0655).

## Setup (shared by all experiments)
- Data: nuScenes v1.0-mini, 6 cameras, keyframes only.
- Target: 200×200 BEV, 0.5 m/cell, ±50 m, ego-centred. Classes: background / vehicle / pedestrian, rasterised from 3D boxes (boxes with 0 lidar+radar points skipped).
- Input ("sparse BEV"): per camera, Depth-Anything-V2 metric depth + SegFormer-B2 (Cityscapes) → unproject → ego frame → 4-channel occupancy (flat, vehicle, person, obstacle), height filter z ∈ [-1, 3] m.

---

## E0 — Projection-only baselines (2026-10-15, commit 76662e3)
- **A:** label a cell vehicle/pedestrian if any projected point of that class lands in it.
- **A+:** same, 3×3 dilation.
- **Result:** A veh 0.096 / ped 0.004 / mIoU 0.050; A+ 0.097 / 0.004 / 0.051.
- **Takeaway:** monocular depth smears objects along camera rays → streaks, not boxes. Dilation doesn't help.

## E1 — U-Net baseline, val = test (2026-10-15, commit 76662e3) — superseded
- 1.9M-param U-Net, weighted CE (inverse-sqrt freq) + soft Dice, AdamW, cosine LR, left-right flip.
- Checkpoint selected on mini_val (same as reported split → leak).
- **Result:** veh 0.178 / ped 0.030 / mIoU 0.104 (optimistic).

## E2 — U-Net baseline, proper splits (2026-10-15, commit 9d90543)
- Change: hold out scene-0655 for model selection/early stopping; mini_val used only for reporting. Train 282 / val 41 / test 81 frames.
- **Result:** veh **0.151** / ped **0.041** / mIoU **0.096**. Best epoch 19, early stop 53.
- **Takeaway:** overfits quickly (val loss ~1.5 vs train ~1.0). Predicts blobs in the right places but merges adjacent cars; misses most pedestrians.

## E3 — Diffusion: SD1.5 + ControlNet on sparse BEV (2026-10-15) — negative result
- Setup: frozen SD1.5 VAE + UNet (bf16), trainable ControlNet (361M params, 4-ch conditioning), fixed prompt, no CFG, 256 px, DDIM 25 steps. Class map painted as 3 colours → VAE latent; decoded output snapped to nearest palette colour.
- **VAE ceiling gate** (target → VAE → snap, no diffusion), test: 256 px veh 0.932 / ped 0.736; 512 px veh 0.970 / ped 0.851. Passed at 256. (A first run showed 0.62 due to an off-by-one-cell resize bug in `snap()` — fixed.)
- Runs: lr 1e-5 (loss flat for 2k steps, killed) → lr 1e-4, 8.5k steps → lr 1e-4 + shift & point-dropout augmentation, ~7k steps. ~40 min / 6k steps, 8.6 GB VRAM.
- Sampler fix: SD's schedule keeps ~7% of the clean latent at t=T (non-zero terminal SNR); starting DDIM from pure noise gave garbage even on train frames. Now start from the blank-map latent noised to the first DDIM step.
- **Result (test):** 1 sample veh **0.037** / ped 0.001 / mIoU 0.019; 4-sample vote 0.003 / 0.000 / 0.001 (samples disagree on locations, voting erases cars).
- **Diagnosis:** outputs are realistic, crisp car boxes but placed in the wrong places. One-step x0 prediction: veh 0.49 on train frames vs 0.002 on test → the ControlNet memorises 282 training frames instead of learning sparse→BEV geometry. ControlNet conditioning (zero-init residuals into a frozen natural-image UNet) is too weak a pathway for this little data.
- **Takeaway:** learned the *look* of BEV maps, not the *where*. Needs a stronger conditioning path and far fewer parameters — see E4.

## E4 — Pixel-space conditional diffusion, same U-Net backbone as B (2026-10-15, commit 849defd)
- Setup: Palette/SR3 style, trained from scratch. x0 = one-hot target in {-1,+1}; denoiser input = concat(noisy map 3ch, sparse 4ch) = 7 ch; output 3 channels = x0 class logits. Backbone = B's U-Net (widths 32-64-128-256) + sinusoidal timestep embedding → MLP → per-channel bias in every conv block (2.05M params; B's code path/checkpoint unchanged). Hand-written cosine schedule (T=1000), q_sample, DDIM (20 steps, eta=0, from pure noise). Loss = B's weighted CE + Dice on the x0 logits. AdamW lr 1e-3 cosine, batch 16, flip, 300 epochs; validation (full 20-step DDIM) every 10 epochs on scene-0655, best by val mIoU.
- Cost: 6.7 min training, 2.0 GB VRAM (peak allocated).
- **Result (test):** 1 sample veh **0.117** / ped 0.007 / mIoU **0.062**; 4-sample mean-of-softmax veh 0.129 / ped 0.002 / mIoU 0.066. (B: 0.151 / 0.041 / 0.096; E3: 0.019.) Best val mIoU 0.067 (epoch 230); val curve is noisy (0.01-0.07), ped val IoU 0 throughout.
- **Probe (one-step x0 IoU, mIoU):** t=50: train/val/test all 1.000 (the noisy map almost contains the answer, so this is trivial). t=999 (pure noise, i.e. sparse-only): train 0.116 (veh 0.161) / val 0.062 / test 0.051 (veh 0.102). E3 probe was train 0.49 vs test 0.002 at t=999: E4 does not show E3's memorise-vs-nothing split, and test is above zero, but the t=999 predictor (the part that decides *where* objects go) is weaker than B (veh 0.151).
- **Takeaway:** the concatenated-condition path fixes E3's failure (objects are in roughly the right places, crisp rectangular boxes instead of B's blobs), but it does not beat the plain U-Net. Most training steps are at low/mid t where the noisy map leaks the answer, so the hard sparse→BEV mapping is learned from only a small fraction of the gradient; pedestrians are essentially lost. Under this metric (strict per-cell IoU) sharper-but-misplaced boxes score below blurry-but-centred blobs. Controlled B vs D comparison: diffusion objective alone gives no gain on 282 frames.

## E5 — Diffusion refiner conditioned on B's softmax (2026-10-15, commit 7148b56)
- Setup = E4 + one change: denoiser input = concat(noisy map 3ch, sparse 4ch, **B's softmax probabilities 3ch**) = 10 ch (2.05M params, +~1k in the first conv). Everything else identical to E4 (cosine schedule, x0-logit loss with B's weighted CE+Dice, AdamW lr 1e-3, batch 16, flip, DDIM 20 steps, seed 0, same sampling generator seeds). B (`checkpoints/unet/best.pt`, frozen) is run once per frame and its softmax kept in memory; flip augmentation flips the probabilities together with sparse/target. Code: `--refiner` flag in `train_pixel_diffusion.py` (sets `condition_on_unet=True`, switches to `checkpoints/pixel_diffusion_refiner` / `runs/pixel_diffusion_refiner`); the condition is just a 7-channel "sparse" argument, so `models/pixel_diffusion.py` sampling/loss are unchanged. B, C, D rows reproduce exactly.
- Cost: 3.2 min (early stop at epoch 140, best val mIoU 0.095 at epoch 60), 2.0 GB VRAM. Val curve noisy (0.079-0.095), ped val IoU ~0.
- **Result (test):** 1 sample veh **0.136** / ped 0.015 / mIoU **0.076**; 4-sample mean veh 0.141 / ped 0.013 / mIoU 0.077. (B 0.151 / 0.041 / 0.096; D 0.117 / 0.007 / 0.062 and 0.129 / 0.002 / 0.066.)
- **Probe (one-step x0 mIoU):** t=50 train/val/test ~1.0; t=999: train 0.244 (veh 0.452) / val 0.093 (veh 0.186) / test 0.070 (veh 0.137, ped 0.003). Versus E4 t=999 test veh 0.102.
- **B train vs test mIoU (the stacking mismatch):** train 0.275 (veh 0.414 / ped 0.137), val 0.112 (0.203 / 0.021), test 0.096 (0.151 / 0.041). B is ~3x better on its own training frames, so the refiner learns to trust B more than it should; its t=999 probe shows the same train/test gap (0.244 vs 0.070).
- **Takeaway:** conditioning on B recovers most of the E4 deficit (mIoU 0.062 -> 0.076, veh 0.117 -> 0.136) but does not beat B itself (0.096); the refiner mostly copies B's placement, drops some of B's blobs and loses most pedestrians. Differences of ~0.01 are within the noise of 81 test frames / one seed. Train/test mismatch is likely the limiter; fixing it would need out-of-fold B predictions (e.g. scene-wise K-fold), not run here.

---
# v2: full BEV (map + objects) on nuScenes trainval

## E6 — Data v2 (2026-10-15, commit e8c9c86)
- Goal change: predict the *whole* BEV from 6 cameras, not just objects. Data: trainval camera blobs 01-04 (keyframes only), 340 scenes, 13,529 frames. Split by scene: test = official val scenes (1,982 frames), val = seeded 8% of train scenes (919), train 10,628.
- Sparse input v2: 6 ch (road, sidewalk, terrain, vehicle, person, obstacle) from Depth-Anything-V2 metric depth + SegFormer-B2 Cityscapes, same unprojection as v1.
- Target v2: multi-label 6 layers — drivable_area, ped_crossing, walkway, divider (road+lane dividers, dilated) from the nuScenes map expansion v1.3, plus vehicle / pedestrian boxes. Loss for all models: BCE with per-layer pos_weight (sqrt neg/pos) + soft Dice. Metric: per-layer IoU at 0.5; map_mIoU = layers 0-3, obj_mIoU = 4-5.
- Cost: preprocessing several hours (GPU, finished 00:55 on 2026-10-16), map targets 39 min (CPU). Cache 162M, DINOv2 features 6.8G.

## E7 — B v2: U-Net on trainval (2026-10-16, commit 1ae7e82)
- Same 1.9M U-Net, 6-ch in / 6-ch out. 39 epochs (early stop), 29.5 min, 1.7 GB VRAM. Best val mIoU 0.262.
- **Result (test):** map 0.316 / obj 0.148 / **mIoU 0.260** (drivable 0.591, vehicle 0.232, pedestrian 0.064).
- **Takeaway:** ~38x more data lifts vehicle IoU from 0.151 (mini v1) to 0.232. Val loss rises from epoch ~12 while IoU still creeps up (overconfident, not worse). Road layout is blobby: lanes and crossings are guessed only where the sparse input shows them.

## E8 — D v2: pixel diffusion on trainval (2026-10-16, commit 1ae7e82)
- E4 denoiser ported to 6 multi-label layers (x0 in {-1,+1}, x0-logit BCE+Dice loss), 80 epochs, 66 min, 2.1 GB. Best val mIoU 0.165.
- **Result (test):** 1 sample mIoU 0.170; 4-sample mean **0.177** (map 0.222 / obj 0.088).
- **Takeaway:** still clearly below B with the same backbone and data — the E4 conclusion holds at 38x the data. Concatenation conditioning in a 2M U-Net is not enough.

## E9 — F: hybrid DiT with camera cross-attention (2026-10-16, commit 1ae7e82)
- DiT-S style (45.6M params, hidden 384, depth 12, 6 heads, 8x8 patches -> 625 BEV tokens, adaLN-Zero). BEV tokens = patchified concat(x_t, sparse). Each block: self-attn -> cross-attn to 672 camera tokens (frozen DINOv2-small 16x28 grid pooled 2x per camera + PETR-style ray MLP (camera centre + ray direction in ego) + camera-id embedding) -> MLP. Cosine schedule, x0-logit BCE+Dice, EMA, grad clip 1, no flip (would break camera geometry). 60 epochs, 92 min, 4.2 GB peak. DINOv2 features precomputed (37 min, fp16).
- **Result (test):** 1 sample mIoU 0.310; 4-sample mean **0.319** (map **0.422** / obj 0.114). drivable 0.690, ped_crossing 0.341, walkway 0.401, divider 0.256, vehicle 0.203, pedestrian 0.025.
- vs B: map +0.106 (ped_crossing 0.155 -> 0.341), objects -0.034. Val mIoU still rising slowly at epoch 60 (0.284 -> 0.285).
- **Qualitative:** DiT draws coherent intersections, crossings and lane dividers that the sparse input barely shows (it attends to the images and has a layout prior); cars are crisp rectangles, but some are misplaced, so per-cell IoU punishes them (same effect as E4).
- **Takeaway:** first diffusion model to beat the U-Net overall. The gain is in map layout, which needs image context + global attention; objects still favour B. Open question: how much comes from the cameras vs the transformer -> ablations use_cameras=False / use_sparse=False.

## E10 — DiT ablations: what does the gain come from? (2026-10-16)
- Same DiT, seed, 60 epochs; one input removed. Configs `configs/dit_no_cams.yaml` (no cross-attention, 77 min) and `configs/dit_no_sparse.yaml` (patch embedding sees only x_t, 92 min); `run_dit_ablations.sh`; `evaluate.py --dit_only`.
- **Result (test, 4-sample mean):**

| Variant | map mIoU | obj mIoU | mIoU |
|---|---|---|---|
| B: U-Net (sparse only) | 0.316 | 0.148 | 0.260 |
| F full (sparse + cameras) | **0.422** | 0.114 | **0.319** |
| F without cameras (sparse only) | 0.331 | 0.113 | 0.258 |
| F without sparse (cameras only) | 0.409 | 0.089 | 0.302 |

- **Takeaway:** the map gain comes from the cameras, not the transformer: without them the DiT is no better than the U-Net (0.258 vs 0.260; ped_crossing 0.169 vs 0.341 with cameras). Cameras alone, without any depth/seg projection, already reach 0.302, so the frozen DINOv2 + ray encoding learns most of the image-to-BEV lifting itself. The sparse projection still adds geometry for objects (obj 0.089 -> 0.114) and a little for the map. Objects remain best with the plain U-Net (0.148): per-cell IoU still favours its blobs over the DiT's sharper but sometimes misplaced boxes.
