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
