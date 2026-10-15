"""Pixel-space conditional diffusion (E4), written out by hand (no diffusers).

Palette / SR3 style: the clean sample x0 is the class map (one-hot scaled to [-1, 1]); the
denoiser sees concat(x_t, sparse BEV) so the condition enters at the very first conv.
The denoiser predicts x0 (as class logits) rather than the noise, see `loss_fn`.
"""
import math
import torch
import torch.nn.functional as F


def cosine_alpha_bar(T, s=0.008):
    """Cosine schedule (Nichol & Dhariwal 2021): alpha_bar(t) = cos^2((t/T + s)/(1 + s) * pi/2),
    divided by its value at t=0 so alpha_bar(0) = 1. alpha_bar_t is the fraction of signal
    variance left at step t: x_t = sqrt(ab) * x0 + sqrt(1 - ab) * noise.
    Returns a tensor [T]; index i is diffusion step i (0 = almost clean, T-1 = almost pure noise).
    Unlike SD's linear schedule it reaches ~0 at the end, so x_{T-1} really is pure noise."""
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    ab = (f / f[0])[1:]                      # drop t=0 -> index i means step i+1 of the formula
    return ab.clamp(min=1e-5).float()        # avoid exact 0 (ab[T-1] would be 0 otherwise)


def to_signed(onehot):
    """{0,1} one-hot -> {-1,+1}: diffusion noise is N(0,1), so data should have a similar scale."""
    return 2 * onehot - 1


def q_sample(x0, t, noise, alpha_bar):
    """Forward process in closed form: jump straight from x0 to noise level t."""
    ab = alpha_bar[t][:, None, None, None]
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise


def make_x0(target, num_classes=3):
    """target [B,H,W] int -> x0 [B,3,H,W] in {-1,+1}."""
    return to_signed(F.one_hot(target, num_classes).permute(0, 3, 1, 2).float())


def denoise_loss(model, sparse, target, alpha_bar, loss_fn):
    """One training step's loss: pick a random noise level per sample, noise the clean map,
    ask the network for x0 and score it with the SAME weighted CE + Dice as baseline B.
    Why predict x0 as logits (not eps): the data are discrete classes, so a classification loss
    on the predicted x0 is the natural fit (x0-prediction with CE is what discrete/segmentation
    diffusion models do), and it keeps the class-imbalance handling (weights, Dice) identical
    to B, so B vs D differs only by the noisy input and the timestep."""
    B = len(target)
    t = torch.randint(0, len(alpha_bar), (B,), device=target.device)
    x_t = q_sample(make_x0(target), t, torch.randn(B, 3, *target.shape[1:], device=target.device), alpha_bar)
    logits = model(torch.cat([x_t, sparse], dim=1), t)
    return loss_fn(logits, target)


@torch.no_grad()
def predict_x0_once(model, sparse, target, t_value, alpha_bar):
    """Probe: noise the true map to level t_value, return the argmax of the one-step x0 prediction."""
    t = torch.full((len(target),), t_value, device=target.device)
    x_t = q_sample(make_x0(target), t, torch.randn(len(target), 3, *target.shape[1:], device=target.device), alpha_bar)
    return model(torch.cat([x_t, sparse], dim=1), t).argmax(1)


@torch.no_grad()
def ddim_sample(model, sparse, alpha_bar, steps=20, generator=None):
    """Deterministic DDIM (eta=0) from pure noise. Returns class probabilities [B,3,H,W].

    At each step: the net gives x0_hat; from x0_hat and x_t we infer the noise
        eps_hat = (x_t - sqrt(ab_t) * x0_hat) / sqrt(1 - ab_t)
    and then re-noise x0_hat to the next (lower) level with that SAME eps_hat:
        x_prev = sqrt(ab_prev) * x0_hat + sqrt(1 - ab_prev) * eps_hat.
    Start: the cosine schedule has ab_{T-1} ~ 0, so x_T is (almost) exactly N(0,1) and we can
    start from pure Gaussian noise. (E3's SD schedule kept ~7% of the signal at t=T, so there
    pure-noise starts were out-of-distribution; that problem does not exist here.)"""
    T = len(alpha_bar)
    B, _, H, W = sparse.shape
    x = torch.randn(B, 3, H, W, device=sparse.device, generator=generator)
    ts = torch.linspace(T - 1, 0, steps).round().long().tolist()      # e.g. 999, 946, ..., 0
    for i, t in enumerate(ts):
        t_batch = torch.full((B,), t, device=sparse.device)
        probs = F.softmax(model(torch.cat([x, sparse], dim=1), t_batch), dim=1)
        if i == len(ts) - 1:
            break                                                      # last step: keep x0_hat only
        x0_hat = to_signed(probs)                                      # = 2*softmax - 1, in [-1, 1]
        ab_t, ab_prev = alpha_bar[t], alpha_bar[ts[i + 1]]
        eps_hat = (x - ab_t.sqrt() * x0_hat) / (1 - ab_t).sqrt()
        x = ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * eps_hat
    return probs


@torch.no_grad()
def predict(model, sparse, alpha_bar, steps=20, num_samples=1, generator=None):
    """Class map [B,H,W]. num_samples > 1: average the softmax probabilities of the samples and
    take the argmax. NOT a hard majority vote: E3 showed that when samples place objects
    differently, a hard vote erases them, while averaged probabilities keep the likely ones."""
    probs = sum(ddim_sample(model, sparse, alpha_bar, steps, generator) for _ in range(num_samples)) / num_samples
    return probs.argmax(1)


@torch.no_grad()
def add_unet_probs(cfg, sparse, device="cuda", bs=16):
    """E5 condition: sparse uint8 [N,4,H,W] numpy -> float32 [N,7,H,W] = concat(sparse, softmax of the
    frozen baseline-B U-Net). The denoiser code above is unchanged: it just receives these 7 channels
    wherever E4 passed the 4-channel `sparse`. Computed once per dataset and kept in memory.
    Caveat: B was trained on the train frames, so its probabilities there are better than on val/test
    (train/test mismatch for stacked models); the refiner learns to trust them more than it should."""
    import os
    from models.unet import build_unet
    unet = build_unet(cfg).to(device)
    unet.load_state_dict(torch.load(os.path.join(cfg["train"]["ckpt_dir"], "best.pt"), map_location=device))
    unet.eval()
    out = []
    for i in range(0, len(sparse), bs):
        x = torch.from_numpy(sparse[i:i + bs]).float().to(device)
        out.append(torch.cat([x, F.softmax(unet(x), dim=1)], dim=1).cpu())
    return torch.cat(out).numpy()
