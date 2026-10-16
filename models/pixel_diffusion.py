"""Pixel-space conditional diffusion (E4), written out by hand (no diffusers).

Palette / SR3 style: the clean sample x0 is the 6 binary BEV layers scaled to [-1, 1]; the
denoiser sees concat(x_t, sparse BEV) so the condition enters at the very first conv.
The denoiser predicts x0 (as per-layer logits, 6 independent binary layers), see `denoise_loss`.
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


def make_x0(target):
    """target [B,6,H,W] {0,1} -> x0 in {-1,+1} (the 6 layers are already multi-hot, no one-hot needed)."""
    return to_signed(target)


def denoise_loss(model, sparse, target, alpha_bar, loss_fn, cams=None):
    """One training step's loss: pick a random noise level per sample, noise the clean layers,
    ask the network for x0 logits and score them with the SAME pos_weighted BCE + Dice as baseline B.
    Why predict x0 as logits (not eps): the data are binary masks, so a classification loss on the
    predicted x0 is the natural fit, and it keeps the class-imbalance handling identical to B, so
    B vs D differs only by the noisy input and the timestep.
    `model(x_t, sparse, t, cams)` is the common denoiser interface (U-Net and DiT)."""
    B = len(target)
    t = torch.randint(0, len(alpha_bar), (B,), device=target.device)
    x_t = q_sample(make_x0(target), t, torch.randn_like(target), alpha_bar)
    return loss_fn(model(x_t, sparse, t, cams).float(), target)


@torch.no_grad()
def predict_x0_once(model, sparse, target, t_value, alpha_bar, cams=None):
    """Probe: noise the true layers to level t_value, return the one-step x0 prediction (bool)."""
    t = torch.full((len(target),), t_value, device=target.device)
    x_t = q_sample(make_x0(target), t, torch.randn_like(target), alpha_bar)
    return model(x_t, sparse, t, cams) > 0


@torch.no_grad()
def ddim_sample(model, sparse, alpha_bar, steps=20, generator=None, cams=None):
    """Deterministic DDIM (eta=0) from pure noise. Returns layer probabilities [B,6,H,W].

    At each step: the net gives x0_hat; from x0_hat and x_t we infer the noise
        eps_hat = (x_t - sqrt(ab_t) * x0_hat) / sqrt(1 - ab_t)
    and then re-noise x0_hat to the next (lower) level with that SAME eps_hat:
        x_prev = sqrt(ab_prev) * x0_hat + sqrt(1 - ab_prev) * eps_hat.
    Start: the cosine schedule has ab_{T-1} ~ 0, so x_T is (almost) exactly N(0,1) and we can
    start from pure Gaussian noise."""
    T = len(alpha_bar)
    B, _, H, W = sparse.shape
    x = torch.randn(B, 6, H, W, device=sparse.device, generator=generator)
    ts = torch.linspace(T - 1, 0, steps).round().long().tolist()      # e.g. 999, 946, ..., 0
    for i, t in enumerate(ts):
        t_batch = torch.full((B,), t, device=sparse.device)
        probs = torch.sigmoid(model(x, sparse, t_batch, cams).float())
        if i == len(ts) - 1:
            break                                                      # last step: keep x0_hat only
        x0_hat = to_signed(probs)                                      # = 2*sigmoid - 1, in [-1, 1]
        ab_t, ab_prev = alpha_bar[t], alpha_bar[ts[i + 1]]
        eps_hat = (x - ab_t.sqrt() * x0_hat) / (1 - ab_t).sqrt()
        x = ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * eps_hat
    return probs


@torch.no_grad()
def predict_probs(model, sparse, alpha_bar, steps=20, num_samples=1, generator=None, cams=None):
    """Layer probabilities [B,6,H,W]; num_samples > 1 averages the sigmoid probabilities of the
    samples (NOT a hard vote: in E3 a hard vote erased objects the samples placed differently,
    averaged probabilities keep the likely ones). Threshold at 0.5 for masks."""
    return sum(ddim_sample(model, sparse, alpha_bar, steps, generator, cams) for _ in range(num_samples)) / num_samples
