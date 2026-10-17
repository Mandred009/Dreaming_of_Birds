"""Hybrid DiT (model F): a diffusion transformer over BEV patches that cross-attends to camera tokens.

Written by hand (DiT, Peebles & Xie 2023, plus a cross-attention step). Same job as the pixel U-Net
denoiser: forward(x_t, sparse, t, cams) -> logits of the clean 6 layers [B,6,200,200].

Tokens
  BEV:    concat(x_t [6], sparse [6]) cut into 8x8 patches -> 25x25 = 625 tokens -> Linear -> hidden,
          plus a fixed 2D sin-cos position embedding.
  camera: frozen DINOv2 token (384) -> Linear, + MLP(ray info 6) (PETR-style 3D position: camera
          centre + unit ray direction in the ego frame), + learned camera-id embedding. 6 x 112 tokens.
Block (adaLN-Zero): the timestep embedding gives shift/scale/gate for each of the three sub-layers
  self-attention (BEV<->BEV), cross-attention (BEV queries -> camera keys/values), MLP.
  The modulation Linear is zero-initialised so every block starts as the identity (stable training).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.unet import timestep_embedding


def sincos_2d(grid, dim):
    """Fixed 2D sin-cos position embedding [grid*grid, dim]: half the channels encode the row, half the column."""
    def sincos_1d(d, pos):
        freqs = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float32) / (d // 2))
        a = pos[:, None] * freqs[None]
        return torch.cat([a.sin(), a.cos()], dim=1)
    rows, cols = torch.meshgrid(torch.arange(grid, dtype=torch.float32), torch.arange(grid, dtype=torch.float32), indexing="ij")
    return torch.cat([sincos_1d(dim // 2, rows.reshape(-1)), sincos_1d(dim // 2, cols.reshape(-1))], dim=1)


class Attention(nn.Module):
    """Multi-head attention; queries from x, keys/values from `context` (= x for self-attention)."""
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.q, self.kv, self.out = nn.Linear(dim, dim), nn.Linear(dim, 2 * dim), nn.Linear(dim, dim)

    def forward(self, x, context):
        B, N, D = x.shape
        split = lambda z: z.reshape(B, -1, self.heads, D // self.heads).transpose(1, 2)   # [B,heads,tokens,d]
        k, v = self.kv(context).chunk(2, dim=-1)
        o = F.scaled_dot_product_attention(split(self.q(x)), split(k), split(v))
        return self.out(o.transpose(1, 2).reshape(B, N, D))


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, use_cameras):
        super().__init__()
        self.use_cameras = use_cameras
        ln = lambda: nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)   # scale/shift come from adaLN
        self.n1, self.n3 = ln(), ln()
        self.self_attn = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(approximate="tanh"), nn.Linear(4 * dim, dim))
        if use_cameras:
            self.n2 = ln()
            self.cross_attn = Attention(dim, heads)
        n_mod = 9 if use_cameras else 6   # (shift, scale, gate) per sub-layer
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, n_mod * dim))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, cam_tokens):
        m = self.ada(c).chunk(9 if self.use_cameras else 6, dim=-1)
        h = modulate(self.n1(x), m[0], m[1])
        x = x + m[2][:, None] * self.self_attn(h, h)
        if self.use_cameras:
            h = modulate(self.n2(x), m[3], m[4])
            x = x + m[5][:, None] * self.cross_attn(h, cam_tokens)
        h = modulate(self.n3(x), m[-3], m[-2])
        return x + m[-1][:, None] * self.mlp(h)


class DiT(nn.Module):
    def __init__(self, hidden=384, depth=12, heads=6, patch=8, size=200, in_layers=6, sparse_ch=6,
                 use_sparse=True, use_cameras=True, feat_dim=384, num_cams=6, tokens_per_cam=112):
        super().__init__()
        self.patch, self.grid, self.use_sparse, self.use_cameras, self.layers = patch, size // patch, use_sparse, use_cameras, in_layers
        in_ch = in_layers + (sparse_ch if use_sparse else 0)   # use_sparse=False: sparse is not even an input
        self.patch_embed = nn.Linear(in_ch * patch * patch, hidden)
        self.register_buffer("pos", sincos_2d(self.grid, hidden)[None], persistent=False)
        self.t_embed = nn.Sequential(nn.Linear(256, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        if use_cameras:
            self.cam_feat = nn.Linear(feat_dim, hidden)
            self.cam_ray = nn.Sequential(nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.cam_id = nn.Embedding(num_cams, hidden)
            self.register_buffer("cam_ids", torch.arange(num_cams).repeat_interleave(tokens_per_cam), persistent=False)
        self.blocks = nn.ModuleList([DiTBlock(hidden, heads, use_cameras) for _ in range(depth)])
        # Final layer: adaLN (shift, scale) + Linear to patch*patch*layers logits (zero-initialised below, as in DiT).
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.final_out = nn.Linear(hidden, patch * patch * in_layers)
        self.apply(self._init_linear)
        for blk in self.blocks:   # zero-init AFTER the generic xavier init: every block starts as the identity
            nn.init.zeros_(blk.ada[1].weight)
            nn.init.zeros_(blk.ada[1].bias)
        for m in (self.final_ada[1], self.final_out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _init_linear(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def patchify(self, x):
        """[B,C,200,200] -> [B,625,C*64]: each token is one flattened 8x8 patch of all channels."""
        B, C, H, W = x.shape
        p, g = self.patch, self.grid
        return x.reshape(B, C, g, p, g, p).permute(0, 2, 4, 1, 3, 5).reshape(B, g * g, C * p * p)

    def unpatchify(self, tok):
        """[B,625,L*64] -> [B,L,200,200] (inverse of patchify)."""
        B, p, g, L = len(tok), self.patch, self.grid, self.layers
        return tok.reshape(B, g, g, L, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, L, g * p, g * p)

    def forward(self, x_t, sparse, t, cams=None):
        x = torch.cat([x_t, sparse], dim=1) if self.use_sparse else x_t
        tok = self.patch_embed(self.patchify(x)) + self.pos
        c = self.t_embed(timestep_embedding(t, 256))
        cam_tokens = None
        if self.use_cameras:
            feats, rays = cams   # [B,672,384] fp16/fp32, [B,672,6]
            cam_tokens = self.cam_feat(feats.to(tok.dtype)) + self.cam_ray(rays.to(tok.dtype)) + self.cam_id(self.cam_ids)
        for blk in self.blocks:
            tok = blk(tok, c, cam_tokens)
        shift, scale = self.final_ada(c).chunk(2, dim=-1)
        return self.unpatchify(self.final_out(modulate(self.final_norm(tok), shift, scale)))


def build_dit(cfg):
    d = cfg["dit"]
    model = DiT(d["hidden"], d["depth"], d["heads"], d["patch"], cfg["grid"]["size"], cfg["unet"]["num_classes"],
                cfg["unet"]["in_channels"], d["use_sparse"], d["use_cameras"])
    print(f"DiT parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    return model


ABLATIONS = {"no_cams": "use_cameras", "no_sparse": "use_sparse"}   # name -> the dit flag it switches off


def apply_ablation(cfg, name):
    """Turn off one DiT input and move checkpoints/logs/results to *_<name> so the full model is untouched."""
    cfg["dit"][ABLATIONS[name]] = False
    cfg["dit"]["ckpt_dir"] += f"_{name}"
    cfg["dit"]["log_dir"] += f"_{name}"
    cfg["eval"]["results_dir"] = f"results/ablation_{name}"
