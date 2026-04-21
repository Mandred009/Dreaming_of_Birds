import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm
from diffusers import AutoencoderKL
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
import os
import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import numpy as np

# Assuming data_loader.py exists with BEVDataset
from data_loader import BEVDataset


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Training configuration with performance settings."""
    
    # Data
    root_dir = "./bev_output"
    batch_size = 8                    # Increased for efficiency
    num_workers = 4                   # More workers for data loading
    target_size = (256, 256)
    val_split = 0.1
    pin_memory = True
    prefetch_factor = 2               # Prefetch batches
    persistent_workers = True         # Keep workers alive
    
    # Training
    epochs = 100
    lr = 1e-4
    weight_decay = 1e-5
    grad_clip_norm = 1.0
    
    # Gradient accumulation (effective batch = batch_size * accum_steps)
    gradient_accumulation_steps = 1
    
    # VAE fine-tuning
    finetune_vae = True
    vae_lr_scale = 0.1
    use_gradient_checkpointing = True  # Save memory
    
    # Loss weights
    pixel_match_weight = 2.0
    consistency_weight = 1.0
    depth_weight = 1.0
    seg_weight = 1.0
    smoothness_weight = 0.1
    fill_quality_weight = 0.5
    
    # Loss thresholds
    valid_threshold = 0.02
    empty_threshold = 0.05
    
    # Checkpointing
    ckpt_dir = "./checkpoints"
    save_every = 20
    log_images_every = 10
    
    # Mixed precision & compilation
    use_amp = True
    compile_model = True              # Use torch.compile (PyTorch 2.0+)
    compile_mode = "reduce-overhead"  # Options: default, reduce-overhead, max-autotune
    
    # TensorBoard
    log_dir = "./runs"
    log_every_n_steps = 10            # Log metrics every N steps
    log_histograms = True             # Log weight histograms
    log_gradients = True              # Log gradient statistics
    
    # Performance monitoring
    profile_training = False          # Enable PyTorch profiler
    benchmark_cudnn = True            # Enable cudnn benchmark
    
    # Early stopping
    early_stopping_patience = 20
    min_improvement = 1e-4
    
    # Learning rate scheduling
    scheduler_type = "cosine"         # Options: cosine, step, plateau
    warmup_epochs = 5


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_device(config):
    """Setup device with optimizations."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        
        # Enable cudnn benchmark for faster training
        if config.benchmark_cudnn:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Enable TF32 for Ampere GPUs (faster matrix ops)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Benchmark: {config.benchmark_cudnn}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
        # Use all CPU cores
        torch.set_num_threads(os.cpu_count())
        print(f"CPU Threads: {torch.get_num_threads()}")
    
    return device


def count_parameters(model, trainable_only=True):
    """Count model parameters."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_time(seconds):
    """Format seconds to human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        return f"{seconds // 3600:.0f}h {(seconds % 3600) // 60:.0f}m"


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        self.values = []
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.values.append(val)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        self.sum += val * n
        self.count += n
    
    @property
    def avg(self):
        return self.sum / max(self.count, 1)
    
    @property
    def moving_avg(self):
        if not self.values:
            return 0
        return sum(self.values) / len(self.values)


class MetricTracker:
    """Track multiple metrics efficiently."""
    
    def __init__(self):
        self.metrics = defaultdict(AverageMeter)
    
    def update(self, metrics_dict, n=1):
        for k, v in metrics_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.metrics[k].update(v, n)
    
    def get_averages(self):
        return {k: v.avg for k, v in self.metrics.items()}
    
    def get_moving_averages(self):
        return {k: v.moving_avg for k, v in self.metrics.items()}
    
    def reset(self):
        for meter in self.metrics.values():
            meter.reset()


# =============================================================================
# TENSORBOARD LOGGER
# =============================================================================

class TensorBoardLogger:
    """Comprehensive TensorBoard logging."""
    
    def __init__(self, log_dir, config):
        self.config = config
        
        # Create unique run directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(log_dir) / f"bev_refiner_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.run_dir))
        
        # Save config
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(vars(config), f, indent=2, default=str)
        
        print(f"TensorBoard logs: {self.run_dir}")
        print(f"Run: tensorboard --logdir={log_dir}")
    
    def log_scalars(self, tag_prefix, metrics, step):
        """Log multiple scalars with a prefix."""
        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(f"{tag_prefix}/{name}", value, step)
    
    def log_scalar(self, tag, value, step):
        """Log a single scalar."""
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.writer.add_scalar(tag, value, step)
    
    def log_learning_rate(self, optimizer, step):
        """Log learning rates for all parameter groups."""
        for i, param_group in enumerate(optimizer.param_groups):
            name = param_group.get('name', f'group_{i}')
            self.writer.add_scalar(f"lr/{name}", param_group['lr'], step)
    
    def log_images(self, tag, images, step, nrow=4):
        """Log images as a grid."""
        from torchvision.utils import make_grid
        
        if images.dim() == 3:
            images = images.unsqueeze(0)
        
        # Clamp values to [0, 1]
        images = images.clamp(0, 1)
        
        grid = make_grid(images[:8], nrow=nrow, normalize=False)
        self.writer.add_image(tag, grid, step)
    
    def log_bev_comparison(self, sparse_bev, refined_bev, step):
        """Log BEV comparison images."""
        from torchvision.utils import make_grid
        
        batch_size = min(4, sparse_bev.size(0))
        
        # Create comparison: sparse | refined | difference
        sparse = sparse_bev[:batch_size].clamp(0, 1)
        refined = refined_bev[:batch_size].clamp(0, 1)
        diff = torch.abs(refined - sparse)
        
        # Concatenate horizontally for each sample
        comparison = torch.cat([sparse, refined, diff], dim=3)  # Concat along width
        
        grid = make_grid(comparison, nrow=1, normalize=False)
        self.writer.add_image("comparison/sparse_refined_diff", grid, step)
    
    def log_prediction_comparison(self, pred_depth, target_depth, 
                                   pred_seg, target_seg, step):
        """Log prediction vs target comparisons."""
        from torchvision.utils import make_grid
        
        batch_size = min(4, pred_depth.size(0))
        
        # Depth comparison
        depth_comp = torch.cat([
            target_depth[:batch_size].clamp(0, 1),
            pred_depth[:batch_size].clamp(0, 1)
        ], dim=3)
        depth_grid = make_grid(depth_comp, nrow=1, normalize=False)
        self.writer.add_image("comparison/depth_target_pred", depth_grid, step)
        
        # Segmentation comparison
        seg_comp = torch.cat([
            target_seg[:batch_size].clamp(0, 1),
            pred_seg[:batch_size].clamp(0, 1)
        ], dim=3)
        seg_grid = make_grid(seg_comp, nrow=1, normalize=False)
        self.writer.add_image("comparison/seg_target_pred", seg_grid, step)
    
    def log_histograms(self, model, step, prefix="model"):
        """Log weight histograms."""
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.writer.add_histogram(
                    f"{prefix}/weights/{name}", 
                    param.data.cpu(), 
                    step
                )
    
    def log_gradients(self, model, step, prefix="model"):
        """Log gradient statistics."""
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_norm += grad_norm ** 2
                self.writer.add_scalar(
                    f"{prefix}/grad_norm/{name}", 
                    grad_norm, 
                    step
                )
        
        total_norm = total_norm ** 0.5
        self.writer.add_scalar(f"{prefix}/grad_norm_total", total_norm, step)
        
        return total_norm
    
    def log_density_stats(self, sparse_bev, refined_bev, threshold, step):
        """Log BEV density statistics."""
        sparse_density = (sparse_bev.mean(1) > threshold).float().mean()
        refined_density = (refined_bev.mean(1) > threshold).float().mean()
        fill_ratio = refined_density / (sparse_density + 1e-8)
        
        self.writer.add_scalar("density/sparse", sparse_density.item(), step)
        self.writer.add_scalar("density/refined", refined_density.item(), step)
        self.writer.add_scalar("density/fill_ratio", fill_ratio.item(), step)
    
    def log_memory_stats(self, step):
        """Log GPU memory statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            
            self.writer.add_scalar("memory/allocated_gb", allocated, step)
            self.writer.add_scalar("memory/reserved_gb", reserved, step)
            self.writer.add_scalar("memory/max_allocated_gb", max_allocated, step)
    
    def log_throughput(self, samples_per_sec, step):
        """Log training throughput."""
        self.writer.add_scalar("performance/samples_per_sec", samples_per_sec, step)
    
    def log_epoch_summary(self, epoch, train_metrics, val_metrics, best_val_loss):
        """Log epoch summary as text."""
        text = f"""
        ## Epoch {epoch + 1} Summary
        
        ### Training Metrics
        | Metric | Value |
        |--------|-------|
        {''.join(f"| {k} | {v:.4f} |" + chr(10) for k, v in train_metrics.items())}
        
        ### Validation Metrics
        | Metric | Value |
        |--------|-------|
        {''.join(f"| {k} | {v:.4f} |" + chr(10) for k, v in val_metrics.items())}
        
        **Best Validation Loss**: {best_val_loss:.4f}
        """
        self.writer.add_text("epoch_summary", text, epoch)
    
    def close(self):
        """Close the writer."""
        self.writer.close()


# =============================================================================
# EFFICIENT DATA LOADING
# =============================================================================

class FastBEVDataset(Dataset):
    """
    Optimized BEV dataset with caching and efficient loading.
    Wraps the original BEVDataset with additional optimizations.
    """
    
    def __init__(self, base_dataset, target_size, cache_size=100):
        self.base_dataset = base_dataset
        self.target_size = target_size
        self.cache_size = cache_size
        self._cache = {}
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Check cache
        if idx in self._cache:
            return self._cache[idx]
        
        # Load from base dataset
        sample = self.base_dataset[idx]
        
        # Pre-resize to target size (faster than doing it in training loop)
        processed = {
            'bev': TF.resize(sample['bev'], self.target_size),
            'depth': TF.resize(sample['depth'], self.target_size),
            'seg': TF.resize(sample['seg'], self.target_size)
        }
        
        # Cache if space available
        if len(self._cache) < self.cache_size:
            self._cache[idx] = processed
        
        return processed


def create_dataloaders(config):
    """Create optimized dataloaders."""
    print("\nSetting up data loaders...")
    
    transform = transforms.ToTensor()
    base_dataset = BEVDataset(root_dir=config.root_dir, transform=transform)
    
    # Wrap with fast dataset
    full_dataset = FastBEVDataset(
        base_dataset, 
        config.target_size,
        cache_size=min(100, len(base_dataset))
    )
    
    # Split into train and validation
    val_size = int(len(full_dataset) * config.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Common dataloader kwargs
    loader_kwargs = {
        'pin_memory': config.pin_memory and torch.cuda.is_available(),
        'num_workers': config.num_workers,
        'persistent_workers': config.persistent_workers and config.num_workers > 0,
    }
    
    if config.num_workers > 0:
        loader_kwargs['prefetch_factor'] = config.prefetch_factor
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        **loader_kwargs
    )
    
    print(f"Train samples: {len(train_dataset)}, batches: {len(train_loader)}")
    print(f"Val samples: {len(val_dataset)}, batches: {len(val_loader)}")
    print(f"Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
    
    return train_loader, val_loader


# =============================================================================
# MODEL COMPONENTS (OPTIMIZED)
# =============================================================================

class UNetBlock(nn.Module):
    """Optimized U-Net convolutional block."""
    
    def __init__(self, in_ch, out_ch, use_residual=False):
        super().__init__()
        self.use_residual = use_residual
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
        if use_residual:
            self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        else:
            self.skip = None
    
    def forward(self, x):
        out = self.conv(x)
        if self.use_residual and self.skip is not None:
            out = out + self.skip(x)
        return out


class BEVRefiner(nn.Module):
    """Optimized BEV Refiner with gradient checkpointing support."""
    
    def __init__(self, finetune=True, use_gradient_checkpointing=False):
        super().__init__()
        print("Loading pretrained Stable Diffusion VAE...")
        
        self.vae = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae"
        )
        
        self.finetune = finetune
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        if not finetune:
            self._freeze_vae()
        else:
            self._partial_freeze()
        
        if use_gradient_checkpointing and finetune:
            self._enable_gradient_checkpointing()
        
        print(f"VAE loaded. Finetune: {finetune}, Gradient checkpointing: {use_gradient_checkpointing}")
    
    def _freeze_vae(self):
        for param in self.vae.parameters():
            param.requires_grad = False
    
    def _partial_freeze(self):
        for param in self.vae.encoder.parameters():
            param.requires_grad = False
        for param in self.vae.quant_conv.parameters():
            param.requires_grad = False
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.vae.decoder, 'gradient_checkpointing_enable'):
            self.vae.decoder.gradient_checkpointing_enable()
    
    def get_param_groups(self, base_lr, lr_scale=0.1):
        if not self.finetune:
            return []
        
        decoder_params = [p for p in self.vae.decoder.parameters() if p.requires_grad]
        post_quant_params = [p for p in self.vae.post_quant_conv.parameters() if p.requires_grad]
        
        return [
            {'params': decoder_params, 'lr': base_lr * lr_scale, 'name': 'vae_decoder'},
            {'params': post_quant_params, 'lr': base_lr * lr_scale, 'name': 'vae_post_quant'}
        ]
    
    def forward(self, x):
        x_scaled = x * 2.0 - 1.0
        
        with torch.set_grad_enabled(self.finetune):
            posterior = self.vae.encode(x_scaled).latent_dist
            z = posterior.sample()
        
        out = self.vae.decode(z).sample
        return (out * 0.5 + 0.5).clamp(0, 1)
    
    @torch.no_grad()
    def encode_only(self, x):
        x_scaled = x * 2.0 - 1.0
        posterior = self.vae.encode(x_scaled).latent_dist
        return posterior.mean


class InverseNet(nn.Module):
    """Optimized Inverse Network with optional gradient checkpointing."""
    
    def __init__(self, base_ch=64, use_gradient_checkpointing=False):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Encoder
        self.down1 = UNetBlock(3, base_ch)
        self.down2 = UNetBlock(base_ch, base_ch * 2, use_residual=True)
        self.down3 = UNetBlock(base_ch * 2, base_ch * 4, use_residual=True)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.mid = UNetBlock(base_ch * 4, base_ch * 8, use_residual=True)
        
        # Decoder
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv1 = UNetBlock(base_ch * 8 + base_ch * 4, base_ch * 4)
        self.up_conv2 = UNetBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.up_conv3 = UNetBlock(base_ch * 2 + base_ch, base_ch)
        
        # Output heads
        self.depth_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, 3, 1),
            nn.Sigmoid()
        )
        
        self.seg_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, 3, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _forward_impl(self, x):
        """Core forward implementation."""
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        
        # Bottleneck
        m = self.mid(self.pool(d3))
        
        # Decoder
        u1 = self.up(m)
        u1 = self.up_conv1(torch.cat([d3, u1], dim=1))
        
        u2 = self.up(u1)
        u2 = self.up_conv2(torch.cat([d2, u2], dim=1))
        
        u3 = self.up(u2)
        feats = self.up_conv3(torch.cat([d1, u3], dim=1))
        
        return self.depth_head(feats), self.seg_head(feats)
    
    def forward(self, x):
        if self.use_gradient_checkpointing and self.training:
            return gradient_checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


# =============================================================================
# OPTIMIZED LOSS FUNCTION
# =============================================================================

class SparseToDenseLoss(nn.Module):
    """Optimized loss function with efficient tensor operations."""
    
    def __init__(self, config):
        super().__init__()
        self.pixel_match_weight = config.pixel_match_weight
        self.depth_weight = config.depth_weight
        self.seg_weight = config.seg_weight
        self.smoothness_weight = config.smoothness_weight
        self.fill_quality_weight = config.fill_quality_weight
        self.valid_threshold = config.valid_threshold
    
    @torch.no_grad()
    def get_valid_mask(self, sparse_bev):
        """Create mask of valid pixels (cached computation)."""
        return (sparse_bev.amax(dim=1, keepdim=True) > self.valid_threshold).float()
    
    def forward(self, pred_depth, pred_seg, target_depth, target_seg,
                refined_bev, sparse_bev):
        """Compute all losses efficiently."""
        
        # Pre-compute valid mask (no gradient needed)
        valid_mask = self.get_valid_mask(sparse_bev)
        fill_mask = 1.0 - valid_mask
        
        # 1. Pixel Matching Loss (preserve known pixels)
        pixel_diff = (refined_bev - sparse_bev).abs()
        pixel_match_loss = (pixel_diff * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        
        # 2. Fill Smoothness Loss (smooth filling in empty regions)
        grad_h = (refined_bev[:, :, 1:, :] - refined_bev[:, :, :-1, :]).abs()
        grad_w = (refined_bev[:, :, :, 1:] - refined_bev[:, :, :, :-1]).abs()
        
        fill_h = fill_mask[:, :, 1:, :] * fill_mask[:, :, :-1, :]
        fill_w = fill_mask[:, :, :, 1:] * fill_mask[:, :, :, :-1]
        
        smooth_loss = (
            (grad_h * fill_h).sum() / (fill_h.sum() + 1e-8) +
            (grad_w * fill_w).sum() / (fill_w.sum() + 1e-8)
        )
        
        # 3. Fill Quality Loss (prevent degenerate filling)
        filled_pixels = refined_bev * fill_mask
        fill_count = fill_mask.sum() * 3 + 1e-8
        
        if fill_mask.sum() > 0:
            filled_mean = filled_pixels.sum() / fill_count
            filled_var = ((filled_pixels - filled_mean * fill_mask) ** 2 * fill_mask).sum() / fill_count
            fill_quality_loss = torch.exp(-filled_var * 10) + (filled_mean - 0.5) ** 2
        else:
            fill_quality_loss = torch.tensor(0.0, device=refined_bev.device)
        
        # 4. Reconstruction Losses
        depth_loss = F.l1_loss(pred_depth, target_depth)
        seg_loss = F.l1_loss(pred_seg, target_seg)
        
        # Total Loss
        total_loss = (
            self.pixel_match_weight * pixel_match_loss +
            self.smoothness_weight * smooth_loss +
            self.fill_quality_weight * fill_quality_loss +
            self.depth_weight * depth_loss +
            self.seg_weight * seg_loss
        )
        
        # Compute stats for logging (no gradient)
        with torch.no_grad():
            valid_ratio = valid_mask.mean()
            refined_density = (refined_bev.mean(1, keepdim=True) > self.valid_threshold).float().mean()
        
        loss_dict = {
            'total': total_loss.item(),
            'pixel_match': pixel_match_loss.item(),
            'smoothness': smooth_loss.item(),
            'fill_quality': fill_quality_loss.item() if isinstance(fill_quality_loss, torch.Tensor) else fill_quality_loss,
            'depth': depth_loss.item(),
            'seg': seg_loss.item(),
            'valid_ratio': valid_ratio.item(),
            'refined_density': refined_density.item()
        }
        
        return total_loss, loss_dict


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

def create_scheduler(optimizer, config, steps_per_epoch):
    """Create learning rate scheduler with warmup."""
    
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch
    
    if config.scheduler_type == "cosine":
        # Cosine annealing with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    elif config.scheduler_type == "step":
        # Step decay with warmup
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            epoch = step // steps_per_epoch
            return 0.1 ** (epoch // 30)
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    elif config.scheduler_type == "plateau":
        # Reduce on plateau (step-based, after warmup)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")
    
    return scheduler


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """Optimized training manager with TensorBoard logging."""
    
    def __init__(self, config):
        self.config = config
        self.device = setup_device(config)
        
        # Setup components
        self.train_loader, self.val_loader = create_dataloaders(config)
        self._setup_models()
        self._setup_training()
        self._setup_logging()
        
        # Training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Load checkpoint if exists
        self._load_checkpoint()
    
    def _setup_models(self):
        """Initialize and optionally compile models."""
        print("\nSetting up models...")
        
        self.refiner = BEVRefiner(
            finetune=self.config.finetune_vae,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing
        ).to(self.device)
        
        self.inverse_net = InverseNet(
            base_ch=64,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing
        ).to(self.device)
        
        # Compile models for faster training (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            print(f"Compiling models with mode: {self.config.compile_mode}")
            try:
                self.inverse_net = torch.compile(
                    self.inverse_net, 
                    mode=self.config.compile_mode
                )
                print("InverseNet compiled successfully")
            except Exception as e:
                print(f"Failed to compile InverseNet: {e}")
        
        print(f"InverseNet parameters: {count_parameters(self.inverse_net):,}")
        print(f"Refiner trainable parameters: {count_parameters(self.refiner):,}")
    
    def _setup_training(self):
        """Setup optimizer, scheduler, loss, and scaler."""
        print("\nSetting up training...")
        
        # Collect parameters
        param_groups = [
            {'params': self.inverse_net.parameters(), 'lr': self.config.lr, 'name': 'inverse_net'}
        ]
        param_groups.extend(
            self.refiner.get_param_groups(self.config.lr, self.config.vae_lr_scale)
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            param_groups,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            fused=torch.cuda.is_available()  # Fused optimizer for faster updates
        )
        
        # Scheduler
        self.scheduler = create_scheduler(
            self.optimizer, 
            self.config, 
            len(self.train_loader)
        )
        
        # Loss
        self.criterion = SparseToDenseLoss(self.config)
        
        # Mixed precision
        self.scaler = GradScaler() if self.config.use_amp else None
        
        # Create directories
        os.makedirs(self.config.ckpt_dir, exist_ok=True)
    
    def _setup_logging(self):
        """Setup TensorBoard logger."""
        self.logger = TensorBoardLogger(self.config.log_dir, self.config)
        self.train_metrics = MetricTracker()
        self.val_metrics = MetricTracker()
    
    def _load_checkpoint(self):
        """Load checkpoint if exists."""
        ckpt_path = os.path.join(self.config.ckpt_dir, "latest_checkpoint.pth")
        
        if os.path.exists(ckpt_path):
            print(f"\nLoading checkpoint from {ckpt_path}...")
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            
            # Handle compiled model state dict
            refiner_state = checkpoint['refiner']
            inverse_net_state = checkpoint['inverse_net']
            
            # Remove _orig_mod prefix if present (from compiled models)
            def clean_state_dict(state_dict):
                new_state_dict = {}
                for k, v in state_dict.items():
                    new_key = k.replace('_orig_mod.', '')
                    new_state_dict[new_key] = v
                return new_state_dict
            
            self.refiner.load_state_dict(clean_state_dict(refiner_state))
            
            # For compiled model, need to access underlying module
            if hasattr(self.inverse_net, '_orig_mod'):
                self.inverse_net._orig_mod.load_state_dict(clean_state_dict(inverse_net_state))
            else:
                self.inverse_net.load_state_dict(clean_state_dict(inverse_net_state))
            
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint.get('global_step', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            if self.scaler and 'scaler' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler'])
            
            print(f"Resumed from epoch {self.start_epoch}, step {self.global_step}")
        else:
            print("\nNo checkpoint found. Starting from scratch.")
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint."""
        # Get state dict (handle compiled models)
        if hasattr(self.inverse_net, '_orig_mod'):
            inverse_net_state = self.inverse_net._orig_mod.state_dict()
        else:
            inverse_net_state = self.inverse_net.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'refiner': self.refiner.state_dict(),
            'inverse_net': inverse_net_state,
            'optimizer': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config)
        }
        
        if self.scaler:
            checkpoint['scaler'] = self.scaler.state_dict()
        
        # Save latest
        torch.save(checkpoint, os.path.join(self.config.ckpt_dir, "latest_checkpoint.pth"))
        
        # Save periodic
        if (epoch + 1) % self.config.save_every == 0:
            torch.save(checkpoint, os.path.join(self.config.ckpt_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        
        # Save best
        if is_best:
            torch.save(checkpoint, os.path.join(self.config.ckpt_dir, "best_checkpoint.pth"))
    
    def train_epoch(self, epoch):
        """Run one training epoch with optimizations."""
        self.refiner.train()
        self.inverse_net.train()
        self.train_metrics.reset()
        
        num_batches = len(self.train_loader)
        accum_steps = self.config.gradient_accumulation_steps
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.epochs}",
            leave=True,
            dynamic_ncols=True
        )
        
        batch_start_time = time.time()
        samples_processed = 0
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device (already resized by dataset)
            sparse_bev = batch['bev'].to(self.device, non_blocking=True)
            target_depth = batch['depth'].to(self.device, non_blocking=True)
            target_seg = batch['seg'].to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=self.config.use_amp):
                refined_bev = self.refiner(sparse_bev)
                pred_depth, pred_seg = self.inverse_net(refined_bev)
                loss, loss_dict = self.criterion(
                    pred_depth, pred_seg,
                    target_depth, target_seg,
                    refined_bev, sparse_bev
                )
                
                # Scale loss for gradient accumulation
                loss = loss / accum_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every accum_steps
            if (batch_idx + 1) % accum_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(self.inverse_net.parameters()) + 
                        [p for p in self.refiner.parameters() if p.requires_grad],
                        self.config.grad_clip_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(self.inverse_net.parameters()) + 
                        [p for p in self.refiner.parameters() if p.requires_grad],
                        self.config.grad_clip_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Step scheduler (if step-based)
                if self.config.scheduler_type != "plateau":
                    self.scheduler.step()
                
                self.global_step += 1
                
                # Log to TensorBoard
                if self.global_step % self.config.log_every_n_steps == 0:
                    self.logger.log_scalars("train", loss_dict, self.global_step)
                    self.logger.log_learning_rate(self.optimizer, self.global_step)
                    self.logger.log_scalar("train/grad_norm", grad_norm.item(), self.global_step)
                    
                    # Log density stats
                    self.logger.log_density_stats(
                        sparse_bev, refined_bev, 
                        self.config.valid_threshold, 
                        self.global_step
                    )
                    
                    # Log memory usage
                    self.logger.log_memory_stats(self.global_step)
            
            # Update metrics
            self.train_metrics.update(loss_dict)
            samples_processed += sparse_bev.size(0)
            
            # Calculate throughput
            elapsed = time.time() - batch_start_time
            throughput = samples_processed / elapsed
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'pix': f"{loss_dict['pixel_match']:.3f}",
                'den': f"{loss_dict['refined_density']:.1%}",
                'spd': f"{throughput:.1f}s/s"
            })
        
        # Log throughput
        total_time = time.time() - batch_start_time
        avg_throughput = samples_processed / total_time
        self.logger.log_throughput(avg_throughput, self.global_step)
        
        return self.train_metrics.get_averages()
    
    @torch.no_grad()
    def validate(self, epoch):
        """Run validation."""
        self.refiner.eval()
        self.inverse_net.eval()
        self.val_metrics.reset()
        
        vis_batch = None
        
        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
            sparse_bev = batch['bev'].to(self.device, non_blocking=True)
            target_depth = batch['depth'].to(self.device, non_blocking=True)
            target_seg = batch['seg'].to(self.device, non_blocking=True)
            
            with autocast('cuda', enabled=self.config.use_amp):
                refined_bev = self.refiner(sparse_bev)
                pred_depth, pred_seg = self.inverse_net(refined_bev)
                _, loss_dict = self.criterion(
                    pred_depth, pred_seg,
                    target_depth, target_seg,
                    refined_bev, sparse_bev
                )
            
            self.val_metrics.update(loss_dict)
            
            # Save first batch for visualization
            if batch_idx == 0:
                vis_batch = {
                    'sparse_bev': sparse_bev.cpu(),
                    'refined_bev': refined_bev.cpu(),
                    'pred_depth': pred_depth.cpu(),
                    'pred_seg': pred_seg.cpu(),
                    'target_depth': target_depth.cpu(),
                    'target_seg': target_seg.cpu()
                }
        
        avg_metrics = self.val_metrics.get_averages()
        
        # Log validation metrics
        self.logger.log_scalars("val", avg_metrics, self.global_step)
        
        # Log images periodically
        if (epoch + 1) % self.config.log_images_every == 0 and vis_batch is not None:
            self.logger.log_bev_comparison(
                vis_batch['sparse_bev'],
                vis_batch['refined_bev'],
                self.global_step
            )
            self.logger.log_prediction_comparison(
                vis_batch['pred_depth'],
                vis_batch['target_depth'],
                vis_batch['pred_seg'],
                vis_batch['target_seg'],
                self.global_step
            )
        
        return avg_metrics
    
    def train(self):
        """Full training loop."""
        print(f"\n{'='*70}")
        print(f"Starting training for {self.config.epochs} epochs")
        print(f"Logging to: {self.logger.run_dir}")
        print(f"{'='*70}\n")
        
        training_start = time.time()
        
        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                epoch_start = time.time()
                
                # Training
                train_losses = self.train_epoch(epoch)
                
                # Validation
                val_losses = self.validate(epoch)
                
                # Update plateau scheduler if used
                if self.config.scheduler_type == "plateau":
                    self.scheduler.step(val_losses['total'])
                
                epoch_time = time.time() - epoch_start
                
                # Check for improvement
                is_best = val_losses['total'] < self.best_val_loss - self.config.min_improvement
                if is_best:
                    self.best_val_loss = val_losses['total']
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Save checkpoint
                self._save_checkpoint(epoch, is_best)
                
                # Log epoch summary
                self.logger.log_epoch_summary(
                    epoch, train_losses, val_losses, self.best_val_loss
                )
                
                # Print summary
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"\nEpoch [{epoch+1}/{self.config.epochs}] - {format_time(epoch_time)}")
                print(f"  Train: loss={train_losses['total']:.4f}, "
                      f"depth={train_losses['depth']:.4f}, "
                      f"density={train_losses['refined_density']:.1%}")
                print(f"  Val:   loss={val_losses['total']:.4f}, "
                      f"depth={val_losses['depth']:.4f}, "
                      f"density={val_losses['refined_density']:.1%}")
                print(f"  LR: {current_lr:.2e}, Best: {self.best_val_loss:.4f}"
                      f"{' *' if is_best else ''}")
                
                # Early stopping check
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {self.patience_counter} epochs without improvement")
                    break
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
        
        finally:
            # Cleanup
            total_time = time.time() - training_start
            print(f"\n{'='*70}")
            print(f"Training complete!")
            print(f"Total time: {format_time(total_time)}")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Checkpoints: {self.config.ckpt_dir}")
            print(f"TensorBoard: {self.logger.run_dir}")
            print(f"{'='*70}\n")
            
            self.logger.close()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='BEV Refinement Training')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--root_dir', type=str, default=None)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--no_compile', action='store_true', help='Disable model compilation')
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision')
    parser.add_argument('--accum_steps', type=int, default=None, help='Gradient accumulation steps')
    parser.add_argument('--workers', type=int, default=None, help='Number of data workers')
    args = parser.parse_args()
    
    # Create config
    config = Config()
    
    # Apply command line overrides
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.root_dir is not None:
        config.root_dir = args.root_dir
    if args.ckpt_dir is not None:
        config.ckpt_dir = args.ckpt_dir
    if args.log_dir is not None:
        config.log_dir = args.log_dir
    if args.no_compile:
        config.compile_model = False
    if args.no_amp:
        config.use_amp = False
    if args.accum_steps is not None:
        config.gradient_accumulation_steps = args.accum_steps
    if args.workers is not None:
        config.num_workers = args.workers
    
    # Print configuration
    print("\n" + "="*70)
    print("Configuration:")
    print("="*70)
    for key, value in sorted(vars(config).items()):
        if not key.startswith('_'):
            print(f"  {key}: {value}")
    print("="*70 + "\n")
    
    # Train
    trainer = Trainer(config)
    trainer.train()