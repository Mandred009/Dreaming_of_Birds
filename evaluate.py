import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from diffusers import AutoencoderKL

# --- 1. Dataset Definition ---
class BEVDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.bev_dir = os.path.join(root_dir, "bev")
        self.depth_dir = os.path.join(root_dir, "depth")
        self.seg_dir = os.path.join(root_dir, "seg")

        self.bev_files = sorted([f for f in os.listdir(self.bev_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.depth_files = sorted([f for f in os.listdir(self.depth_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.seg_files = sorted([f for f in os.listdir(self.seg_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.bev_files)

    def __getitem__(self, idx):
        bev_path = os.path.join(self.bev_dir, self.bev_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        seg_path = os.path.join(self.seg_dir, self.seg_files[idx])

        bev_img = cv2.cvtColor(cv2.imread(bev_path), cv2.COLOR_BGR2RGB)
        depth_img = cv2.cvtColor(cv2.imread(depth_path), cv2.COLOR_BGR2RGB)
        seg_img = cv2.cvtColor(cv2.imread(seg_path), cv2.COLOR_BGR2RGB)

        if self.transform:
            bev_img = self.transform(bev_img)
            depth_img = self.transform(depth_img)
            seg_img = self.transform(seg_img)

        return {'bev': bev_img, 'depth': depth_img, 'seg': seg_img, 'file_name': self.bev_files[idx]}

# --- 2. Model Definitions ---
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class BEVRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")

    def forward(self, x):
        x_scaled = x * 2.0 - 1.0
        posterior = self.vae.encode(x_scaled).latent_dist
        z = posterior.sample()
        out = self.vae.decode(z).sample
        return (out * 0.5 + 0.5).clamp(0, 1)

class InverseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = UNetBlock(3, 64)
        self.pool = nn.MaxPool2d(2)
        self.mid = UNetBlock(64, 128)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_conv = UNetBlock(192, 64)
        self.depth_head = nn.Conv2d(64, 3, 1)
        self.seg_head = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        d = self.down(x)
        m = self.mid(self.pool(d))
        u = self.up(m)
        feats = self.up_conv(torch.cat([d, u], dim=1))
        pred_depth = torch.sigmoid(self.depth_head(feats))
        pred_seg = torch.sigmoid(self.seg_head(feats))
        return pred_depth, pred_seg

# --- 3. Evaluation Routine ---
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if data directory exists
    data_dir = "./bev_output"
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory '{data_dir}' not found.")
        return

    # Load Dataset
    transform = transforms.ToTensor()
    dataset = BEVDataset(root_dir=data_dir, transform=transform)
    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        return
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize Models
    refiner = BEVRefiner().to(device)
    inverse_net = InverseNet().to(device)

    # Check for Checkpoints
    ckpt_dir = "./checkpoints"
    refiner_ckpt = os.path.join(ckpt_dir, "refiner_latest.pth")
    inverse_ckpt = os.path.join(ckpt_dir, "inverse_latest.pth")

    if os.path.exists(refiner_ckpt) and os.path.exists(inverse_ckpt):
        print("Loading checkpoints...")
        refiner.load_state_dict(torch.load(refiner_ckpt, map_location=device))
        inverse_net.load_state_dict(torch.load(inverse_ckpt, map_location=device))
    else:
        print("Error: Checkpoints not found. Please ensure you have trained the models and saved them to './checkpoints'.")
        return

    # Set to Eval Mode
    refiner.eval()
    inverse_net.eval()

    print("Generating evaluation collage...")
    batch = next(iter(dataloader))
    target_size = (256, 256)

    rough_bev = TF.resize(batch['bev'], target_size).to(device)
    target_depth = TF.resize(batch['depth'], target_size).to(device)
    target_seg = TF.resize(batch['seg'], target_size).to(device)

    with torch.no_grad():
        refined_bev = refiner(rough_bev)
        pred_depth, pred_seg = inverse_net(refined_bev)

    def to_np(tensor):
        img = tensor.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        return np.clip(img, 0, 1)

    num_imgs = min(4, rough_bev.size(0))
    fig, axes = plt.subplots(num_imgs, 6, figsize=(20, 3.5 * num_imgs))
    col_titles = ["Rough BEV", "Refined BEV", "Target Depth", "Pred Depth", "Target Seg", "Pred Seg"]

    if num_imgs == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_imgs):
        axes[i, 0].imshow(to_np(rough_bev[i]))
        axes[i, 1].imshow(to_np(refined_bev[i]))
        axes[i, 2].imshow(to_np(target_depth[i]))
        axes[i, 3].imshow(to_np(pred_depth[i]))
        axes[i, 4].imshow(to_np(target_seg[i]))
        axes[i, 5].imshow(to_np(pred_seg[i]))
        
        for j in range(6):
            axes[i, j].axis('off')
            if i == 0:
                axes[i, j].set_title(col_titles[j], fontsize=14)

    plt.tight_layout()
    collage_path = "evaluation_collage.png"
    plt.savefig(collage_path, dpi=300)
    print(f"Saved evaluation collage to '{collage_path}'")
    plt.show()

if __name__ == "__main__":
    evaluate()
