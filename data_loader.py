import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class BEVDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the generated outputs (should contain 'bev', 'depth', 'seg' subfolders).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.bev_dir = os.path.join(root_dir, "bev")
        self.depth_dir = os.path.join(root_dir, "depth")
        self.seg_dir = os.path.join(root_dir, "seg")
        
        # Retrieve all file paths and sort them so they match
        self.bev_files = sorted([f for f in os.listdir(self.bev_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.depth_files = sorted([f for f in os.listdir(self.depth_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.seg_files = sorted([f for f in os.listdir(self.seg_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Ensure we have the same number of files in each folder
        assert len(self.bev_files) == len(self.depth_files) == len(self.seg_files), \
            "Mismatch in the number of files across bev, depth, and seg folders!"
            
    def __len__(self):
        return len(self.bev_files)

    def __getitem__(self, idx):
        bev_path = os.path.join(self.bev_dir, self.bev_files[idx])
        depth_path = os.path.join(self.depth_dir, self.depth_files[idx])
        seg_path = os.path.join(self.seg_dir, self.seg_files[idx])
        
        # Read images using OpenCV and convert BGR to RGB
        bev_img = cv2.cvtColor(cv2.imread(bev_path), cv2.COLOR_BGR2RGB)
        depth_img = cv2.cvtColor(cv2.imread(depth_path), cv2.COLOR_BGR2RGB)
        seg_img = cv2.cvtColor(cv2.imread(seg_path), cv2.COLOR_BGR2RGB)
        
        # Apply transforms if provided
        if self.transform:
            bev_img = self.transform(bev_img)
            depth_img = self.transform(depth_img)
            seg_img = self.transform(seg_img)
            
        return {
            'bev': bev_img,
            'depth': depth_img,
            'seg': seg_img,
            'file_name': self.bev_files[idx] # returning filename for reference
        }

# Example usage:
# define a basic transform to convert numpy arrays to PyTorch tensors
# transform = transforms.ToTensor()

# Initialize the dataset
# dataset = BEVDataset(root_dir="./bev_output", transform=transform)

# Initialize the dataloader
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

# print(f"Created DataLoader with {len(dataset)} paired samples.")

# Fetch a single batch as a quick test (uncomment to test if you have generated data)
# batch = next(iter(dataloader))
# print("BEV batch shape:", batch['bev'].shape)
# print("Depth batch shape:", batch['depth'].shape)
# print("Seg batch shape:", batch['seg'].shape)