import glob, os
from tqdm import tqdm
import torch
from depth_anything_3.api import DepthAnything3
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class BevGenerator:
  def __init__(self, out_dir, device="cuda"):
    self.out_dir=out_dir
    self.device=device
    self.da3_model=DepthAnything3.from_pretrained("depth-anything/DA3-Base").to(self.device)
    self.seg_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    self.seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to(device)
    self.image_paths = []
    self.image_names = []
    self.image_width=0
    self.image_height=0
    cmap = plt.colormaps.get_cmap("tab20")
    num_labels = self.seg_model.config.num_labels
    self.palette = np.array([cmap(i % 20)[:3] for i in range(num_labels)]) * 255
    
    # Create subdirectories
    self.bev_dir = os.path.join(self.out_dir, "bev")
    self.depth_dir = os.path.join(self.out_dir, "depth")
    self.seg_dir = os.path.join(self.out_dir, "seg")
    os.makedirs(self.bev_dir, exist_ok=True)
    os.makedirs(self.depth_dir, exist_ok=True)
    os.makedirs(self.seg_dir, exist_ok=True)
    
    print("Initialized BevGenerator and output directories")

  def load_imgs(self, img_dir,cnt=-1):
    imgs=sorted(os.listdir(img_dir))
    for i in imgs:
      if cnt !=-1 and len(self.image_paths) >= cnt:
        break
      if i.lower().endswith(('.png', '.jpg', '.jpeg')):
          self.image_paths.append(os.path.join(img_dir, i))
          self.image_names.append(i)

    if len(self.image_paths) > 0:
        img = cv2.imread(self.image_paths[0])
        if img is not None:
            self.image_width=img.shape[1]
            self.image_height=img.shape[0]

  def process_batches(self, batch_size=4, bev_res=800):
    print(f"Processing {len(self.image_paths)} images in batches of {batch_size}...")
    for i in tqdm(range(0, len(self.image_paths), batch_size)):
        batch_paths = self.image_paths[i:i+batch_size]
        batch_names = self.image_names[i:i+batch_size]

        for img_path, img_name in zip(batch_paths, batch_names):
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            with torch.no_grad():
                # 1. Depth & Intrinsics
                out = self.da3_model.inference([img])
                depth = out.depth[0]
                K = out.intrinsics[0]

                # 2. Semantic Segmentation
                inputs = self.seg_processor(images=img, return_tensors="pt").to(self.device)
                outputs = self.seg_model(**inputs)

                h, w = depth.shape
                upsampled_logits = torch.nn.functional.interpolate(
                    outputs.logits,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False
                )
                seg_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

            # Save Depth Map (normalized for visualization)
            depth_min = depth.min()
            depth_max = depth.max()
            if depth_max > depth_min:
                depth_norm = (depth - depth_min) / (depth_max - depth_min)
            else:
                depth_norm = depth
            depth_vis = (depth_norm * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            
            # Save Segmentation Map
            seg_color = self.palette[seg_mask].astype(np.uint8)
            seg_color_bgr = cv2.cvtColor(seg_color, cv2.COLOR_RGB2BGR)

            # 3. Unproject 2D to 3D
            v, u = np.indices((h, w))
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            valid_mask = (depth > 0) & (seg_mask != 10)
            Z = depth[valid_mask]
            X = (u[valid_mask] - cx) * Z / fx

            colors = self.palette[seg_mask[valid_mask]]

            # 4. Generate BEV
            bev_map = np.zeros((bev_res, bev_res, 3), dtype=np.uint8)
            if len(Z) > 0:
                z_min, z_max = 0, np.percentile(Z, 98)
                x_min, x_max = np.percentile(X, 2), np.percentile(X, 98)

                mask = (Z >= z_min) & (Z <= z_max) & (X >= x_min) & (X <= x_max)
                x_valid = X[mask]
                z_valid = Z[mask]
                colors_valid = colors[mask]

                if len(x_valid) > 0:
                    col_indices = ((x_valid - x_min) / (x_max - x_min) * (bev_res - 1)).astype(int)
                    row_indices = ((1 - (z_valid - z_min) / (z_max - z_min)) * (bev_res - 1)).astype(int)
                    bev_map[row_indices, col_indices] = colors_valid

            # 5. Save outputs
            base_name = os.path.splitext(img_name)[0]
            ext = os.path.splitext(img_name)[1]
            
            bev_map_bgr = cv2.cvtColor(bev_map, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.bev_dir, f"bev_{base_name}{ext}"), bev_map_bgr)
            cv2.imwrite(os.path.join(self.depth_dir, f"depth_{base_name}{ext}"), depth_color)
            cv2.imwrite(os.path.join(self.seg_dir, f"seg_{base_name}{ext}"), seg_color_bgr)

        # Clean up GPU memory to avoid crashes
        torch.cuda.empty_cache()

if __name__=="__main__":
    if not os.path.exists("./bev_output"):
      os.mkdir("./bev_output")
    bev_obj=BevGenerator(out_dir="./bev_output")
    # Note: Ensure the path below exists or update it accordingly.
    bev_obj.load_imgs("/home/vbontal1/Downloads/v1.0-trainval01_blobs_camera/samples/CAM_FRONT")
    bev_obj.process_batches(batch_size=4)
