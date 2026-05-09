import cv2
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from depth_anything_3.api import DepthAnything3
from tqdm import tqdm


BASE_DIR = "/home/vbontal1/Downloads/Occupancy3D"

# Cityscapes to nuScenes Lidar Segmentation class mapping
CITYSCAPES_TO_NUSCENES = {
    0: 11,   # road -> driveable_surface
    1: 13,   # sidewalk -> sidewalk
    2: 15,   # building -> manmade
    3: 15,   # wall -> manmade
    4: 15,   # fence -> manmade
    5: 15,   # pole -> manmade
    6: 15,   # traffic light -> manmade
    7: 15,   # traffic sign -> manmade
    8: 16,   # vegetation -> vegetation
    9: 14,   # terrain -> terrain
    10: 0,   # sky -> void/ignore
    11: 7,   # person -> pedestrian
    12: 4,   # rider -> car (best guess)
    13: 4,   # car -> car
    14: 10,  # truck -> truck
    15: 3,   # bus -> bus
    16: 9,   # train -> trailer (best guess)
    17: 6,   # motorcycle -> motorcycle
    18: 2,   # bicycle -> bicycle
}

# Official nuScenes Lidar Segmentation Color Map (17 classes + 1 free space)
# Classes: 0=void, 1=barrier, 2=bicycle, 3=bus, 4=car, 5=construction, 6=motorcycle, 
#          7=pedestrian, 8=traffic_cone, 9=trailer, 10=truck, 11=driveable_surface,
#          12=other_flat, 13=sidewalk, 14=terrain, 15=manmade, 16=vegetation, 17=free_space
NUSCENES_COLORS = np.array([
    [0, 0, 0],           # 0: void/ignore - black
    [112, 128, 144],     # 1: barrier - slate gray
    [220, 20, 60],       # 2: bicycle - crimson
    [255, 127, 80],      # 3: bus - coral
    [255, 0, 0],         # 4: car - red
    [0, 0, 142],         # 5: construction_vehicle - dark blue
    [255, 165, 0],       # 6: motorcycle - orange
    [220, 20, 60],       # 7: pedestrian - crimson
    [255, 255, 0],       # 8: traffic_cone - yellow
    [0, 100, 0],         # 9: trailer - dark green
    [128, 128, 0],       # 10: truck - olive
    [0, 255, 127],       # 11: driveable_surface - spring green
    [169, 169, 169],     # 12: other_flat - dark gray
    [128, 0, 128],       # 13: sidewalk - purple
    [210, 180, 140],     # 14: terrain - tan
    [105, 105, 105],     # 15: manmade - dim gray
    [0, 128, 128],       # 16: vegetation - teal
    [192, 192, 192],     # 17: free_space - silver
]) / 255.0

# Cityscapes Color Map (for Segmentation) - standard Cityscapes colors
CITYSCAPES_COLORS = np.array([
    [128, 64, 128],    # 0: road
    [244, 35, 232],    # 1: sidewalk
    [70, 70, 70],      # 2: building
    [102, 102, 156],   # 3: wall
    [190, 153, 153],   # 4: fence
    [153, 153, 153],   # 5: pole
    [250, 170, 30],    # 6: traffic light
    [220, 220, 0],     # 7: traffic sign
    [107, 142, 35],    # 8: vegetation
    [152, 251, 152],   # 9: terrain
    [70, 130, 180],    # 10: sky
    [220, 20, 60],     # 11: person
    [255, 0, 0],       # 12: rider
    [0, 0, 142],       # 13: car
    [0, 0, 70],        # 14: truck
    [0, 60, 100],      # 15: bus
    [0, 80, 100],      # 16: train
    [0, 0, 230],       # 17: motorcycle
    [119, 11, 32],     # 18: bicycle
]) / 255.0

def save_occ_as_bev(npz_path, save_path):
    # 1. Load data
    data = np.load(npz_path)
    # The flat array is 640,000 pixels: 200x200x16
    semantics_3d = data['semantics'].reshape(200, 200, 16)
    
    # 2. Flatten Z-axis (original behavior).
    # Class 17 is 'free space'. We set 17 to a high value so np.min picks the actual
    # object classes (0-16), then reset empty columns back to 17.
    temp_occ = np.copy(semantics_3d)
    temp_occ[temp_occ == 17] = 255
    bev_map = np.min(temp_occ, axis=2)
    bev_map[bev_map == 255] = 17  # Reset pure empty columns back to 17
    
    # 3. Map indices to RGB colors
    # Any index above 17 (like noise) gets mapped to free space
    bev_map[bev_map > 17] = 17
    
    # Debug: check what classes are in BEV
    unique_classes = np.unique(bev_map)
    print(f"  BEV classes present: {sorted(unique_classes)}")
    
    bev_rgb = NUSCENES_COLORS[bev_map.astype(int)]
    
    # 4. Convert to uint8 and BGR for saving
    bev_uint8 = (bev_rgb * 255).astype(np.uint8)
    bev_bgr = cv2.cvtColor(bev_uint8, cv2.COLOR_RGB2BGR)
    
    # 5. Save the image with origin='lower' effect (flip vertically)
    bev_bgr_flipped = cv2.flip(bev_bgr, 0)  # Flip vertically for 'lower' origin
    cv2.imwrite(save_path, bev_bgr_flipped)
    print(f"✓ Saved BEV to: {save_path}")

def get_depth_image(img, save_path, model):
    """Generate and save depth map at original image size"""
    try:
        # Store original size
        original_h, original_w = img.shape[:2]
        
        # inference expects a list of images and returns a Prediction object
        prediction = model.inference([img])
        
        # depth is already a numpy array with shape (N, H, W) - get the first (only) image
        depth = prediction.depth[0]
        
        # Resize depth back to original image size
        depth_resized = cv2.resize(depth, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        # Normalize depth to 0-1 for visualization
        depth_normalized = (depth_resized - depth_resized.min()) / (depth_resized.max() - depth_resized.min() + 1e-8)
        # Convert to 0-255 uint8 for proper saving
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        cv2.imwrite(save_path, depth_uint8)
        print(f"✓ Saved depth to: {save_path}")
        return depth_resized
    except Exception as e:
        print(f"✗ Error generating depth for {save_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_segmentation_image(img, save_path, seg_processor, seg_model):
    """Generate segmentation, map to nuScenes IDs, save with nuScenes colors."""
    try:
        original_h, original_w = img.shape[:2]
        
        device = "cuda"
        inputs = seg_processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = seg_model(**inputs)
        seg_map = outputs.logits.argmax(dim=1).squeeze().cpu().numpy()
        
        # Resize segmentation map back to original image size
        seg_map_resized = cv2.resize(seg_map.astype(np.float32), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        seg_map_resized = seg_map_resized.astype(np.int32)
        
        # Clamp to valid Cityscapes class range (0-18)
        seg_map_resized = np.clip(seg_map_resized, 0, 18)

        # Map Cityscapes IDs -> nuScenes occupancy semantic IDs so colors align with BEV.
        free_id = 17
        lut = np.full((19,), free_id, dtype=np.uint8)
        for cs_id, ns_id in CITYSCAPES_TO_NUSCENES.items():
            if 0 <= cs_id < 19:
                lut[cs_id] = np.uint8(ns_id)

        seg_nuscenes = lut[seg_map_resized.astype(int)]

        # Apply nuScenes palette (same as BEV)
        seg_rgb = NUSCENES_COLORS[seg_nuscenes.astype(int)]
        seg_uint8 = (seg_rgb * 255).astype(np.uint8)
        
        # Convert RGB to BGR for cv2.imwrite
        seg_bgr = cv2.cvtColor(seg_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, seg_bgr)
        print(f"✓ Saved segmentation to: {save_path}")
        return seg_nuscenes
    except Exception as e:
        print(f"✗ Error generating segmentation for {save_path}: {e}")
        return None

if __name__ == "__main__":
    annot_file = os.path.join(BASE_DIR, "annotations.json")
    with open(annot_file, 'r') as f:
        annots = json.load(f)

    os.mkdir("bev_outputs") if not os.path.exists("bev_outputs") else None
    os.makedirs("bev_outputs/bev", exist_ok=True) 
    os.makedirs("bev_outputs/depth_CAM_FRONT", exist_ok=True)
    os.makedirs("bev_outputs/depth_CAM_FRONT_LEFT", exist_ok=True)
    os.makedirs("bev_outputs/depth_CAM_FRONT_RIGHT", exist_ok=True)
    os.makedirs("bev_outputs/depth_CAM_BACK", exist_ok=True)
    os.makedirs("bev_outputs/depth_CAM_BACK_LEFT", exist_ok=True)
    os.makedirs("bev_outputs/depth_CAM_BACK_RIGHT", exist_ok=True)
    os.makedirs("bev_outputs/seg_CAM_FRONT", exist_ok=True)
    os.makedirs("bev_outputs/seg_CAM_FRONT_LEFT", exist_ok=True)
    os.makedirs("bev_outputs/seg_CAM_FRONT_RIGHT", exist_ok=True)
    os.makedirs("bev_outputs/seg_CAM_BACK", exist_ok=True)
    os.makedirs("bev_outputs/seg_CAM_BACK_LEFT", exist_ok=True)
    os.makedirs("bev_outputs/seg_CAM_BACK_RIGHT", exist_ok=True)

    # Load models once
    print("Loading depth model...")
    depth_model = DepthAnything3.from_pretrained("depth-anything/DA3-Base").to("cuda")
    
    print("Loading segmentation model...")
    seg_processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to("cuda")
    
    cameras = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    
    for scene_name, scene_info in tqdm(annots['scene_infos'].items(), desc="Processing scenes"):
        print(f"\n{'='*60}")
        print(f"Processing scene: {scene_name}")
        print(f"{'='*60}")
        
        # Iterate through frames in the scene
        for frame_token, frame_data in tqdm(scene_info.items(), desc="Processing frames", leave=False):
            # Save BEV occupancy map
            gt_path = frame_data.get('gt_path')
            if gt_path:
                npz_path = os.path.join(BASE_DIR, gt_path)
                if os.path.exists(npz_path):
                    save_occ_as_bev(npz_path, f"bev_outputs/bev/{scene_name}_{frame_token}.png")
                else:
                    print(f"Warning: NPZ file not found at {npz_path}")
            
            # Process each camera for depth and segmentation
            camera_sensor = frame_data.get('camera_sensor', {})
            
            for camera in tqdm(cameras, desc="Processing cameras", leave=False):
                if camera not in camera_sensor:
                    continue
                
                img_path = os.path.join(BASE_DIR, "imgs", camera_sensor[camera]['img_path'])
                
                if not os.path.exists(img_path):
                    print(f"Warning: Image not found at {img_path}")
                    continue
                
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Error: Could not read image from {img_path}")
                    continue
                
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Generate and save depth map
                depth_output_path = f"bev_outputs/depth_{camera}/{scene_name}_{frame_token}.png"
                get_depth_image(img_rgb, depth_output_path, depth_model)
                
                # Generate and save segmentation map with nuScenes colors
                seg_output_path = f"bev_outputs/seg_{camera}/{scene_name}_{frame_token}.png"
                get_segmentation_image(img_rgb, seg_output_path, seg_processor, seg_model)




