"""Build the BEV cache: for every nuScenes keyframe save target BEV + sparse BEV.

Both networks (depth ~335M params, seg ~27M) run in fp16 on the GPU at the same time;
that needs ~2 GB VRAM, so a single pass per image is simple and fits easily.
"""
import csv
import os
import numpy as np
import torch
import yaml
from PIL import Image
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import mini_train, mini_val
from transformers import (AutoImageProcessor, AutoModelForDepthEstimation,
                          SegformerImageProcessor, SegformerForSemanticSegmentation)

import bev_geometry as geo


def transform_matrix(translation, rotation):
    """4x4 matrix for a (translation, quaternion) pose: maps child-frame points to parent frame."""
    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T


def get_boxes_in_ego(nusc, sample):
    """Annotation boxes (global frame) expressed in the ego frame of the LIDAR_TOP timestamp."""
    lidar = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    pose = nusc.get("ego_pose", lidar["ego_pose_token"])
    ego_to_global = transform_matrix(pose["translation"], pose["rotation"])
    global_to_ego = np.linalg.inv(ego_to_global)
    out = []
    for tok in sample["anns"]:
        ann = nusc.get("sample_annotation", tok)
        center_g = np.array(ann["translation"] + [1.0])
        center = (global_to_ego @ center_g)[:2]
        # Box yaw in ego frame = box rotation in global composed with inverse ego rotation.
        R = (global_to_ego[:3, :3] @ Quaternion(ann["rotation"]).rotation_matrix)
        yaw = np.arctan2(R[1, 0], R[0, 0])   # heading of the box's x axis in the ground plane
        out.append(dict(center=center, size=(ann["size"][0], ann["size"][1]),  # (width, length)
                        yaw=yaw, category=ann["category_name"],
                        num_pts=ann["num_lidar_pts"] + ann["num_radar_pts"]))
    return out


@torch.no_grad()
def predict_depth(model, processor, image, device):
    """Metric depth in metres, resized back to the image size [H, W] (float32 numpy)."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].half()
    out = model(**inputs)
    h, w = image.size[1], image.size[0]
    depth = processor.post_process_depth_estimation(out, target_sizes=[(h, w)])[0]["predicted_depth"]
    return depth.float().cpu().numpy()


@torch.no_grad()
def predict_seg(model, processor, image, device):
    """Cityscapes class id per pixel [H, W] (int64 numpy)."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].half()
    out = model(**inputs)
    seg = processor.post_process_semantic_segmentation(
        out, target_sizes=[(image.size[1], image.size[0])])[0]
    return seg.cpu().numpy()


def main():
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))
    dataroot = os.path.expanduser(cfg["dataroot"])
    out_dir = os.path.expanduser(cfg["out_dir"])
    os.makedirs(out_dir, exist_ok=True)
    grid, pcfg = cfg["grid"], cfg["points"]
    device = "cuda"

    nusc = NuScenes(version=cfg["version"], dataroot=dataroot, verbose=False)
    depth_proc = AutoImageProcessor.from_pretrained(cfg["depth_model"])
    depth_model = AutoModelForDepthEstimation.from_pretrained(cfg["depth_model"]).half().to(device).eval()
    seg_proc = SegformerImageProcessor.from_pretrained(cfg["seg_model"])
    seg_model = SegformerForSemanticSegmentation.from_pretrained(cfg["seg_model"]).half().to(device).eval()

    train_scenes, val_scenes = set(mini_train), set(mini_val)
    rows = []
    for sample in tqdm(nusc.sample):
        scene_name = nusc.get("scene", sample["scene_token"])["name"]
        split = "train" if scene_name in train_scenes else "val" if scene_name in val_scenes else "other"
        rows.append((sample["token"], scene_name, split))
        path = os.path.join(out_dir, sample["token"] + ".npz")
        if os.path.exists(path):   # resumable
            continue

        target = geo.boxes_to_bev(get_boxes_in_ego(nusc, sample), grid)

        sparse = np.zeros((4, grid["size"], grid["size"]), dtype=np.uint8)
        for cam in cfg["cameras"]:
            sd = nusc.get("sample_data", sample["data"][cam])
            image = Image.open(os.path.join(dataroot, sd["filename"])).convert("RGB")
            calib = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            K = np.array(calib["camera_intrinsic"])
            # cam->ego from calibration. The camera's own ego pose differs slightly from the
            # LIDAR ego pose (different timestamps, a few ms); we ignore that offset.
            cam_to_ego = transform_matrix(calib["translation"], calib["rotation"])

            depth = predict_depth(depth_model, depth_proc, image, device)
            seg = predict_seg(seg_model, seg_proc, image, device)

            pts, v, u = geo.depth_to_ego_points(depth, K, cam_to_ego, pcfg["stride"])
            labels = geo.CITYSCAPES_TO_COARSE[seg[v, u]]
            # Union over the 6 cameras (a cell is occupied if any camera saw it).
            sparse |= geo.points_to_sparse_bev(pts, labels, grid, pcfg["z_range"], pcfg["max_depth"])

        np.savez_compressed(path, target=target, sparse=sparse, scene_name=scene_name)

    with open(os.path.join(out_dir, "index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token", "scene_name", "split"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
