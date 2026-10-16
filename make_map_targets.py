"""Add the 6-layer BEV target to every cached npz (CPU only, no NuScenes metadata needed).

target uint8 [6,200,200]: 0 drivable_area, 1 ped_crossing, 2 walkway, 3 divider (road+lane divider),
4 vehicle, 5 pedestrian. Map layers come from the nuScenes map expansion
(<dataroot>/maps/expansion/<location>.json); vehicle/pedestrian are the cached `boxes`.

Usage: python make_map_targets.py
"""
import os
import cv2
import numpy as np
import yaml
from pyquaternion import Quaternion
from tqdm import tqdm
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer

from dataset import dataset_cfg

def mask_for_lines(lines, mask):
    """Devkit's version iterates a MultiLineString directly, which shapely 2 forbids; same code with .geoms."""
    for line in (lines.geoms if lines.geom_type == "MultiLineString" else [lines]):
        coords = np.asarray(list(line.coords), np.int32).reshape((-1, 2))
        cv2.polylines(mask, [coords], False, 1, 2)
    return mask


NuScenesMapExplorer.mask_for_lines = staticmethod(mask_for_lines)

LAYERS = ["drivable_area", "ped_crossing", "walkway", "road_divider", "lane_divider"]


def map_layers(nmap, translation, rotation, size=200, extent=100.0):
    """Map layers [4,size,size] in OUR BEV convention (row 0 = forward +x, col 0 = left +y).

    get_map_mask(patch_box=[x, y, h, w], patch_angle) returns masks whose image axes are
    col = patch "east" (right), row = patch "north", with row 0 = the SOUTH edge (nothing is flipped).
    A patch with angle a has its north axis pointing at global heading 90 + a degrees. We want
    north = ego forward, whose global heading is the ego yaw, so a = yaw_deg - 90.
    Then col increases to the ego's right (ours: col 0 = left, increases to the right: OK) and
    row increases forward (ours: row 0 = forward), so only the rows need flipping.
    """
    yaw = Quaternion(rotation).yaw_pitch_roll[0]   # heading of the ego x axis in the global frame
    masks = nmap.get_map_mask([translation[0], translation[1], extent, extent],
                              np.degrees(yaw) - 90.0, LAYERS, (size, size))
    masks = masks[:, ::-1]                          # row 0 = far forward
    dividers = masks[3] | masks[4]                  # road + lane dividers (devkit draws lines 2 px thick)
    return np.concatenate([masks[:3], dividers[None]]).astype(np.uint8)


def main():
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "config.yaml")))
    dcfg = dataset_cfg(cfg)
    dataroot, out_dir = os.path.expanduser(dcfg["dataroot"]), os.path.expanduser(dcfg["out_dir"])
    maps = {}
    for name in tqdm(sorted(f for f in os.listdir(out_dir) if f.endswith(".npz"))):
        path = os.path.join(out_dir, name)
        try:
            d = dict(np.load(path))
        except Exception:      # file still being written by prepare_data.py
            continue
        if "target" in d:      # resumable
            continue
        loc = str(d["location"])
        if loc not in maps:
            maps[loc] = NuScenesMap(dataroot=dataroot, map_name=loc)
        layers = map_layers(maps[loc], d["ego_translation"], d["ego_rotation"], cfg["grid"]["size"])
        d["target"] = np.concatenate([layers, d["boxes"]])
        np.savez_compressed(path + ".tmp.npz", **d)
        os.replace(path + ".tmp.npz", path)


if __name__ == "__main__":
    main()
