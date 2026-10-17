Per-layer IoU at threshold 0.5 on test (1982 frames); map_mIoU = layers 0-3, obj_mIoU = 4-5, mIoU = all 6.

| Method | drivable | ped_crossing | walkway | divider | vehicle | pedestrian | map_mIoU | obj_mIoU | mIoU |
|---|---|---|---|---|---|---|---|---|---|
| A: projection only | 0.204 | 0.000 | 0.075 | 0.000 | 0.059 | 0.005 | 0.070 | 0.032 | 0.057 |
| B: U-Net | 0.591 | 0.155 | 0.302 | 0.218 | 0.232 | 0.064 | 0.316 | 0.148 | 0.260 |
| D: pixel diffusion (1 sample) | 0.487 | 0.075 | 0.182 | 0.115 | 0.139 | 0.020 | 0.215 | 0.080 | 0.170 |
| D: pixel diffusion (4-sample mean) | 0.528 | 0.063 | 0.188 | 0.109 | 0.161 | 0.015 | 0.222 | 0.088 | 0.177 |
| F: hybrid DiT (1 sample) | 0.672 | 0.334 | 0.377 | 0.250 | 0.189 | 0.035 | 0.408 | 0.112 | 0.310 |
| F: hybrid DiT (4-sample mean) | 0.690 | 0.341 | 0.401 | 0.256 | 0.203 | 0.025 | 0.422 | 0.114 | 0.319 |
