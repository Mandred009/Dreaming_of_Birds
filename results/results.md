mIoU = mean of vehicle and pedestrian IoU, on test = mini_val (81 samples)

| Method | Vehicle IoU | Pedestrian IoU | mIoU |
|---|---|---|---|
| A: projection only | 0.096 | 0.004 | 0.050 |
| A+: projection + 3x3 dilation | 0.097 | 0.004 | 0.051 |
| B: U-Net | 0.151 | 0.041 | 0.096 |
