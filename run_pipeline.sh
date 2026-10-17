#!/usr/bin/env bash
# Full pipeline, one step at a time (RAM is tight, never two jobs at once). Each step logs to logs/<name>.log
# and the console shows free RAM/disk before it starts.
# Usage: nohup ./run_pipeline.sh > logs/pipeline.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

step() {   # step <name> <command...>
  local name=$1; shift
  echo "$(date) START $name | $(free -m | awk '/Mem:/{print "RAM avail " $7 " MB"}') | $(df -h ~ | awk 'END{print "disk free " $4}')"
  "$@" > "logs/$name.log" 2>&1
  echo "$(date) DONE  $name"
}

step prepare_data  python prepare_data.py
step map_targets   python make_map_targets.py
step train_unet    python train.py --config config.yaml
step train_pixdiff python train_pixel_diffusion.py --config config.yaml
step img_features  python extract_image_features.py --config config.yaml
step train_dit     python train_dit.py --config config.yaml
step evaluate      python evaluate.py --config config.yaml
for name in no_cams no_sparse; do
  step train_dit_$name python train_dit.py --config config.yaml --ablation $name
  step eval_dit_$name  python evaluate.py --config config.yaml --ablation $name
done
echo "$(date) ALL DONE"
