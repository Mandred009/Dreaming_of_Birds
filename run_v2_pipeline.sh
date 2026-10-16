#!/usr/bin/env bash
# Runs the full v2 trainval pipeline in order, one step at a time (RAM is tight, never two at once).
# Usage: nohup ./run_v2_pipeline.sh [pid_to_wait_for] > logs/pipeline.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

if [ -n "${1:-}" ]; then
  echo "$(date) waiting for PID $1 (preprocessing) to finish"
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi

step() {   # step <name> <command...>
  local name=$1; shift
  echo "$(date) START $name | $(free -m | awk '/Mem:/{print "RAM avail " $7 " MB"}') | $(df -h ~ | awk 'END{print "disk free " $4}')"
  "$@" > "logs/$name.log" 2>&1
  echo "$(date) DONE  $name"
}

step map_targets   python make_map_targets.py
step train_unet    python train.py --config config.yaml
step train_pixdiff python train_pixel_diffusion.py --config config.yaml
step img_features  python extract_image_features.py --config config.yaml
step train_dit     python train_dit.py --config config.yaml
step evaluate      python evaluate.py --config config.yaml
echo "$(date) ALL DONE"
