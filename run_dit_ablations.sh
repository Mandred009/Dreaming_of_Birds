#!/usr/bin/env bash
# DiT ablations (E10): train + evaluate without cameras, then without the sparse BEV. One job at a time.
# Usage: nohup ./run_dit_ablations.sh > logs/ablations.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
for name in no_cams no_sparse; do
  echo "$(date) START $name | $(free -m | awk '/Mem:/{print "RAM avail " $7 " MB"}') | $(df -h ~ | awk 'END{print "disk free " $4}')"
  python train_dit.py --config configs/dit_$name.yaml > logs/train_dit_$name.log 2>&1
  python evaluate.py --config configs/dit_$name.yaml --dit_only > logs/eval_dit_$name.log 2>&1
  echo "$(date) DONE  $name"
done
echo "$(date) ALL DONE"
