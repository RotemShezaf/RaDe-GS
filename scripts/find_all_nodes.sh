#!/bin/bash
sinfo -N -h -o "%N" > nodes.txt
for n in $(cat nodes.txt); do
  echo "===== $n ====="
  ssh $n "nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader"
done