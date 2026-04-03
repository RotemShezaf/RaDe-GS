#!/bin/bash
tmux new-session -d -s colmap_session
tmux send-keys -t colmap_session 'srun --nodelist=gipdeep11 --gres=gpu:1 --time=10:00:00 --pty bash' C-m
tmux send-keys -t colmap_session 'cd /home/rotem.shezaf/RaDe-GS' C-m
tmux send-keys -t colmap_session 'bash scripts/render_all_surfaces.sh' C-m
tmux send-keys -t colmap_session 'conda activate geo_splat' C-m
tmux attach -t colmap_session
