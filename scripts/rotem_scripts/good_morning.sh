
snode -N
srun --nodelist=gipdeep10 --gres=gpu:1 --time=05:00:00 --pty bash
conda activate geo_splat
cd RaDe-GS/GenerateData

srun --nodelist=gipdeep10 --gres=gpu:1 --time=05:00:00 --pty bash
