#run 3d viewer
 code /home/rotem.shezaf/RaDe-GS/data
 /tnt/TNT_GOF/TrainingSet/Barn/barn_sparse_mesh_poisson.ply


 #run jobs
  srun --nodelist=gipdeep9 --gres=gpu:2 --time=00:10:00 --pty bash
#checkavaible gpus
 sinfo -N
 snode -N
 nvidia-smi
#need not to use gipdeep1, gipdeep6
# doanload submodules
 git submodule update --init --recursive
 python -m pip install --no-build-isolation  submodules/simple-knn

 python train.py -s <path to DTU dataset> -m <output folder> -r 2 --use_decoupled_appearance

unzip /home/rotem.shezaf/RaDe-GS/data/TNT_GOF.zip -d /home/rotem.shezaf/RaDe-GS/data/tnt/TNT_GOF

unzip /home/rotem.shezaf/RaDe-GS/data/TNT_GOF.zip -d /home/rotem.shezaf/RaDe-GS/data/tnt/TNT_GOF

unzip /home/rotem.shezaf/RaDe-GS/data/TNT_GOF.zip -d /home/rotem.shezaf/RaDe-GS/data/tnt/TNT_GOF

python train.py -s /home/rotem.shezaf/RaDe-GS/data/TNT_GOF/TrainingSet -m /home/rotem.shezaf/RaDe-GS/data/TNT_GOF/TrainingSet/geussians -r 2 --eval --use_decoupled_appearance

python train.py -s /home/rotem.shezaf/RaDe-GS/data/TNT_GOF/TrainingSet/Caterpillar -m /home/rotem.shezaf/RaDe-GS/data/TNT_GOF/geussians/Caterpillar -r 2 --eval --use_decoupled_appearance

python mesh_extract_tetrahedra.py -s /home/rotem.shezaf/RaDe-GS/data/TNT_GOF/TrainingSet/Caterpillar -m /home/rotem.shezaf/RaDe-GS/data/TNT_GOF/geussians/Caterpillar -r 2

#evaluate TNT
python eval_tnt/run.py --dataset-dir \
/home/rotem.shezaf/RaDe-GS/data/TNT_GOF/ground_truth/Truck --traj-path \
/home/rotem.shezaf/RaDe-GS/data/TNT_GOF/TrainingSet/Truck/Truck_COLMAP_SfM.log \
--ply-path /home/rotem.shezaf/RaDe-GS/data/TNT_GOF/geussians/Truck/recon.ply



