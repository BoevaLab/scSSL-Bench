#!/bin/sh
#SBATCH -o logs/scvi-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=09:00:00
#SBATCH --gres=gpumem:8G
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END,FAIL

mkdir -p logs

conda activate myenv

# python utils/run_scVI.py --data /home/baunsgaard/scBench/scButterfly/Olga_Data/ImmuneAtlas.h5ad --batch batchlb --celltype cell_type --epoch 100

# python utils/run_scVI.py --data /home/baunsgaard/scBench/scButterfly/Olga_Data/ImmHuman.h5ad --batch batch --celltype CellType --epoch 100

# python utils/run_scVI.py --data /home/baunsgaard/scBench/scButterfly/Olga_Data/Lung.h5ad --batch batch --celltype cell_type --epoch 100

# python utils/run_scVI.py --data /home/baunsgaard/scBench/scButterfly/Olga_Data/MCA.h5ad --batch batch --celltype CellType --epoch 100

# python utils/run_scVI.py --data /home/baunsgaard/scBench/scButterfly/Olga_Data/Pancreas.h5ad --batch batch --celltype celltype --epoch 100

# python utils/run_scVI.py --data /home/baunsgaard/scBench/scButterfly/Olga_Data/PBMC.h5ad --batch batch --celltype CellType --epoch 100

python utils/run_scVI_qr.py 