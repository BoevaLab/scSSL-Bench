# nohup python main.py --multirun +experiment=multimodal_PBMC_base_ablation +cluster=slurm >> PBMC_bc.log 2>&1 &
# nohup python main.py --multirun +experiment=multimodal_PBMC_base_ablation_qr +cluster=slurm >> PBMC_qr.log 2>&1 &

# nohup python main.py --multirun +experiment=multimodal_Neurips_base_ablation +cluster=slurm >> Neurips_bc.log 2>&1 &
# nohup python main.py --multirun +experiment=multimodal_Neurips_base_ablation_qr +cluster=slurm >> Neurips_qr.log 2>&1 &

# nohup python main.py --multirun +experiment=multimodal_PBMC_base_ablation_temp +cluster=slurm >> PBMC_temp.log 2>&1 &
# nohup python main.py --multirun +experiment=multimodal_Neurips_base_ablation_temp +cluster=slurm >> Neurips_temp.log 2>&1 &

# nohup python main.py --multirun +experiment=multimodal_Neurips_base_ablation_temp_left +cluster=slurm >> Neurips_temp_left.log 2>&1 &

# nohup python main.py --multirun +experiment=temperature_single_modality_experiment +cluster=slurm >> unimodal_temp.log 2>&1 &

# nohup python main.py --multirun +experiment=unimodal_projection_no +cluster=slurm >> unimodal_projection.log 2>&1 &

nohup python main.py --multirun +experiment=unimodal_bc_dsbn +cluster=slurm >> unimodal_bc_dsbn.log 2>&1 &
