# # Multi-modal batch correction
# nohup python main.py --multirun +experiment=multimodal_PBMC_base_ablation +cluster=slurm >> PBMC_bc.log 2>&1 &
# nohup python main.py --multirun +experiment=multimodal_PBMC_base_ablation_qr +cluster=slurm >> PBMC_qr.log 2>&1 &

# # Multi-modal cell-typing
# nohup python main.py --multirun +experiment=multimodal_Neurips_base_ablation +cluster=slurm >> Neurips_bc.log 2>&1 &
# nohup python main.py --multirun +experiment=multimodal_Neurips_base_ablation_qr +cluster=slurm >> Neurips_qr.log 2>&1 &

# # Temperature
# nohup python main.py --multirun +experiment=multimodal_PBMC_base_ablation_temp +cluster=slurm >> PBMC_temp.log 2>&1 &
# nohup python main.py --multirun +experiment=multimodal_Neurips_base_ablation_temp +cluster=slurm >> Neurips_temp.log 2>&1 &

# nohup python main.py --multirun +experiment=multimodal_Neurips_base_ablation_temp_left +cluster=slurm >> Neurips_temp_left.log 2>&1 &

# nohup python main.py --multirun +experiment=temperature_single_modality_experiment +cluster=slurm >> unimodal_temp.log 2>&1 &

# # Single-modal projection
# nohup python main.py --multirun +experiment=unimodal_projection_no +cluster=slurm >> unimodal_projection_no.log 2>&1 &
# nohup python main.py --multirun +experiment=unimodal_projection_yes +cluster=slurm >> unimodal_projection_yes.log 2>&1 &

# # Cell typing
# nohup python main.py --multirun +experiment=unimodal_qr_b2 +cluster=slurm >> unimodal_qr_b2.log 2>&1 &
# nohup python main.py --multirun +experiment=unimodal_qr_b3 +cluster=slurm >> unimodal_qr_b3.log 2>&1 &
# nohup python main.py --multirun +experiment=unimodal_qr_b4 +cluster=slurm >> unimodal_qr_b4.log 2>&1 &
# nohup python main.py --multirun +experiment=unimodal_qr_b5 +cluster=slurm >> unimodal_qr_b5.log 2>&1 &

# nohup python main.py --multirun +exp=qr_big +cluster=slurm >> qr_big.log 2>&1 &

# Augmentations
# nohup python main.py --multirun +aug_exp=bbknn_ablation +cluster=slurm >> bbknn_qr_b2.log 2>&1 &
# nohup python main.py --multirun +aug_exp=gauss_ablation +cluster=slurm >> gauss_ablation.log 2>&1 &
# nohup python main.py --multirun +aug_exp=innerswap_ablation +cluster=slurm >> innerswap_ablation.log 2>&1 &
# nohup python main.py --multirun +aug_exp=mask_ablation +cluster=slurm >> mask_ablation.log 2>&1 &
# nohup python main.py --multirun +aug_exp=mnn_ablation +cluster=slurm >> mnn_ablation.log 2>&1 &
# nohup python main.py --multirun +aug_exp=combi-bbknn-all +cluster=slurm >> combi_bbknn_qr_b2.log 2>&1 &
# nohup python main.py --multirun +aug_exp=combi-gauss-all +cluster=slurm >> combi_gauss_qr_b2.log 2>&1 &
# nohup python main.py --multirun +aug_exp=combi-mask-all +cluster=slurm >> combi_mask_qr_b2.log 2>&1 &
# nohup python main.py --multirun +aug_exp=combi-swap-all +cluster=slurm >> combi_swap_qr_b2.log 2>&1 &

# nohup python main.py --multirun +aug_exp=combi-bbknn-all >> combi_bbknn_qr_b2.log 2>&1 &
# nohup python main.py --multirun +aug_exp=combi-mask-all >> combi_mask_qr_b2.log 2>&1 &
# nohup python main.py --multirun +aug_exp=combi-gauss-all >> combi_gauss_qr_b2.log 2>&1 &
# nohup python main.py --multirun +aug_exp=combi-swap-all >> combi_swap_qr_b2.log 2>&1 &

# nohup python main.py --multirun +aug_exp=combi-bbknn-all_pbmc >> combi_bbknn_qr_b2_new.log 2>&1 &
# nohup python main.py --multirun +aug_exp=combi-mask-all_pbmc >> combi_mask_qr_b2.log 2>&1 &
# nohup python main.py --multirun +aug_exp=combi-gauss-all_pbmc >> combi_gauss_qr_b2.log 2>&1 &
# nohup python main.py --multirun +aug_exp=combi-swap-all_pbmc >> combi_swap_qr_b2.log 2>&1 &

# Representations dimensionality
# nohup python main.py --multirun +exp=barlowtwins_ablation +cluster=slurm_5h >> logs_dim_5h/barlowtwins_ablation.log 2>&1 &
# nohup python main.py --multirun +exp=byol_ablation +cluster=slurm_5h >> logs_dim_5h/byol_ablation.log 2>&1 &
# nohup python main.py --multirun +exp=moco_ablation +cluster=slurm_5h >> logs_dim_5h/moco_ablation.log 2>&1 &
# nohup python main.py --multirun +exp=nnclr_ablation +cluster=slurm_5h >> logs_dim_5h/nnclr_ablation.log 2>&1 &
# nohup python main.py --multirun +exp=simclr_ablation +cluster=slurm_5h >> logs_dim_5h/simclr_ablation.log 2>&1 &
# nohup python main.py --multirun +exp=simsiam_ablation +cluster=slurm_5h >> logs_dim_5h/simsiam_ablation.log 2>&1 &
# nohup python main.py --multirun +exp=vicreg_ablation +cluster=slurm_5h >> logs_dim_5h/vicreg_ablation.log 2>&1 &

# nohup python main.py --multirun +aug_exp=combi-mnn >> combi_mnn.log 2>&1 &
# nohup python main.py --multirun +aug_exp=combi-crossover >> combi_crossover.log 2>&1 &
# nohup python main.py --multirun +aug_exp=combi-bbknn-all >> combi_bbknn.log 2>&1 &

# nohup python run_scib.py  >> metrics_sapiens.log 2>&1 &
nohup python run_scib_vicreg.py  >> metrics_sapiens_vicreg.log 2>&1 &

# nohup python main.py --multirun +big_data=sapiens_bc >> sapiens_bc.log 2>&1 &
# nohup python main.py --multirun +big_data=sapiens_qr >> sapiens_qr.log 2>&1 &

nohup python main.py --multirun +big_data=sapiens_bc_1 >> sapiens_bc_1_100.log 2>&1 &
nohup python main.py --multirun +big_data=sapiens_bc_2 >> sapiens_bc_2_100.log 2>&1 &
nohup python main.py --multirun +big_data=sapiens_bc_3 >> sapiens_bc_3_100.log 2>&1 &

nohup python main.py --multirun +big_data=sapiens_bc_4 >> sapiens_bc_4_100.log 2>&1 &
# nohup python main.py --multirun +big_data=sapiens_bc_5 >> sapiens_bc_5_100.log 2>&1 &
nohup python main.py --multirun +big_data=sapiens_bc_6 >> sapiens_bc_6_100.log 2>&1 &

nohup python main.py --multirun +big_data=sapiens_qr_1 >> sapiens_qr_1_100.log 2>&1 &
nohup python main.py --multirun +big_data=sapiens_qr_2 >> sapiens_qr_2_100.log 2>&1 &
nohup python main.py --multirun +big_data=sapiens_qr_3 >> sapiens_qr_3_100.log 2>&1 &
