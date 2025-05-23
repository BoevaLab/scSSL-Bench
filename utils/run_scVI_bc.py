import argparse
import os
import tempfile

import scanpy as sc
import scvi
import seaborn as sns
import torch
from rich import print
from scib_metrics.benchmark import Benchmarker


from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

_BIO_METRICS = BioConservation(isolated_labels=True, 
                               nmi_ari_cluster_labels_leiden=True, 
                               nmi_ari_cluster_labels_kmeans=False, 
                               silhouette_label=True, 
                               clisi_knn=True
                               )
_BATCH_METRICS = BatchCorrection(graph_connectivity=True, 
                                 kbet_per_label=True, 
                                 ilisi_knn=True, 
                                 pcr_comparison=True, 
                                 silhouette_batch=True
                                 )

def get_args():
    parser = argparse.ArgumentParser(description='CONCERTO Batch Correction.')

    parser.add_argument('--data', type=str, required=True,
                        help='Dataset (Simulated/Not)')

    parser.add_argument('--epoch', type=int, required=True,
                        help='Number of epochs')
    
    parser.add_argument('--batch', type=str, required=True,
                        help='Batch key')
    
    parser.add_argument('--celltype', type=str, required=True,
                        help='Cell type')

    args = parser.parse_args()
    return args

def run_scVI(adata, batch_key, cell_type):
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key=batch_key)
    model = scvi.model.SCVI(adata, n_layers=8, n_latent=30, gene_likelihood="nb")

    model.train(accelerator="gpu")

    SCVI_LATENT_KEY = "X_scVI"
    adata.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

    # sc.pp.neighbors(adata, use_rep=SCVI_LATENT_KEY)
    # sc.tl.leiden(adata)

    # SCVI_MDE_KEY = "X_scVI_MDE"
    # adata.obsm[SCVI_MDE_KEY] = scvi.model.utils.mde(adata.obsm[SCVI_LATENT_KEY], accelerator="cpu")

    # sc.pl.embedding(adata, basis=SCVI_MDE_KEY, color=["batch", "leiden"], frameon=False, ncols=1,)
    # sc.pl.embedding(adata, basis=SCVI_MDE_KEY, color=["cell_type"], frameon=False, ncols=1)

    return adata

def evaluate_model(adata, batch_key, cell_type_label):
    names_obs = ['X_scVI']
    print(names_obs)
    bm = Benchmarker(
                adata,
                batch_key=batch_key,
                label_key=cell_type_label,
                embedding_obsm_keys=names_obs,
                bio_conservation_metrics=_BIO_METRICS,
                batch_correction_metrics=_BATCH_METRICS,
                n_jobs=4,
            )
    bm.benchmark()
    a = bm.get_results(False, True)
    results = a.round(decimals=4)
    return results

args = get_args()
adata_path = args.data # "/home/baunsgaard/scBench/scButterfly/Olga_Data/ImmuneAtlas.h5ad"
epochs = args.epoch
batch_key = args.batch
cell_type = args.celltype

adata = sc.read(adata_path)

print(adata)

adata = run_scVI(adata, batch_key=batch_key, cell_type=cell_type)
results = evaluate_model(adata=adata, batch_key=batch_key, cell_type_label=cell_type)
print(args.data.split("/")[-1].split(".")[0])
results.to_csv(f'results/scVI/{args.data.split("/")[-1].split(".")[0]}_totalvi_metrics_unscaled.csv')
