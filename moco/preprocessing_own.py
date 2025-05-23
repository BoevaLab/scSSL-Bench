import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
# from imblearn.under_sampling import RandomUnderSampler

from moco.config import Config

configs = Config()
sc.settings.verbosity = configs.verbose 

# ====================================================
# preprocessing for computing MNNs
# ====================================================
def preprocess_dataset(sps_x, cell_name, gene_name, df_meta, select_hvg=None, scale=False, preprocess=True):
    # compute hvg first, anyway
    adata = sc.AnnData(sps.csr_matrix(sps_x.T))  # transposed, (gene, cell) -> (cell, gene)
    adata.obs_names = cell_name
    adata.var_names = gene_name
    adata.obs = df_meta.loc[cell_name].copy()
    adata.layers['counts'] = sps.csr_matrix(sps_x.T)

    if preprocess:
        sc.pp.filter_genes(adata, min_cells=configs.min_cells) # 0
        sc.pp.normalize_total(adata, target_sum=configs.scale_factor) # TODO : 10K oder 100K
        sc.pp.log1p(adata)

    if select_hvg is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=min(adata.shape[1], select_hvg), 
                                    # min_mean=0.0125, max_mean=3, min_disp=0.5,
                                    batch_key=configs.batch_key, subset=True)

        #adata = adata[:, adata.var.highly_variable].copy()
        # DEV-INFO: PBMCFull had issues because > hvg genes were kept.

    
    if scale: # try
        warnings.warn('Scaling per batch! This may cause memory overflow!')
        ada_batches = []
        for bi in adata.obs[configs.batch_key].unique():
            bidx = adata.obs[configs.batch_key] == bi
            adata_batch = adata[bidx].copy()
            sc.pp.scale(adata_batch)

            ada_batches.append(adata_batch)

        adata = sc.concat(ada_batches)

    X = sps.csr_matrix(adata.X)    # some times 
    df_meta = adata.obs.copy()
    cell_name = adata.obs_names

    df_meta[[configs.batch_key, configs.label_key]] = df_meta[[configs.batch_key, configs.label_key]].astype('category')
    #adata.layers['raw_counts'] = sps.csr_matrix(sps_x.T)

    return adata, X, cell_name, adata.var_names, df_meta

def LouvainPipe(sps_x, df_meta, hvg=True, scale=True, npcs=50, n_neighbors=15, r=1.):
    print(f'preprocessing dataset, shape=({sps_x.shape[0]}, {sps_x.shape[1]})')

    # compute hvg first, anyway
    adata = sc.AnnData(sps.csr_matrix(sps_x))  # transposed before
    adata.obs = df_meta.copy()

    sc.pp.filter_genes(adata, min_cells=configs.min_cells) 
    sc.pp.normalize_total(adata, target_sum=configs.scale_factor)
    sc.pp.log1p(adata)

    if hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=min(adata.shape[1]-1, configs.n_hvgs), 
                                    min_mean=0.0125, max_mean=3, min_disp=0.5,
                                    )

        adata = adata[:, adata.var.highly_variable].copy()
    if scale:
        sc.pp.scale(adata, max_value=10)   # X -> array
    sc.pp.pca(adata, n_comps=npcs) # svd_solver='arpack' not accept sparse input

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=npcs)
    if r is not None:
        sc.tl.louvain(adata, resolution=r, key_added='louvain')

    return adata

def hvgPipe(X, meta=None, scale=False, n_neighbors=15, npcs=50, umap=True):
    adata = sc.AnnData(X)

    if meta is not None:
        adata.obs = meta.copy()

    if scale:
        sc.pp.scale(adata, max_value=None)

    sc.pp.pca(adata, n_comps=npcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)

    if umap:
        sc.tl.umap(adata)

    return adata


def embPipe(X, meta, n_neighbors=15):
    adata = sc.AnnData(X)
    adata.obs = meta.copy()

    #sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
    #sc.tl.umap(adata)

    return adata

