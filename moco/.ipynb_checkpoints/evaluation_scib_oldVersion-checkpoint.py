import os
import scIB
import datetime
# import scanorama
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
import matplotlib.pyplot as plt

from os.path import join
from functools import reduce

from scIB.clustering import opt_louvain
from scIB.metrics import nmi, ari, silhouette, silhouette_batch, isolated_labels
from scIB.metrics import graph_connectivity, trajectory_conservation, clisi_graph, ilisi_graph

# batch_key = 'batchlb'
# label_key = 'CellType'
embed = 'X_emb'

def scib_pp(
           csv_path,
           npcs=50,
           batch_key = 'batchlb',
           label_key = 'celltype'
            ):
    '''
        假设csv.file是已经降维过的数据： pca 或者 correction本身导出的low-d embedding(经pca变换)
        take the former npcs=50
        最后两列为batch与type的标签: batchlb, CellType
    '''
    df = pd.read_csv(csv_path, sep='\t', index_col=0)
    pcCols = [f'PC{i+1}' for i in range(npcs)]
    adata = sc.AnnData(df[pcCols].values)
    adata.obs_names = df.index.values
    adata.var_names = pcCols
    if "X_pca" in adata.obsm.keys():
        adata.obsm['X_emb'] = adata.obsm['X_pca']
    else:
        adata.obsm["X_emb"] = adata.X
    adata.obs[batch_key] = df[batch_key].astype('category')
    adata.obs[label_key] = df[label_key].astype('category')

    sc.pp.neighbors(adata, n_pcs=npcs)  # default 15

    return adata

def sc_prep(data, metadata, scale=False, n_neighbors=15, n_pcs=50, umap=True):
    '''
        suppose data is after batch corrected, in normalized format 
    '''
    adata = sc.AnnData(data)
    adata.obs = metadata
    
    if scale:
        sc.pp.scale(adata, max_value=None)
    
    if data.shape[1] > n_pcs:
        sc.pp.pca(adata, n_comps=n_pcs, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=min(n_pcs, min(adata.shape[0]-1, adata.shape[1]-1)))
    else:
        print(f'n_features <= n_pcs, {data.shape[1]} <= {n_pcs}')
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=None) # use raw.X
    
    if umap:
        sc.tl.umap(adata)
    
    return adata

def scib_process(
            adata, 
            adata_raw=None,
            embed = embed,
            batch_key='batchlb', 
            label_key='celltype',
            nmi_=True, nmi_method='arithmetic', save_nmi=None,
            ari_=True,
            silhouette_=False, si_metric='euclidean',
            # pcr_=pcr_,
            # cell_cycle_=cell_cycle_, organism=organism,
            isolated_labels_=False, n_isolated=None,
            graph_conn_=False,
            type_='embed', # all the data is in the embedded dimensions
            kBET_=False, 
            subsample=0.5,
            clisi_= False,
            ilisi_= False, 
            trajectory_=False
            ):
    '''
        adata.obsm['X_emb']中存储各correction方法导出的低维结果(或直接结果或经pca变换)
    '''
    cluster_key = 'tmp_cluster'  # used for saving tmp clustering results
    if nmi_ or ari_:
        res_max, nmi_max, nmi_all = opt_louvain(
            adata,
            label_key=label_key,
            cluster_key=cluster_key,
            function=nmi,
            plot=False,
            verbose=False,
            inplace=True,
            force=True
        )
        # save all the (resolution, nmi) to csv
        if save_nmi is not None:
            nmi_all.to_csv(save_nmi, header=False)
            print(f'saved clustering NMI values to {save_nmi}')

    # results = {}

    if nmi_:
        print('NMI...')
        nmi_score = nmi(
            adata,
            group1=cluster_key,
            group2=label_key,
            method=nmi_method,
            nmi_dir=save_nmi
        )
    else:
        nmi_score = np.nan

    if ari_:
        print('ARI...')
        ari_score = ari(
            adata,
            group1=cluster_key,
            group2=label_key
        )
    else:
        ari_score = np.nan

    # 该asw值比common asw要高，计算为ASW_c = (ASW_type + 1)/2 
    if silhouette_:
        print('Silhouette score...')
        # global silhouette coefficient
        sil_global = silhouette(
            adata,
            group_key=label_key,
            embed=embed,
            metric=si_metric
        )
        # 按label划分样本，如某个label下包含多个batch，则计算: 1 - abs(ASW_batch)
        # 越大越好
        _, sil_clus = silhouette_batch(
            adata,
            batch_key=batch_key,
            group_key=label_key,
            embed=embed,
            metric=si_metric,
            verbose=False
        )
        # 各batch sil.score取mean
        # 分析各batch在correction前后，heterogeneity的变化，但只从corr后来反应，越高越好
        sil_clus = sil_clus['silhouette_score'].mean()  
    else:
        sil_global = np.nan
        sil_clus = np.nan

    if isolated_labels_:
        print("Isolated labels F1...")
        il_score_f1 = isolated_labels(
            adata,
            label_key=label_key,
            batch_key=batch_key,
            embed=embed,
            cluster=True,  # if True, 通过louvain判别识别各isolated_type的F1.score
            n=n_isolated,
            verbose=False
        )

        print("Isolated labels ASW...")
        il_score_asw = isolated_labels(
            adata,
            label_key=label_key,
            batch_key=batch_key,
            embed=embed,
            cluster=False,  # if else, 基于silhouette判断isolated_type与rest.cluster的gap
            n=n_isolated,   
            verbose=False
        )
    else:
        il_score_f1 = np.nan
        il_score_asw = np.nan


    if graph_conn_:
        print('Graph connectivity...')
        graph_conn_score = graph_connectivity(
            adata,
            label_key=label_key
        )
    else:
        graph_conn_score = np.nan

    if trajectory_:
        print('trajectory conservation...')
        trajectory_score = trajectory_conservation(
            adata_raw,
            adata,
            label_key=label_key,
            # batch_key=batch_key
        )
    else:
        trajectory_score = np.nan
        
    if clisi_:
        print('cLISI score...')
        clisi = clisi_graph(
            adata,
            batch_key=batch_key,
            label_key=label_key,
            type_=type_,
            subsample=subsample * 100,
            scale=True,
            multiprocessing=True,
            verbose=False
        )
    else:
        clisi = np.nan

    if ilisi_:
        print('iLISI score...')
        ilisi = ilisi_graph(
            adata,
            batch_key=batch_key,
            type_=type_,
            subsample=subsample * 100,
            scale=True,
            multiprocessing=True,
            verbose=False
        )
    else:
        ilisi = np.nan
   

    results = {
        'NMI_cluster/label': nmi_score,
        'ARI_cluster/label': ari_score,
        'ASW_label': sil_global,
        'ASW_label/batch': sil_clus,
        'isolated_label_F1': il_score_f1,
        'isolated_label_silhouette': il_score_asw,
        'graph_conn': graph_conn_score,
        'trajectory': trajectory_score,
        'iLISI': ilisi,
        'cLISI': clisi,
    }

    return pd.DataFrame.from_dict(results, orient='index')

from scipy.stats import entropy, itemfreq
from sklearn.neighbors import NearestNeighbors
def entropy_batch_mixing(latent, labels, n_neighbors=50, n_pools=50, n_samples_per_pool=100):
    n_batches = np.unique(labels).size

    def entropy_from_indices(indices):
        
        # return entropy(np.array(itemfreq(indices)[:, 1].astype(np.int32)))
        return entropy(np.unique(indices, return_counts=True)[1].astype(np.int32), base=n_batches)

    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(latent)
    indices = neighbors.kneighbors(latent, return_distance=False)[:, 1:]
    batch_indices = np.vectorize(lambda i: labels[i])(indices)

    entropies = np.apply_along_axis(entropy_from_indices, axis=1, arr=batch_indices)

    # average n_pools entropy results where each result is an average of n_samples_per_pool random samples.
    if n_pools == 1:
        score = np.mean(entropies)
    else:
        score = np.mean([
            np.mean(entropies[np.random.choice(len(entropies), size=n_samples_per_pool)])
            for _ in range(n_pools)
        ])    
    
    return score


# nearest neighbors error
def nearest_neighbor_error(latent, labels, n_neighbors=20, n_pools=50, n_samples_per_pool=100):
    '''
        suppose latent, labels boch saved in array

    '''
    def correct_ratio_along_sample(ind):
        return ind.sum()*1.0 / len(ind)

    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(latent)
    indices = neighbors.kneighbors(latent, return_distance=False)[:, 1:]

    correct_nn = labels[indices.reshape(-1)].reshape((-1, n_neighbors)) == labels.reshape((-1, 1))
    correct_ratio_per_sample = np.apply_along_axis(correct_ratio_along_sample, axis=1, arr=correct_nn)

    if n_pools == 1:
        return np.mean(correct_ratio_per_sample)
    else:
        scores = np.mean(
                [np.mean(correct_ratio_per_sample[np.random.choice(len(correct_ratio_per_sample), size=n_samples_per_pool)])
                for _ in range(n_pools)]
            )
        return scores



