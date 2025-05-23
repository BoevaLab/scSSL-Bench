import math
import pickle
import datetime
import scipy.sparse as sps
import scanpy as sc
import pandas as pd
import numpy as np
import os
from os.path import join
from sklearn.preprocessing import MinMaxScaler

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def create_dirs(dirs):
    for _dir in dirs:
        os.makedirs(_dir, exist_ok=True)


def py_read_data(_dir, fname):
    # read data in sps
    # saved in (cells, genes)
    sps_X = sps.load_npz(join(_dir, fname+'.npz'))

    # read gene names
    with open(join(_dir, fname+'_genes.pkl'), 'rb') as f:
        genes = pickle.load(f)


    # read cell names
    with open(join(_dir, fname+'_cells.pkl'), 'rb') as f:
        cells = pickle.load(f)

    return sps_X, cells, genes

def load_meta_txt(path, delimiter='\t'):
    st = datetime.datetime.now()
    data, colname, cname = [], [], []
    with open(path, 'r') as f:
        for li, line in enumerate(f):
            line = line.strip().replace("\"", '').split(delimiter)

            if li==0:
                colname = line
                continue

            cname.append(line[0])
            
            data.append(line[1:])
    df = pd.DataFrame(data, columns=colname, index=cname)
    
    ed = datetime.datetime.now()
    total_seconds = (ed-st).total_seconds() * 1.0
    print('The reading cost time {:.4f} secs'.format(total_seconds))
    return df

def load_meta_txt7(path, delimiter='\t'):
    st = datetime.datetime.now()
    data, colname, cname = [], [], []
    with open(path, 'r') as f:
        for li, line in enumerate(f):
            line = line.strip().replace("\"", '').split(delimiter)

            if li==0:
                colname = line
                continue

            cname.append(line[0])
            
            data.append(line[1:])
    df = pd.DataFrame(data, columns=colname[1:], index=cname)
    
    ed = datetime.datetime.now()
    total_seconds = (ed-st).total_seconds() * 1.0
    print('The reading cost time {:.4f} secs'.format(total_seconds))
    return df


def find_last(log_dir):
    ckpts = next(os.walk(log_dir))[2]
    ckpts = sorted(filter(lambda x:x.endswith('.pth'), ckpts))

    if not ckpts:
        import errno
        raise FileNotFoundError(
            errno.ENOENT,
            "Could not find model ckpts"
            )

    ckpt = join(log_dir, ckpts[-1])
    return ckpt

def ListAppend(Ls, values):
    for L,v in zip(Ls, values):
        L.append(v)

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr
    if args.cosine:  # control lr decay from lr to eta_min=lr*1e-3
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.eps)) / 2
    else:               # cool, every steps[i], decay: lr=lr * rate 
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    # linear warm up the learning rate
    # increasing from warmup_from -> warmup_to
    if args.warm and epoch <= args.warm_eps:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_eps * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


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

from scanorama import assemble
def integrate_gene(integrated_emb, raw_X, metadata, gname, 
                   scale=False, batch_key='batchlb', label_key='CellType',
                   knn=10, sigma=20, alpha=0.1):

    adata_raw = sc.AnnData(raw_X)
    adata_raw.obs = metadata.copy()
    adata_raw.var_names = gname

    # split batches 
    batch_vect = adata_raw.obs[batch_key].values
    batch_set = np.unique(batch_vect)

    # create inputs for scanorama
    datasets_dimred, datasets, metas = [], [], []
    for bi in batch_set:
        idx = batch_vect == bi

        datasets_dimred.append(integrated_emb[idx].copy())
        
        adata_bi = adata_raw[idx].copy()
        if scale:
            sc.pp.scale(adata_bi)
        datasets.append(sps.csr_matrix(adata_bi.X))  # sparse matrix is required

        metas.append(adata_bi.obs)
        
    datasets_dimred = assemble(
            datasets_dimred, # Assemble in low dimensional space.
            expr_datasets=datasets, # Modified in place.
            verbose=False, knn=knn, sigma=sigma, approx=True,
            alpha=alpha, ds_names=None, batch_size=None,
    )
    
    integrated_gene_matrix = sps.vstack(datasets)
    return integrated_gene_matrix