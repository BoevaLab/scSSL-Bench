import torch
from torch.utils.data import Dataset

import os
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps
from collections import defaultdict
from scipy.stats import bernoulli
from scipy.sparse.csgraph import connected_components

# from geosketch import gs
from os.path import join
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture

from collections import defaultdict
from moco.prepare_dataset import prepare_dataset
from moco.preprocessing import preprocess_dataset, hvgPipe
from moco.config import Config
from moco.NNs import mnn_approx, nn_approx, reduce_dimensionality, random_walk1, computeMNNs
from moco.sNNs import generate_graph, computeAnchors

configs = Config()

# sketch_r = 0.4

class ClaireDataset(Dataset):
    '''
        unsupervised contrastive batch correction:
        postive anchors: if MNN pairs else augmented by gausian noise
        negtive anchors: all samples in batch except pos anchors
    '''
    def __init__(
            self, 
            data_dir, 
            mode='train',
            anchor_path=None,
            select_hvg=True,
            scale=False, 
            alpha=0.9,
            knn = 10,           # used to compute knn
            augment_set=['int', 'geo', 'exc'],  # 'int': interpolation, 'geo': geometric, 'exc': exchange
            exclude_fn=True,
            verbose=0
        ):
        self.mode = mode
        self.scale = scale
        self.verbose = verbose
        self.data_dir = data_dir  # data_root/dataset_name
        self.select_hvg = select_hvg
        self.augment_op_names = augment_set

        self.alpha = alpha

        # self.reset_anchors = reset_anchors
        # self.anchor_metadata = anchor_metadata
        self.load_data()     # load data first to get self.cname

        # define set of augment operatio
        self.augment_set = []
        for ai in augment_set:
            if ai=='int':
                self.augment_set.append(partial(interpolation, alpha=alpha))
            elif ai=='geo':
                self.augment_set.append(partial(geo_interpolation, alpha=alpha))
            elif ai=='exc':
                self.augment_set.append(partial(binary_switch, alpha=alpha))
            else:
                raise ValueError("Unrecognized augment operation")
        if self.verbose:
            print('Defined ops: ', self.augment_op_names)

        # define intra-batch knn
        if mode=='train':
            k_anchor = knn
            # 1. computing anchors
            self.compute_anchors(self.X, self.batch_label, self.cname, self.gname, k_anchor=k_anchor, filtering=True)

            # 2. use the anchors exported from seurat
            #self.load_anchors(anchor_path)
            
            
            self.getMnnDict()
            self.exclude_sampleWithoutMNN(exclude_fn)

            self.computeKNN(knn)


    def __len__(self):
        if self.mode=='train':
            return len(self.valid_cellidx)
        else:
            return self.X.shape[0]

    def load_anchors(self, anchor_path):
        if anchor_path is None:
            anchor_path = join(self.data_dir, 'seuratAnchors.csv')

        if self.verbose:
            print('loading anchors from ', anchor_path)

        self.anchor_metadata = pd.read_csv(anchor_path, sep=',', index_col=0)

        # convert name to global cell index
        self.anchor_metadata['cell1'] = self.anchor_metadata.name1.apply(lambda x:self.name2idx[x])
        self.anchor_metadata['cell2'] = self.anchor_metadata.name2.apply(lambda x:self.name2idx[x])

        anchors = self.anchor_metadata[['cell1', 'cell2']].values
        anchors = anchors[:(len(anchors)//2)]   # delete symmetric anchors
        self.pairs = anchors

        # print anchor info
        if self.verbose and (self.type_label is not None):
            print_AnchorInfo(self.pairs, self.batch_label, self.type_label)

    def compute_anchors(self, X, batch_label, cname, gname, k_anchor=5, filtering=True):
        print('computeing anchors')
        anchors = computeAnchors(X, batch_label, cname, gname, k_anchor=k_anchor, filtering=filtering)

        anchors.cell1 = anchors.cell1_name.apply(lambda x: self.name2idx[x])
        anchors.cell2 = anchors.cell2_name.apply(lambda x: self.name2idx[x])
        pairs = np.array(anchors[['cell1', 'cell2']])
        self.pairs = pairs

        # print anchor info
        if self.verbose and (self.type_label is not None):
            print_AnchorInfo(self.pairs, self.batch_label, self.type_label)


    def computeKNN(self, knn=10):
        # calculate knn within each batch
        self.nns = np.ones((self.n_sample, knn), dtype='long')  # allocate (N, k+1) space
        bs = self.batch_label.unique()
        for bi in bs:
            bii = np.where(self.batch_label==bi)[0]

            # dim reduction for efficiency
            X_pca = reduce_dimensionality(self.X, 50)
            nns = nn_approx(X_pca[bii], X_pca[bii], knn=knn+1)  # itself and its nns
            nns = nns[:, 1:]

            # convert local batch index to global cell index
            self.nns[bii, :] = bii[nns.ravel()].reshape(nns.shape)

        if self.verbose and (self.type_label is not None):
            print_KnnInfo(self.nns, np.array(self.type_label))

    def filter_anchors(self, emb=None, fltr='gmm', yita=.5):
        # if embeddings not provided, then using HVGs
        if emb is None:
            emb = self.X.A.copy()
            emb = emb / np.sqrt(np.sum(emb**2, axis=1, keepdims=True)) # l2-normalization

        pairs = self.pairs
        cos_sim = np.sum(emb[pairs[:, 0]] * emb[pairs[:, 1]], axis=1)  # dot prod

        if fltr=='gmm':    
            sim_pairs = cos_sim.reshape(-1, 1)
            gm = GaussianMixture(n_components=2, random_state=0).fit(sim_pairs)

            gmm_c = gm.predict(sim_pairs)
            gmm_p = gm.predict_proba(sim_pairs)

            # take the major component
            _, num_c = np.unique(gmm_c, return_counts=True)  
            c = np.argmax(num_c)

            filter_mask = gmm_p[:, c]>=yita
        # if filter is not gmm => naive filter
        # given similarity, taking quantile
        else:
            pairs = self.pairs
            cos_sim = np.sum(emb[pairs[:, 0]] * emb[pairs[:, 1]], axis=1)  # dot prod

            filter_thr = np.quantile(cos_sim, yita)   
            filter_mask = cos_sim >= filter_thr

        self.pairs = pairs[filter_mask]

    def exclude_sampleWithoutMNN(self, exclude_fn):
        self.valid_cellidx = np.unique(self.pairs.ravel()) if exclude_fn else np.arange(self.n_sample)

        if self.verbose:
            print(f'Number of training samples = {len(self.valid_cellidx)}')

    def getMnnDict(self):
        self.mnn_dict = get_mnn_graph(self.n_sample, self.pairs)

    def load_data(self):
        # customized
        sps_x, genes, cells, metadata = prepare_dataset(self.data_dir)
        X, cell_name, gene_name, metadata = preprocess_dataset(
            sps_x, 
            cells, 
            genes, 
            metadata, 
            self.select_hvg, 
            self.scale, 
            )

        self.X = X   # sparse
        self.metadata = metadata
        self.gname = gene_name
        self.cname = cell_name 
        self.n_sample = self.X.shape[0]
        self.n_feature = self.X.shape[1]
        self.name2idx = dict(zip(cell_name, np.arange(self.n_sample)))
        self.n_batch = metadata[configs.batch_key].unique().size
        self.batch_label = metadata[configs.batch_key].values

        if configs.label_key in metadata.columns:
            self.type_label = metadata[configs.label_key].values
            self.n_type = len(self.type_label.unique())
        else:
            self.type_label = None
            self.n_type = None

    def getTrainItem(self, i):
        i = self.valid_cellidx[i] if self.mode=='train' else i # translate local cell idx to global cell idx
        x = self.X[i].A.squeeze()

        # all the story starts from mnn
        pos_anchors = self.mnn_dict[i]

        # augment self
        ni = i if len(self.nns[i]) == 0 else np.random.choice(self.nns[i])
        x_n = self.X[ni].A.squeeze()
        x_aug = augment_positive(self.augment_set, x, x_n)  # augment x with knn

        # augmenting one of x's positives
        pi = np.random.choice(pos_anchors) if len(pos_anchors)>0 else i
        x_p = self.X[pi].A.squeeze()
        p_ni = pi if len(self.nns[pi])==0 else np.random.choice(self.nns[pi])
        x_p_ni = self.X[p_ni].A.squeeze()
        x_p_aug = augment_positive(self.augment_set, x_p, x_p_ni)

        return [x_aug.astype('float32'), x_p_aug.astype('float32')], [i, pi]

    def getValItem(self, i):
        x = self.X[i].A.squeeze()

        return [x.astype('float32'), x.astype('float32')], [i, i]

    def __getitem__(self, i):
        if self.mode=='train':
            return self.getTrainItem(i)
        else:
            return self.getValItem(i)


# utils
def get_mnn_graph(n_cells, anchors):
    # sparse Mnn graph
    # mnn_graph = sps.csr_matrix((np.ones(anchors.shape[0]), (anchors['cell1'], anchors['cell2'])),
    #                             dtype=np.int8)

    # # create a sparse identy matrix
    # dta, csr_ind = np.ones(n_cells,), np.arange(n_cells)
    # I = sps.csr_matrix((dta, (csr_ind, csr_ind)), dtype=np.int8)  # identity_matrix

    # mnn_list for all cells
    mnn_dict = defaultdict(list)
    for r,c in anchors:   
        mnn_dict[r].append(c)
        mnn_dict[c].append(r)
    return mnn_dict

# x_bar = x*lambda + x_p*(1-lambda)
def interpolation(x, x_p, alpha):
    lamda = np.random.uniform(alpha, 1.)  # [alpha, 1.]
    x = lamda * x + (1 - lamda) * x_p
    return x

# x_bar = x^lambda + x_p^(1-lambda)
def geo_interpolation(x, x_p, alpha):
    lamda = np.random.uniform(alpha, 1.)  # [alpha, 1.]
    x = (x**lamda) * (x_p**(1-lamda))
    return x

# x_bar = x * ber_vector + x_p * (1-ber_vector)
def binary_switch(x, x_p, alpha):
    bernou_p = bernoulli.rvs(alpha, size=len(x))
    x = x * bernou_p + x_p * (1-bernou_p)
    return x

def augment_positive(ops, x, x_p):
    # op_i = np.random.randint(0, 3)
    if len(ops)==0:  # if ops is empty, return x
        return x

    opi = np.random.randint(0, len(ops))
    sel_op = ops[opi]

    return sel_op(x, x_p) 


def print_AnchorInfo(anchors, global_batch_label, global_type_label):
    anchor2type = np.array(global_type_label)[anchors]
    correctRatio = (anchor2type[:, 0] == anchor2type[:, 1]).sum() / len(anchors)
    print('Anchors n={}, ratio={:.4f}'.format(len(anchors), correctRatio))

    anchors = anchors.ravel()
    df = pd.DataFrame.from_dict({"type": list(global_type_label[anchors]), 'cidx':anchors, \
                                "batch": list(global_batch_label[anchors])},
                                orient='columns')
    print(df.groupby('batch')['cidx'].nunique() / global_batch_label.value_counts())
    print(df.groupby('type')['cidx'].nunique() / global_type_label.value_counts())


def print_KnnInfo(nns, type_label, verbose=0):
    def sampleWise_knnRatio(ti, nn, tl):
        knn_ratio = ti == tl[nn]
        knn_ratio = np.mean(knn_ratio)
        return knn_ratio

    # corr_ratio_per_sample = np.apply_along_axis(sampleWise_knnRatio, axis=1, arr=nns)
    if isinstance(nns, defaultdict):
        corr_ratio_per_sample = []
        for k,v in nns.items():
            corr_ratio_per_sample.append(np.mean(type_label[k] == type_label[v]))
    else:
        corr_ratio_per_sample = list(map(partial(sampleWise_knnRatio, tl=type_label), type_label, nns))

    ratio = np.mean(corr_ratio_per_sample)
    print('Sample-wise knn ratio={:.4f}'.format(ratio))

    if verbose:
        return corr_ratio_per_sample