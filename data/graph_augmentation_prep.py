"""
This file revises & optimizes the claire & bbknn data-augmentation.
"""
import os
import scanpy as sc
import pandas as pd
import numpy as np
import json
import dbm
import omegaconf

import omegaconf

import scipy.sparse as sps
from itertools import product
from collections import defaultdict
from sklearn.mixture import GaussianMixture

from moco.prepare_dataset import prepare_dataset
from moco.preprocessing_own import preprocess_dataset # remove hvgPipe from source code
from moco.sNNs import generate_graph # remove generate_graph from source code
from moco.NNs import nn_approx, reduce_dimensionality # remove mnn_approx, random_walk1, computeMNNs from source code


# TODO: solve this later.
from moco.config import Config
configs = Config()


class PreProcessingModule():

    def __init__(
            self,
            data_dir,
            select_hvg=True,
            scale=False,
            preprocess=True,
            multimodal=False,
            holdout_batch=None
        ):
        self.scale = scale
        self.data_dir = data_dir  # data_root/dataset_name
        self.select_hvg = select_hvg
        self.preprocess = preprocess
        self.multimodal = multimodal
        self.holdout_batch = holdout_batch
        self.load_data()
    
    def load_data(self):
        # customized
        sps_x, genes, cells, metadata = prepare_dataset(self.data_dir)
        if type(metadata) == list and len(metadata) == 3:
            metadata, X_cnv, cnv_mapping = metadata
        elif type(metadata) == list and len(metadata) == 2 and self.multimodal:
            metadata, modality = metadata

        if type(self.holdout_batch) == str:
            fltr = list(metadata[configs.batch_key] != self.holdout_batch)
            sps_x, cells, metadata = sps_x[:, fltr], cells[fltr], metadata[fltr]
        elif type(self.holdout_batch) == omegaconf.listconfig.ListConfig:
            fltr = [metadata[configs.batch_key][i] not in self.holdout_batch for i in range(len(metadata))]
            sps_x, cells, metadata = sps_x[:, fltr], cells[fltr], metadata[fltr]

        adata, X, cell_name, gene_name, metadata = preprocess_dataset(
            sps_x,
            cells,
            genes,
            metadata,
            self.select_hvg,
            self.scale,
            self.preprocess
        )
        
        self.X = X   # sparse
        self.metadata = metadata
        self.gname = gene_name
        self.cname = cell_name
        self.adata = sc.AnnData(X)
        self.adata.layers['counts'] = adata.layers['counts']
        
        self.adata.var_names = gene_name
        self.adata.obs = metadata.copy()
        self.adata.uns = adata.uns
        self.adata.var = adata.var

        if self.multimodal:
            self.adata.var["modality"] = modality
        
        self.n_sample = self.X.shape[0]
        self.n_feature = self.X.shape[1]
        self.name2idx = dict(zip(cell_name, np.arange(self.n_sample)))
        self.n_batch = metadata[configs.batch_key].unique().size
        self.batch_label = metadata[configs.batch_key].values

        if self.n_batch > 0:
            self.unique_label = list(set(self.batch_label))
            self.label_encoder = {k: v for k, v in zip(self.unique_label, range(len(self.unique_label)))}
            self.label_decoder = {v: k for k, v in self.label_encoder.items()}

        if configs.label_key in metadata.columns:
            self.type_label = metadata[configs.label_key].values
            self.n_type = len(self.type_label.unique())
        else:
            self.type_label = None
            self.n_type = None
        
        if self.type_label is not None:
            self.unique_clabel = list(set(self.type_label))
            self.clabel_encoder = {k: v for k, v in zip(self.unique_clabel, range(len(self.unique_clabel)))}
            self.clabel_decoder = {v: k for k, v in self.clabel_encoder.items()}

    def __len__(self):
        return self.n_sample

#############################################################################################################################

class ClaireAugment(PreProcessingModule):

    def __init__(
            self,
            data_dir,
            #mode='train',
            select_hvg=True,
            scale=False, 
            #alpha=0.9,
            knn = 10, # defines number of neighbors to compute
            exclude_fn=True,
            k_anchor=5,
            filtering=True, 
            preprocess=True,
            multimodal=False,
            **kwargs,
        ):
        super().__init__(data_dir, select_hvg, scale, preprocess, multimodal, **kwargs)

        self.k_anchor = k_anchor
        self.knn = knn
        self.exclude_fn = exclude_fn

        # 1. computing anchors
        flag = self.load_anchors()
        print(f"Reusing anchors: {flag}")
        if not flag:
            self.compute_anchors(self.X, self.batch_label, self.cname, self.gname, k_anchor=k_anchor, filtering=filtering)
        
        # 2. use the anchors exported from seurat
        self.exclude_sampleWithoutMNN(exclude_fn)
        flag = self.load_knn_graph()
        if not flag:
            self.computeKNN(knn)
            self.store_knn_graph()
    
    def compute_anchors(self, X, batch_label, cname, gname, k_anchor=5, filtering=True):
        print('Computing Anchors')
        norm_list, cname_list = [], []
        bs = np.unique(batch_label)
        n_batch = bs.size

        X = X.copy()

        for bi in bs:
            bii = np.where(batch_label==bi)[0]
            X_bi = X[bii].A if sps.issparse(X) else X[bii]
            cname_bi = cname[bii]

            norm_list.append(X_bi.T)  # (gene*, cell)
            cname_list.append(np.array(cname_bi))

        combine = list(product(np.arange(n_batch), np.arange(n_batch)))
        combine = list(filter(lambda x: x[0]<x[1], combine))               # non symmetric 
        anchors = generate_graph(norm_list, cname_list, gname, combine, 
                                    num_cc=20, k_filter=200, k_neighbor=k_anchor, filtering=filtering)
        self.anchors = anchors

        """
        TODO: More for filtering?
        """

        self.mnn_dict = defaultdict(list)
        anchors = self.anchors

        anchors = anchors[['cell1', 'cell2']].values
        for l,r in anchors:
            l, r = int(l), int(r)
            if r not in self.mnn_dict[l]:
                self.mnn_dict[l].append(r)
            if l not in self.mnn_dict[r]:
                self.mnn_dict[r].append(l)
        print('Finished Computing Anchors.')
        self.store_mnn_dict()

    def store_mnn_dict(self):
        # Call this after computing the anchors or after filtering.
        if self.exclude_fn:
            path = os.path.join(self.data_dir, f'anchors-{self.k_anchor}-{self.select_hvg}-{str(self.holdout_batch)}-exclude_fn.json')
        else:
            path = os.path.join(self.data_dir, f'anchors-{self.k_anchor}-{self.select_hvg}-{str(self.holdout_batch)}.json')
        with open(path, 'w') as f:
            json.dump(self.mnn_dict, f)
    
    def load_anchors(self):
        if self.exclude_fn:
            path = os.path.join(self.data_dir, f'anchors-{self.k_anchor}-{self.select_hvg}-{str(self.holdout_batch)}-exclude_fn.json')
        else:
            path = os.path.join(self.data_dir, f'anchors-{self.k_anchor}-{self.select_hvg}-{str(self.holdout_batch)}.json')
        if os.path.exists(path):
            with open(os.path.join(self.data_dir, f'anchors-{self.k_anchor}-{self.select_hvg}-{str(self.holdout_batch)}.json'), 'r') as f:
                try:
                    self.mnn_dict = json.load(f)
                    self.mnn_dict = {int(k): self.mnn_dict[k] for k in self.mnn_dict.keys()}
                except:
                    return False
            print('loading anchors from existing file.')
            return True
        else:
            return False
    
    def exclude_sampleWithoutMNN(self, exclude_fn):
        self.valid_cellidx = np.unique(list(self.mnn_dict.keys())) if exclude_fn else np.arange(self.n_sample)
        print(f"Samples before: {self.n_sample}, Samples after: {self.valid_cellidx.shape}")
        dct = defaultdict(list)
        for cid in self.valid_cellidx:
            dct[int(cid)] = self.mnn_dict[cid] if cid in self.mnn_dict.keys() else [int(cid)]
        self.mnn_dict = dct
        self.n_sample = len(self.valid_cellidx)
        self.store_mnn_dict()
        print(f'Number of training samples = {len(self.valid_cellidx)}')

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
    
    def store_knn_graph(self):
        assert self.nns is not None and len(self.nns) > 0, "Error wwhen storing nearest nns for Claire-Augmentation."
        dct = defaultdict(list)
        for i in range(len(self.nns)):
            dct[i] = [int(k) for k in self.nns[i]]
        with open(os.path.join(self.data_dir, f'knn-{self.knn}-{self.select_hvg}.json'), 'w') as f:
            json.dump(dct, f)
        self.nns = dct
    
    def load_knn_graph(self):
        if os.path.exists(os.path.join(self.data_dir, f'knn-{self.knn}-{self.select_hvg}-{str(self.holdout_batch)}.json')):
            with open(os.path.join(self.data_dir, f'knn-{self.knn}-{self.select_hvg}-{str(self.holdout_batch)}.json'), 'r') as f:
                try:
                    self.nns = json.load(f)
                    self.nns = {int(k): self.nns[k] for k in self.nns.keys()}
                except:
                    return False
            print('loading knn-graph from existing file.')
            return True
        else:
            return False

"""
DEV-INFO -- DEPRECATED:
What __get__(i) does at the moment:

Intra-batch:
(1) Use a neighboring cell from the same batch as i, and augment by combining the two.
ni = [i] if len(self.nns[i]) == 0 else np.random.choice(self.nns[i], size=1)

Inter-batch:
(1) find all anchors for a cell i
pos_anchors = self.mnn_dict[i]
(2) Choose one of the anchors to use for the augmentation.
pi = np.random.choice(pos_anchors) if len(pos_anchors)>0 else i
(3) Find NN in that batch, and augment.
"""

"""
So. What to do:

Per Batch, compute k-nearest-neighbor graph. --> NNs-list

Per Cell, compute k_anchor mutual-nearest-neighbors for each of the disjunct batches. --> pos_anchors : {id : [MNN-ids]}, eg. as a json-file.
Handle what happens with cells with MNNs.

Move the actual augmentation process into the "inheriting" (torch.nn.Module).
"""

#############################################################################################################################
#############################################################################################################################
#############################################################################################################################
#############################################################################################################################


class BbknnAugment(PreProcessingModule):
    
    def __init__(
            self,
            data_dir,
            select_hvg=True,
            scale=False, 
            knn = 10, # defines number of neighbors to compute
            exclude_fn=False,
            trim_val=None, 
            preprocess=True,
            multimodal=False,
            **kwargs,
        ):
        super().__init__(data_dir, select_hvg, scale, preprocess, multimodal, **kwargs,)
        self.knn = knn
        self.exclude_fn = exclude_fn
        self.trim = trim_val

        # 1. computing anchors
        flag = self.load_neighbors()
        print(f"Reusing bbknn: {flag}")
        if not flag:
            # TODO: Change hard-coded batchlb
            self.compute_nns(adata=self.adata, batch_label="batchlb", knn=knn, trim=trim_val)
            self.store_nn_dict()
        
        #self.exclude_sampleWithoutNNS(exclude_fn)#.exclude_sampleWithoutMNN(exclude_fn)

    def store_nn_dict(self):
        # Call this after computing the anchors or after filtering.
        if self.exclude_fn:
            path = os.path.join(self.data_dir, f'bbknn-{self.knn}-{self.select_hvg}-exclude_fn.json')
        else:
            path = os.path.join(self.data_dir, f'bbknn-{self.knn}-{self.select_hvg}.json')
        with open(path, 'w') as f:
            json.dump(self.nns, f)
    
    def load_neighbors(self):
        if self.exclude_fn:
            path = os.path.join(self.data_dir, f'bbknn-{self.knn}-{self.select_hvg}-exclude_fn.json')
        else:
            path = os.path.join(self.data_dir, f'bbknn-{self.knn}-{self.select_hvg}.json')
        if os.path.exists(path):
            with open(os.path.join(self.data_dir, f'bbknn-{self.knn}-{self.select_hvg}.json'), 'r') as f:
                try:
                    self.nns = json.load(f)
                    self.nns = {int(k): self.nns[k] for k in self.nns.keys()}
                    self.valid_cellidx = np.unique(list(self.nns.keys()))
                    tmp = self.nns.keys()
                    for i in range(len(self.adata)):
                        if i not in tmp:
                            self.nns[i].append(i)
                except:
                    return False
            print('loading anchors from existing file.')
            return True
        else:
            return False
    
    def compute_nns(self, adata, batch_label, knn, trim):
        print('computing bbknn graph')
        sc.tl.pca(data=adata,n_comps=50)
        if trim is not None:
            sc.external.pp.bbknn(adata, batch_key=batch_label, use_rep='X_pca', n_pcs=50, neighbors_within_batch=knn, metric='euclidean', trim=min(trim, knn*len(adata.obs[batch_label].unique())))
        else:
            sc.external.pp.bbknn(adata, batch_key=batch_label, use_rep='X_pca', n_pcs=50, neighbors_within_batch=knn, metric='euclidean', trim=None)
        
        self.nns = defaultdict(list)
        """tmp = adata.obsp['connectivities'].nonzero()
        for i, j in zip(tmp[0], tmp[1]):
            i, j = int(i), int(j)
            if adata.obs_names[i] == adata.obs_names[j]:
                # handle the case of self in neighbors?
                pass
            if j not in self.nns[i]:
                self.nns[i].append(j)
        del tmp"""
        ns = sc.Neighbors(adata)
        print("Converting to igraph.")
        g = ns.to_igraph()
        tuple_list = g.to_tuple_list()
        anchors = pd.DataFrame(tuple_list, columns=['cell1', 'cell2'])
        pairs = np.array(anchors[['cell1', 'cell2']])
        print(f'Number of Training-Samples: {len(anchors)}')
        for i, j in pairs:
            i, j = int(i), int(j)
            if adata.obs_names[i] == adata.obs_names[j]:
                # handle the case of self in neighbors?
                pass
            self.nns[i].append(j)
        tmp = self.nns.keys()
        for i in range(len(adata)):
            if i not in tmp:
                self.nns[i].append(i)