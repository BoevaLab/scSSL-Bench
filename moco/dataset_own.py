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
from scipy import sparse

from itertools import combinations

# from geosketch import gs
from os.path import join
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture

from collections import defaultdict
from moco.prepare_dataset import prepare_dataset
from moco.preprocessing_own import preprocess_dataset, hvgPipe
from moco.config import Config
from moco.NNs import mnn_approx, nn_approx, reduce_dimensionality, random_walk1, computeMNNs
from moco.sNNs import generate_graph, computeAnchors

configs = Config()


class OWNscRNAMatrixInstance(Dataset):
    
    def __init__(self, data_bbknn, data_original, mode, p_original):
        self.datasets = [data_bbknn, data_original]
        self.p = p_original
        self.mode = mode
        #assert len(data_bbknn) == len(data_original), "Datasets in MultiDataset not matching!"
        self.n_samples = len(data_bbknn) + len(data_original)
        self.split_index = len(data_bbknn)

        self.label = None

    # Call this function after filtering
    def recalc_lengths(self):
        self.split_index = len(self.datasets[0])
        self.n_samples = len(self.datasets[0]) + len(self.datasets[1])
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        #i_dataset = np.random.choice([0,1])
        if self.mode=='train':
            if i < self.split_index:
                o = self.datasets[0].getTrainItem(i)
                return o[0], i, o[3] 
            else:
                o = self.datasets[1].getTrainItem(i-self.split_index)
                return o[0], i, o[3]
        else:
            return self.datasets[0].getValItem(i)

class MultiDataset(Dataset):

    def __init__(self, data_bbknn, data_original, mode, p_original):
        self.datasets = [data_bbknn, data_original]
        self.p = p_original
        self.mode = mode
        #assert len(data_bbknn) == len(data_original), "Datasets in MultiDataset not matching!"
        self.n_samples = len(data_bbknn) + len(data_original)
        self.split_index = len(data_bbknn)

    # Call this function after filtering
    def recalc_lengths(self):
        self.split_index = len(self.datasets[0])
        self.n_samples = len(self.datasets[0]) + len(self.datasets[1])
    
    def __len__(self):
        return self.n_samples

    def __getitem__(self, i):
        #i_dataset = np.random.choice([0,1])
        if self.mode=='train':
            if i < self.split_index:
                return self.datasets[0].getTrainItem(i)
            else:
                return self.datasets[1].getTrainItem(i-self.split_index)
        else:
            return self.datasets[0].getValItem(i)

    
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
            verbose=0,
            k_anchor=5,
            filtering=True, 
            bbknn_exchange=False,
            bbknn_fair=False,
            trimming=False,
            trim_val=50,
            holdout=[],# [(Batch, Celltype)] which can contain multiple holdouts
        ):
        self.mode = mode
        self.scale = scale
        self.verbose = verbose
        self.data_dir = data_dir  # data_root/dataset_name
        self.select_hvg = select_hvg
        self.augment_op_names = augment_set

        self.bbknn_mode = bbknn_exchange
        self.alpha = alpha

        self.trimming = trimming
        self.trim_val = trim_val
        self.bbknn_fair = bbknn_fair

        # self.reset_anchors = reset_anchors
        # self.anchor_metadata = anchor_metadata

        if len(holdout) > 0:
            holdout = self.parse_holdout(holdout)
            print(f"Loading with holdout set {holdout}")
            self.load_data(holdout)
        else:
            self.load_data()     # load data first to get self.cname
        
        # DEV-INFO: Dictionary to get gene-name->index mapping for cna_naive transform.
        self.index_transformer = {}
        
        # define set of augment operatio
        self.augment_set = []
        for ai in augment_set:
            if ai=='int':
                self.augment_set.append(partial(interpolation, alpha=alpha))
            elif ai=='geo':
                self.augment_set.append(partial(geo_interpolation, alpha=alpha))
            elif ai=='exc':
                #self.augment_set.append(partial(binary_switch_mv, alpha=alpha))
                self.augment_set.append(partial(binary_switch, alpha=alpha))
            elif ai=='cna-n':
                gene_indices = {}
                signature_df = pd.read_csv("../Wang_cancer_genomic-co-localization.csv")
                num_signatures = 101
                gene_names = pd.read_csv("../Wang_unique_gene_symbols.csv")["Gene symbol"]
                for gene_name in gene_names:
                    k = np.argwhere(self.gname == gene_name)
                    gene_indices[gene_name] = k
                self.augment_set.append(partial(cna_naive, alpha=alpha, gene_indices=gene_indices, signature_df=signature_df,
                                                num_signatures=num_signatures))
            elif ai=='tacna-tcga':
                gene_indices = {}
                signature_df = pd.read_csv("../TACNA_TCGA.csv")
                signature_df = signature_df.iloc[:8500]
                # get the genes which are in the dataset after processing.
                signature_df = signature_df.set_index(signature_df['Gene symbol'])
                signature_df = signature_df[signature_df['Gene symbol'].isin(signature_df.index.intersection(self.gname))]
                gene_names = signature_df['Gene symbol']
                for gene_name in gene_names:
                    k = np.argwhere(self.gname == gene_name)
                    gene_indices[gene_name] = k
                self.augment_set.append(partial(tacna_naive, alpha=alpha, gene_indices=gene_indices, signature_df=signature_df))
            else:
                raise ValueError("Unrecognized augment operation")
            """elif ai=='tacna-geo':
                gene_indices = {}
                signature_df = pd.read_csv("../TACNA_GEO.csv")
                num_signatures = 101
                gene_names = pd.read_csv("../Wang_unique_gene_symbols.csv")["Gene symbol"]
                for gene_name in gene_names:
                    k = np.argwhere(self.gname == gene_name)
                    gene_indices[gene_name] = k
                self.augment_set.append(partial(cna_naive, alpha=alpha, gene_indices=gene_indices, signature_df=signature_df,
                                                num_signatures=num_signatures))"""
        if self.verbose:
            print('Defined ops: ', self.augment_op_names)

        # define intra-batch knn
        if mode=='train' and bbknn_exchange:
            """
            Compute the anchors with bbknn instead of MNN+kNN
            """
            # 1. computing anchors
            k_anchor = k_anchor
            self.compute_anchors_bbknn(self.adata, self.X, configs.batch_key, self.cname, self.gname, k_anchor=k_anchor, filtering=filtering)

            # 2. use the anchors exported from seurat
            #self.load_anchors(anchor_path)
            self.getMnnDict()
            self.exclude_sampleWithoutMNN(exclude_fn)
            # TODO: Remove??
            self.computeKNN(knn)
            # DEV-INFO: returns pd DataFrame which gives the sampling probabilities.
            self.probas = self.compute_batch_probabilities(self.adata, key="X_umap")
        elif mode=='train':
            """
            This is the original CLAIRE pipeline.
            """
            # 1. computing anchors
            k_anchor = k_anchor
            self.compute_anchors(self.X, self.batch_label, self.cname, self.gname, k_anchor=k_anchor, filtering=filtering)

            # 2. use the anchors exported from seurat
            #self.load_anchors(anchor_path)
            self.getMnnDict()
            self.exclude_sampleWithoutMNN(exclude_fn)
            self.computeKNN(knn)
            # DEV-INFO: returns pd DataFrame which gives the sampling probabilities.
            self.probas = self.compute_batch_probabilities(self.adata, key="X_umap")
        elif mode =='train_other':
            self.mode='train'

    def __len__(self):
        if self.mode=='train':
            return len(self.valid_cellidx)
        else:
            return self.X.shape[0]
    
    # DEV-INFO: TESTING THIS FUNCTION
    def update_pos_nn_info(self, knn):
        print("WARNING! Update pos_nn_info has been added to test its impact on the non-optim-dataset.")
        # create positive sample index
        rand_ind1 = np.random.randint(0, knn, size=(self.n_sample))
        rand_nn_ind1 = self.nns[np.arange(self.n_sample), rand_ind1]
        rand_ind2 = np.random.randint(0, knn, size=(self.n_sample))
        # rand_nn_ind2 = self.nns[np.arange(self.n_sample), rand_ind2]

        self.lambdas1 = np.random.uniform(self.alpha, 1, size=(self.n_sample, 1))
        self.lambdas2 = np.random.uniform(self.alpha, 1, size=(self.n_sample, 1))

        self.rand_pos_ind = [np.random.choice(self.mnn_dict[i]) if len(self.mnn_dict[i])>0 else i for i in range(self.n_sample)]

        X_arr = self.X.A
        X_pos = X_arr[self.rand_pos_ind]
        pos_knns_ind = self.nns[self.rand_pos_ind]
        pos_nn_ind = pos_knns_ind[np.arange(self.n_sample), rand_ind2]
        self.X1 = X_arr*self.lambdas1 + X_arr[rand_nn_ind1]*(1-self.lambdas1)
        self.X2 = X_pos*self.lambdas2 + X_arr[pos_nn_ind] * (1-self.lambdas2)

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

        # Don't print anchor info. Resource consumption for large datasets is too much.
        """if self.verbose and (self.type_label is not None):
            print_AnchorInfo(self.pairs, self.batch_label, self.type_label)"""
        
    def compute_batch_probabilities(self, ad, key):
        sc.pp.neighbors(ad)
        sc.tl.umap(ad) # X_umap stored in ad.obsm['X_umap']
        sc.tl.pca(ad, n_comps=12)
        # Generate separate matrices
        batches = list(ad.obs['batchlb'].unique())

        # Calculate distances between batches

        combos = list(combinations(batches, r=2))
        distances = pd.DataFrame(np.zeros((len(batches), len(batches))), index=batches, columns=batches)

        for batch_a, batch_b in combos:
            distances.loc[batch_a][batch_b] = distances.loc[batch_b][batch_a] = float(torch.median(torch.cdist(torch.from_numpy(ad[ad.obs['batchlb']==batch_a].obsm[key]).unsqueeze(0), torch.from_numpy(ad[ad.obs['batchlb']==batch_b].obsm[key]).unsqueeze(0))))
            
        for batch in batches:
            distances.loc[batch][batch] = float(torch.median(torch.cdist(torch.from_numpy(ad[ad.obs['batchlb']==batch].obsm[key]).unsqueeze(0), torch.from_numpy(ad[ad.obs['batchlb']==batch].obsm[key]).unsqueeze(0))))

        s = torch.nn.Softmax(dim=1)
        distances = distances-distances.min(axis=1)
        np.fill_diagonal(distances.values, 0)
        self.probas = pd.DataFrame(np.array(s(torch.from_numpy(np.array(distances)))), index=distances.index, columns=distances.columns)
        return self.probas
    
    def recompute_batch_probabilities(self, ad, key):
        """if key == 'X_emb' and 'X_emb' in ad.obsm.columns:
            pass
        el"""
        if key == 'X_emb': # TODO: fix to fit comment above.
            ad.obsm['X_emb'] = ad.X
        elif key == 'X_umap':
            sc.pp.neighbors(ad)
            sc.tl.umap(ad) # X_umap stored in ad.obsm['X_umap']
        elif key == 'X_pca':
            sc.tl.pca(ad, n_comps=12)
        else:
            print("Recomputing cancelled. Something went wrong. Resorting to old probabilities.")
        
        # Generate separate matrices
        batches = list(ad.obs['batchlb'].unique())

        # Calculate distances between batches

        combos = list(combinations(batches, r=2))
        distances = pd.DataFrame(np.zeros((len(batches), len(batches))), index=batches, columns=batches)

        for batch_a, batch_b in combos:
            distances.loc[batch_a][batch_b] = distances.loc[batch_b][batch_a] = float(torch.median(torch.cdist(torch.from_numpy(ad[ad.obs['batchlb']==batch_a].obsm[key]).unsqueeze(0), torch.from_numpy(ad[ad.obs['batchlb']==batch_b].obsm[key]).unsqueeze(0))))
            
        for batch in batches:
            distances.loc[batch][batch] = float(torch.median(torch.cdist(torch.from_numpy(ad[ad.obs['batchlb']==batch].obsm[key]).unsqueeze(0), torch.from_numpy(ad[ad.obs['batchlb']==batch].obsm[key]).unsqueeze(0))))

        s = torch.nn.Softmax(dim=1)
        distances = distances-distances.min(axis=1)
        np.fill_diagonal(distances.values, 0)
        self.probas = pd.DataFrame(np.array(s(torch.from_numpy(np.array(distances)))), index=distances.index, columns=distances.columns)

    def compute_anchors(self, X, batch_label, cname, gname, k_anchor=5, filtering=True):
        print('computing anchors')
        anchors = computeAnchors(X, batch_label, cname, gname, k_anchor=k_anchor, filtering=filtering)
        print(f'Anchors-length: {len(anchors)}')
        anchors.cell1 = anchors.cell1_name.apply(lambda x: self.name2idx[x])
        anchors.cell2 = anchors.cell2_name.apply(lambda x: self.name2idx[x])
        pairs = np.array(anchors[['cell1', 'cell2']])
        print(f'Pairs-length: {len(anchors)}')

        self.pairs = pairs
        anchors.to_csv(join(self.data_dir, 'seuratAnchors.csv'))

        # print anchor info
        if self.verbose and (self.type_label is not None):
            print_AnchorInfo(self.pairs, self.batch_label, self.type_label)


    def compute_anchors_bbknn(self, adata, X, batch_label, cname, gname, k_anchor=5, filtering=True):
            print('computing bbknn graph')
            sc.tl.pca(data=adata,n_comps=50)
            # TODO: Set the following parameters? [trim, metric]
            if self.trimming:
                sc.external.pp.bbknn(adata, batch_key=batch_label, use_rep='X_pca', n_pcs=50, neighbors_within_batch=k_anchor, metric='euclidean', trim=min(self.trim_val, k_anchor*len(adata.obs[batch_label].unique())))#None)#trim=int(1.5*k_anchor*len(adata.obs[batch_label].unique())))#trim=int(0.8*k_anchor*len(adata.obs[batch_label].unique())))
            else:
                sc.external.pp.bbknn(adata, batch_key=batch_label, use_rep='X_pca', n_pcs=50, neighbors_within_batch=k_anchor, metric='euclidean', trim=None)
            
            
            """
            DEV-INFO:   This is the code to remove neighbors of the same batch when using bbknn.
                        All edges to cells within the same batch are removed.
            """
            if self.bbknn_fair:
                tmp = adata.obsp['connectivities'].nonzero()
                conns = adata.obsp['connectivities']
                dists = adata.obsp['distances']

                for i, j in zip(tmp[0], tmp[1]):
                    if adata.obs_names[i] == adata.obs_names[j]:
                        conns[i][j] = 0.0
                        dists[i][j] = 0.0

                adata.obsp['connectivities'] = sparse.csr_matrix(conns)
                adata.obsp['distances'] = sparse.csr_matrix(dists)
            
            """
            The goal below is to get (cell, neighbor) for all neighborhoods and cells.
            """
            print("Getting neighbors")
            ns = sc.Neighbors(adata)
            print("Converting to igraph.")
            g = ns.to_igraph()
            
            print(f"The bbknn graph contains {ns.n_neighbors} neighbors per cell (n_batches x n_anchors)")
            print('get anchors...')
            
            tuple_list = g.to_tuple_list()

            anchors = pd.DataFrame(tuple_list, columns=['cell1', 'cell2'])

            # anchors = computeAnchors(X, batch_label, cname, gname, k_anchor=k_anchor, filtering=filtering)
            print(f'Anchors-length: {len(anchors)}')
            """
            TODO: The code below is likely not required. (??)
            anchors.cell1 = anchors.cell1_name.apply(lambda x: self.name2idx[x])
            anchors.cell2 = anchors.cell2_name.apply(lambda x: self.name2idx[x])
            print(f'Pairs-length: {len(anchors)}')
            """
            pairs = np.array(anchors[['cell1', 'cell2']])

            self.pairs = pairs
            anchors.to_csv(join(self.data_dir, 'seuratAnchors.csv'))

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
    
    """
    DEV-INFO
    TODO: Reference mapping only works on a single-add-basis with this. Need to enable more "B1,*;B3,*;..." at the end.
    """
    def parse_holdout(self, holdout, adata):
        if len(holdout) > 1:
            return holdout
        elif len(holdout) == 1:
            if holdout[0][1]!= "*":
                return holdout
            batchname = holdout[0][0]
            celltypes = adata.obs["CellType"][adata.obs["batchlb"]==batchname].unique()
            holdout = [(batchname, celltype) for celltype in celltypes]
            return holdout

    def load_data(self, holdout_set=[]):
        # customized
        sps_x, genes, cells, metadata = prepare_dataset(self.data_dir)
        if type(metadata) == list and len(metadata) == 3:
            metadata, X_cnv, cnv_mapping = metadata
        adata, X, cell_name, gene_name, metadata = preprocess_dataset(
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
        self.adata = sc.AnnData(X)
        self.adata.layers['counts'] = adata.layers['counts']
        
        self.adata.var_names = gene_name
        self.adata.obs = metadata.copy()
        self.adata.uns = adata.uns
        self.adata.var = adata.var
        
        """
        DEV-INFO: The code below was added to allow testing on holdout data
        """
        if self.mode == "train" and len(holdout_set) > 0: # THE MODEL IS 
            holdout_set = self.parse_holdout(holdout_set)
            print(f"{self.X.shape[0]} cells before holdout.")
            for batch, celltype in holdout_set:
                print([batch, celltype])
                self.adata = self.adata[~((self.adata.obs['CellType'] == celltype) & (self.adata.obs['batchlb'] == batch))]
            """self.cname = self.adata.obs_names
            self.gname = self.adata.var_names"""
            self.X = self.adata.X
            df_meta = self.adata.obs.copy()
            df_meta[[configs.batch_key, configs.label_key]] = df_meta[[configs.batch_key, configs.label_key]].astype('category')
            self.metadata = df_meta
            print(f"{self.X.shape[0]} cells after holdout-filtering.")
        elif self.mode == "holdout" and len(holdout_set) > 0:
            holdout_set = self.parse_holdout(holdout_set)
            first = True
            holdout_adata = None
            for batch, celltype in holdout_set:
                if not first:
                    holdout_adata = holdout_adata.concatenate(self.adata[((self.adata.obs['CellType'] == celltype) & (self.adata.obs['batchlb'] == batch))])
                else:
                    holdout_adata = self.adata[((self.adata.obs['CellType'] == celltype) & (self.adata.obs['batchlb'] == batch))]
                    first = False
            if first == False:
                # Can only be false if we have something in the holdout set
                self.adata = holdout_adata
                """self.cname = self.adata.obs_names
                self.gname = self.adata.var_names"""
                self.X = self.adata.X
                df_meta = self.adata.obs.copy()
                df_meta[[configs.batch_key, configs.label_key]] = df_meta[[configs.batch_key, configs.label_key]].astype('category')
                self.metadata = df_meta
        
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

    def getTrainItem(self, i):
        i = self.valid_cellidx[i] if self.mode=='train' else i # translate local cell idx to global cell idx
        x = self.X[i].A.squeeze()

        batch = self.label_encoder[self.batch_label[i]]
        
        if self.type_label is not None:
            celltype = self.clabel_encoder[self.type_label[i]]
        else:
            celltype = None
        #print(batch)

        # all the story starts from mnn
        pos_anchors = self.mnn_dict[i]

        probas = self.probas.loc[self.batch_label[i]][self.batch_label[pos_anchors]].values
        """print(f"Pos. anchors are from batches {self.batch_label[pos_anchors]}")
        print(f"Probabilities: {probas}")"""

        # augment self <-- in this code, this is not the case...!!!
        # TODO: When using bbknn approach, don't need positive knn pairs from own batch.!! TODO TODO TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """if self.bbknn_mode:
            self.nns = [[] for i in range(self.n_sample)]"""
        
        # TODO: is i always 0 now???? Or only when bbknn-fair??? Or what??
        #ni = [i] if len(self.nns[i]) == 0 else np.random.choice(self.nns[i], size=1)
        #DEV-INFO: Remove any information of cells within batch! This can be adapted with some random augmentations to give two views and not just one!
        ni = [i]
        x_n = self.X[ni].A#.squeeze()
        x_aug = augment_positive(self.augment_set, x, x_n).squeeze()  # augment x with knn

        # augmenting one of x's positives
        pi = np.random.choice(pos_anchors, p=torch.nn.Softmax(dim=0)(torch.tensor(probas))) if len(pos_anchors)>0 else i
        """
        probas = self.probas.loc[self.batch_label[i]][self.batch_label[self.nns[pi]]]
        print(self.batch_label[self.nns[pi]])
        #### !!!! USE mnn_dict, not self.nns!!
        """
        x_p = self.X[pi].A.squeeze()
        #p_ni = pi if len(self.nns[pi])==0 else np.random.choice(self.nns[pi])
        
        p_ni = np.random.choice(self.nns[pi], size=1, replace=False) if len(self.nns[pi]) > 0 else [pi]
        #print(f"Neighbor batches: {list(self.batch_label[p_ni])}")
        x_p_ni = self.X[p_ni].A#.squeeze()
        x_p_aug = augment_positive(self.augment_set, x_p, x_p_ni).squeeze()

        # ADDED information about the origin-batch.
        return [x_aug.astype('float32'), x_p_aug.astype('float32')], [i, pi]#, batch, celltype

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

def interpolation_mv(x, x_p, alpha):
    lamda = np.random.uniform(alpha, 1.)  # [alpha, 1.]
    factor = (1-lamda)/len(x_p)
    x = lamda * x
    for k in range(len(x_p)):
        x+= factor * x_p[k]
    return x

def binary_switch_mv(x, x_p, alpha):
    n = len(x)
    indices = np.random.choice(n, size=(len(x_p), int(n*(1-alpha))//len(x_p)), replace=False)
    p_orig = np.ones(n)
    p_other = np.zeros(n)
    other = np.zeros(n)

    x_new = np.zeros(n)
    for k in range(len(x_p)):
        x_new[indices[k]] = x_p[k,indices[k]]
        x[indices[k]] = 0
    x_new = x + x_new
    """for k in range(len(x_p)):
        p_other[indices[k]] = 1
        other += x_p[k] * p_other
        p_orig -= p_other
        p_other.fill(0)
    x *= p_orig"""
    return x_new #x + other

def cna_naive(x, x_p, alpha, gene_indices, signature_df, num_signatures):
    #np.random.randint(2, size=num_signatures)
    # choose random signature
    i = np.random.randint(1, num_signatures+1)
    # choose which genes to change
    sig = signature_df[f"Gene.{i}"]
    bernou_p = bernoulli.rvs(alpha, size=len(sig))
    factor = np.random.normal(loc=1.0, scale=0.2)
    p = bernou_p * factor
    change = np.ones(len(x))
    for gname, j in zip(sig, range(len(sig))):
        change[gene_indices[gname]] = p[j]
    #x = (x + np.random.randint(2)) * bernou_p + x_p * (1-bernou_p)
    lamda = np.random.uniform(alpha, 1.)  # [alpha, 1.]
    x = lamda * x * change + (1 - lamda) * x_p * change
    return x

def tacna_naive(x, x_p, alpha, gene_indices, signature_df):
    #np.random.randint(2, size=num_signatures)
    # choose the strength of the augmentations
    factor = np.random.normal(loc=1.2, scale=0.2)
    # choose which genes to augment
    bernou_p = bernoulli.rvs(alpha, size=len(signature_df))
    # TODO: get the current cnv and then amplify the directionality of the CNA
    degree = signature_df["Degree of TACNA"]
    p = bernou_p * factor * degree
    change = np.ones(len(x))
    for gname, j in zip(list(signature_df['Gene symbol']), range(len(signature_df))):
        change[gene_indices[gname]] = p[j]
    #x = (x + np.random.randint(2)) * bernou_p + x_p * (1-bernou_p)
    lamda = np.random.uniform(alpha, 1.)  # [alpha, 1.]
    x = lamda * x * change + (1 - lamda) * x_p * change
    return x


"""
def build_mask(masked_percentage, gene_num):
    mask = np.concatenate([np.ones(int(gene_num * masked_percentage), dtype=bool), 
                           np.zeros(gene_num - int(gene_num * masked_percentage), dtype=bool)])
    np.random.shuffle(mask)
    return mask

def RandomCrop(cell_profile, crop_percentage=0.8):
    mask = build_mask(crop_percentage)
    cell_profile = cell_profile[mask]
    gene_num = len(cell_profile)
    dataset = self.dataset[:,mask]

def random_mask(self, 
                    mask_percentage: float = 0.15, 
                    apply_mask_prob: float = 0.5):

        s = np.random.uniform(0,1)
        if s<apply_mask_prob:
            # create the mask for mutation
            mask = self.build_mask(mask_percentage)
            
            # do the mutation with prob
        
            self.cell_profile[mask] = 0



def random_gaussian_noise(self, 
                              noise_percentage: float=0.2, 
                              sigma: float=0.5, 
                              apply_noise_prob: float=0.3):

        s = np.random.uniform(0,1)
        if s<apply_noise_prob:
            # create the mask for mutation
            mask = self.build_mask(noise_percentage)
            
            # create the noise
            noise = np.random.normal(0, 0.5, int(self.gene_num*noise_percentage))
            
            # do the mutation (maybe not add, simply change the value?)
            self.cell_profile[mask] += noise


def random_swap(self,
                    swap_percentage: float=0.1,
                    apply_swap_prob: float=0.5):

        ##### for debug
        #     from copy import deepcopy
        #     before_swap = deepcopy(cell_profile)
        s = np.random.uniform(0,1)
        if s<apply_swap_prob:
            # create the number of pairs for swapping 
            swap_instances = int(self.gene_num*swap_percentage/2)
            swap_pair = np.random.randint(self.gene_num, size=(swap_instances,2))
            
            # do the inner crossover with p
        
            self.cell_profile[swap_pair[:,0]], self.cell_profile[swap_pair[:,1]] = \
                self.cell_profile[swap_pair[:,1]], self.cell_profile[swap_pair[:, 0]]

def instance_crossover(self,
                           cross_percentage: float=0.25,
                           apply_cross_prob: float=0.4):
        
        # it's better to choose a similar profile to crossover
        
        s = np.random.uniform(0,1)
        if s<apply_cross_prob:
            # choose one instance for crossover
            cross_idx = np.random.randint(self.cell_num)
            cross_instance = self.dataset[cross_idx]
            
            # build the mask
            mask = self.build_mask(cross_percentage)
            
            # apply instance crossover with p
            
            tmp = cross_instance[mask].copy()
        
            cross_instance[mask], self.cell_profile[mask]  = self.cell_profile[mask], tmp
"""

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
