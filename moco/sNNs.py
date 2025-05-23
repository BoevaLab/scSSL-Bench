import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_decomposition import PLSCanonical
import random
from sklearn.neighbors import KDTree
from annoy import AnnoyIndex
from geosketch import gs
import scipy.sparse as sps
from itertools import product
from moco.NNs import reduce_dimensionality

# 把Seurat的FindIntegrationAnchors 翻译了一遍，几乎没有改动
SCALE_BEFORE = True
EPS = 1e-12
VERBOSE = False

def svd1(mat, num_cc):
    U, s, V = np.linalg.svd(mat)
    d = s[0:int(num_cc)]
    u = U[:, 0:int(num_cc)]
    v = V[0:int(num_cc), :].transpose()
    return u, v, d

def pls(x, y, num_cc):
    random.seed(42)
    plsca = PLSCanonical(n_components=int(num_cc), algorithm='svd')
    fit = plsca.fit(x, y)
    u = fit.x_weights_
    v = fit.y_weights_
    a1 = np.matmul(np.matrix(x), np.matrix(u)).transpose()
    d = np.matmul(np.matmul(a1, np.matrix(y)), np.matrix(v))
    ds = [d[i, i] for i in range(0, 30)]
    return u, v, ds


#' scale each features of X, axis=0
def scale2(x):
    # y = preprocessing.scale(x)  # scale each col separately
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    y = (x - x_mean) / (x_std + EPS)

    return y

#' @param num.cc Number of canonical vectors to calculate
#' @param seed.use Random seed to set.
#' @importFrom svd1
def runcca(data1, data2, num_cc=20, scale_before=SCALE_BEFORE):
    random.seed(42)

    object1 = scale2(data1) if scale_before else data1.copy()
    object2 = scale2(data2) if scale_before else data2.copy()

    mat3 = object1.T.dot(object2)
    a = svd1(mat=mat3, num_cc=num_cc)
    cca_data = np.concatenate((a[0], a[1]))
    ind = np.where(
        [cca_data[:, col][0] < 0 for col in range(cca_data.shape[1])])[0]
    cca_data[:, ind] = cca_data[:, ind] * (-1)

    d = a[2]
    #' d = np.around(a[2], 3)  #.astype('int')
    return cca_data, d


def l2norm(mat):
    stats = np.sqrt(np.sum(mat**2, axis=1, keepdims=True)) + EPS
    mat = mat / stats
    return mat

#' @param data_use1 pandas data frame
#' @param data_use2 pandas data frame
#' @rdname runCCA
#' @export feature loadings and embeddings
def runCCA(data_use1, data_use2, features, num_cc):
    features_idx = np.arange(len(features))
    features_idx = checkFeature(data_use1, features_idx)   # filter features (genes) with var=0 
    features_idx = checkFeature(data_use2, features_idx)
    # features = features[features_idx]
    if VERBOSE:
        print(f'====>{len(features_idx)} left ')

    data1 = data_use1[features_idx, ]  # genes * cells
    data2 = data_use2[features_idx, ]
    cca_results = runcca(data1=data1, data2=data2, num_cc=num_cc)
    cell_embeddings = cca_results[0]

    combined_data = np.hstack([data1, data2])
    loadings = combined_data.dot(cell_embeddings)
    return cca_results, loadings, features_idx


# Check if features have zero variance
# data_use: array, n_genes * n_cells
# features_idx, array, gene idx
def checkFeature(data_use, features_idx):
    gene_var = data_use[features_idx, ].std(axis=1)
    features_idx = features_idx[gene_var != 0]
    return features_idx


#' @param data Input data
#' @param query Data to query against data
#' @param k Number of nearest neighbors to compute
# Approximate nearest neighbors using locality sensitive hashing.
def NN(data, query=None, k=10, metric='manhattan', n_trees=10):
    if query is None:
        query = data

    # Build index.
    a = AnnoyIndex(data.shape[1], metric=metric)
    for i in range(data.shape[0]):
        a.add_item(i, data[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(query.shape[0]):
        ind.append(a.get_nns_by_vector(query[i, :], k, search_k=-1))
    ind = np.array(ind)

    return ind
    

#' @param cell_embedding : pandas data frame
def findNN(data1, data2, k):
    if VERBOSE:
        print("Finding nearest neighborhoods")
    # nnaa = NN(data1, k=k+1)
    # nnbb = NN(data2, k=k+1)
    nnab = NN(data=data2, query=data1, k=k)
    nnba = NN(data=data1, query=data2, k=k)
    return nnab, nnba

# def NN(data, k, query=None):
#     tree = KDTree(data)
#     if query is None:
#         query = data
#     dist, ind = tree.query(query, k)
#     return dist, ind
# def findNN(cell_embedding, cells1, cells2, k):
    # print("Finding nearest neighborhoods")
    # embedding_cells1 = cell_embedding.loc[cells1, ]
    # embedding_cells2 = cell_embedding.loc[cells2, ]
    # nnaa = NN(embedding_cells1, k=k + 1)
    # nnbb = NN(embedding_cells2, k=k + 1)
    # nnab = NN(data=embedding_cells2, k=k, query=embedding_cells1)
    # nnba = NN(data=embedding_cells1, k=k, query=embedding_cells2)
    # return nnaa, nnab, nnba, nnbb, cells1, cells2



def findMNN(neighbors, num):
    max_nn = np.array([neighbors[0].shape[1], neighbors[1].shape[1]])
    if ((num > max_nn).any()):
        num = np.min(max_nn)
        # convert cell name to neighbor index
    
    nnab, nnba = neighbors[0], neighbors[1]

    pairs_ab, pairs_ba = set(), set()
    # build set of mnn of (b1, b2)
    for i,nni in enumerate(nnab):
        nni = nni[:num]  # take the top num neighbors
        for j in nni:
            pairs_ab.add((i, j))
    # build set of mnn of (b2, b1), -> (b1, b2)
    for i, nni in enumerate(nnba):
        nni = nni[:num]
        for j in nni:
            pairs_ba.add((j, i))

    pairs = pairs_ab & pairs_ba
    pairs = np.array([[p[0], p[1]] for p in pairs])
    
    mnns = pd.DataFrame(pairs, columns=['cell1', 'cell2'])
    if VERBOSE:
        print(f'\t Found {mnns.shape[0]} mnn pairs')
    return mnns


#' @param dim Dimension to use
#' @param numG Number of genes to return
#' @return Returns a vector of top genes
def topGenes(Loadings, dim, numG):                # 取与cca.dim正相关和负相关的前numG//2个gene
    data = Loadings[:, dim]
    num = numG//2

    gene_sort = np.argsort(data)
    neg_idx = gene_sort[:num]
    pos_idx = gene_sort[-num:]

    topG_idx = np.concatenate((pos_idx, neg_idx))
    return topG_idx


#' Get top genes across different dimensions
#' dims: array, range(num_cc)
#' dimGenes: int, max dimension of correlated genes
#  maxGenes: int, max number of selected genes per cc dim

def TopGenes(Loadings, dims, DimGenes, maxGenes):
    maxG = max(len(dims) * 2, maxGenes)
    gens = [None] * DimGenes
    idx = -1
    for i in range(1, DimGenes + 1):
        idx = idx + 1
        selg = []
        for j in dims:
            selg.extend(set(topGenes(Loadings, dim=j, numG=i)))
        gens[idx] = set(selg)
    lens = np.array([len(i) for i in gens])
    lens = lens[lens < maxG]
    maxPer = np.where(lens == np.max(lens))[0][0] + 1 # woc, 还有这种傻狗代码，另外，非要+1干嘛
    selg = []
    for j in dims:
        selg.extend(set(topGenes(Loadings, dim=j, numG=maxPer)))
    selgene = np.array(list(set(selg)), dtype='int64')

    if VERBOSE:
        print(f'======> Top {len(selgene)} genes selected to filter anchors')

    return (selgene)

# pairs: dataframe={'cell1', 'cell2'}, mnns
# data1, data2: array, normalized data
# features: array, top gene idx
def filterPair(pairs, data1, data2, features, k_filter):
    top_data1 = data1[features, ].T
    top_data2 = data2[features, ].T

    cn_data1 = l2norm(top_data1)
    cn_data2 = l2norm(top_data2)

    # print('===========> IsNa', cn_data1.isna().sum().sum(), cn_data2.isna().sum().sum())
    nn = NN(data=cn_data2,
            query=cn_data1,
            k=k_filter)
    position = [
        (pairs["cell2"][x] in nn[pairs['cell1'][x]])  # mnn=(a,b), np.where(b == nnab[a]) # 谁tm能告诉我这是什么勾吧东西
        for x in range(pairs.shape[0])                                          # 这样导致，position最后的数目确实和目标一致，但是pos的值<=k_filter
    ]
    # print(position)
    # nps = np.concatenate(position, axis=0)
    fpair = pairs.loc[position, ].copy()

    if VERBOSE:
        print("\t Retained ", fpair.shape[0], " MNN pairs")
    return (fpair)

# norm_list: [np.array] * N, list of normalized data
# features: array, highly variable features, array
# cname_list: [np.array] * N, list of cell names of each batch
# num_cc:  int,  dim of cca
# k_filter: int, knn for filtering
# k_neighbors: int, knn for calculating knn
def generate_graph(norm_list, cname_list, features, combine, num_cc=30, k_filter=200, k_neighbor=5, filtering=True):
    all_pairs = []
    # embeddings = []
    for row in combine:
        i = row[0]
        j = row[1]
        norm_data1 = norm_list[i]
        norm_data2 = norm_list[j]

        cell_embedding, loading, features_filtered = runCCA(data_use1=norm_data1,    # scale and runcca
                                         data_use2=norm_data2,
                                         features=features,
                                         num_cc=num_cc)
        norm_embedding = l2norm(mat=cell_embedding[0])
        # embeddings.append(norm_embedding)
        #' identify nearest neighbor
        cells1 = cname_list[i]
        cells2 = cname_list[j]
        neighbor = findNN(
                    data1 = norm_embedding[:len(cells1)],
                    data2 = norm_embedding[len(cells1):],
                    k=30)
        #' identify mutual nearest neighbors
        #' @param neighbors,colnames
        #' @export mnn_pairs
        mnn_pairs = findMNN(neighbors=neighbor,
                            num=k_neighbor)
        # Mat = pd.concat([norm_data1, norm_data2], axis=1)
        if filtering:
            select_genes = TopGenes(Loadings=loading,
                                dims=range(num_cc),
                                DimGenes=100,
                                maxGenes=200)
            final_pairs = filterPair(pairs=mnn_pairs,
                                     data1=norm_data1[features_filtered, ],  # take the filtered features
                                     data2=norm_data2[features_filtered, ],
                                     features=select_genes,             # idx within the filtered features
                                     k_filter=k_filter)
        else:
            final_pairs = mnn_pairs

        final_pairs['cell1_name'] = final_pairs.cell1.apply(lambda x: cells1[x])
        final_pairs['cell2_name'] = final_pairs.cell2.apply(lambda x: cells2[x])

        final_pairs['Dataset1'] = [i + 1] * final_pairs.shape[0]
        final_pairs['Dataset2'] = [j + 1] * final_pairs.shape[0]
        all_pairs.append(final_pairs)

    all_pairs = pd.concat(all_pairs)
    return all_pairs #, embeddings

'''
    X: array or csr_matrix, 
    batch_label: array,
    cname: array
    gname: array
    sketch: using geosketching to reduce number of cells
    k_anchor: number of K used to select mnn
'''
def computeAnchors(X, batch_label, cname, gname, k_anchor=5, filtering=True):
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

    return anchors


def sketching_per_batch(X, bl, rho):
    bl = np.array(bl)
    bs = np.unique(bl)
    n_batch = bs.size

    sketch_index, bi_count = [], []
    for bi in bs:
        bii = np.where(bl == bi)[0]
        X_bi = X[bii].A if sps.issparse(X) else X[bii].copy()
        bi_sketch_ind = gs(X_bi, int(rho*len(bii)), replace=False)

        sketch_index.extend(bii[bi_sketch_ind])  # local index to global index

    return sketch_index

def computeAnchors_sketch(X, batch_label, cname, gname, rho=.4, r=10, k_anchor=5, filtering=True):
    X = reduce_dimensionality(X, 50)

    anchors_queue = []
    # repeating sketching
    for i in range(r):
        sketch_index = sketching_per_batch(X, batch_label, rho)
        X_sketch = X[sketch_index]
        cname_sketch = cname[sketch_index]
        bl_sketch = batch_label[sketch_index]

        norm_list, cname_list = [], []
        bs = np.unique(bl_sketch)
        n_batch = bs.size

        for bi in bs:
            bii = np.where(bl_sketch==bi)[0]
            X_bi = X_sketch[bii].A if sps.issparse(X_sketch) else X_sketch[bii].copy()
            cname_bi = cname_sketch[bii]

            norm_list.append(X_bi.T)  # (gene*, cell)
            cname_list.append(np.array(cname_bi))

        combine = list(product(np.arange(n_batch), np.arange(n_batch)))
        combine = list(filter(lambda x: x[0]<x[1], combine))               # non symmetric 
        feats = np.array([f'pca{_+1}' for _ in range(X.shape[1])])
        anchors = generate_graph(norm_list, cname_list, feats, combine, 
                                    num_cc=20, k_filter=200, k_neighbor=k_anchor, filtering=filtering)

        anchors_queue.append(anchors)

    # anchors = pd.concat(anchors_queue)

    return anchors_queue


# import os
# from os.path import join
# from prepare_dataset import prepare_dataset
# from preprocessing import preprocess_dataset
# from config import Config
# from line_profiler import LineProfiler
# from itertools import product

# configs = Config()

# if __name__=="__main__":
#     data_dir = join(configs.data_root, 'MouseCellAtlas')
#     sps_x, genes, cells, metadata = prepare_dataset(data_dir)
#     X, cell_name, gene_name, metadata = preprocess_dataset(
#         sps_x, 
#         cells, 
#         genes, 
#         metadata, 
#         2000, 
#         False, 
#         )

#     norm_list, cname_list = [], []
#     batch_label = np.array(metadata[configs.batch_key])
#     bs = np.unique(batch_label)
#     n_batch = bs.size
#     for bi in bs:
#         bii = np.where(batch_label==bi)[0]
#         print(f'==> {bi}, n_sample={len(bii)}')

#         # id(sps_x.A) != id(sps_x), so i didn't make a copy here
#         norm_list.append(X[bii].A.T)
#         cname_list.append(cell_name[bii])

#     combine = list(product(np.arange(n_batch), np.arange(n_batch)))
#     combine = list(filter(lambda x: x[0]<x[1], combine))
#     anchors = generate_graph(norm_list, cname_list, gene_name, combine, 
#                                 num_cc=20, k_filter=200, k_neighbor=5)

#     lp = LineProfiler()
#     lp.add_function(runcca)
#     lp.add_function(runCCA)
#     lp_wrapper = lp(generate_graph)
#     lp_wrapper(norm_list, cname_list, gene_name, combine, 
#                                 20, 200, 5)
#     lp.print_stats()
