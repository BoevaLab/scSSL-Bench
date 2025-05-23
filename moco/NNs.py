from fbpca import pca
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse as sps

from annoy import AnnoyIndex
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances

KNN = 20
APPROX = True

def reduce_dimensionality(X, dim_red_k=100):
    k = min((dim_red_k, X.shape[0], X.shape[1]))
    U, s, Vt = pca(X, k=k) # Automatically centers.
    return U[:, range(k)] * s[range(k)]

# Exact nearest neighbors search.
def nn(ds1, ds2, knn=KNN, metric_p=2):
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(n_neighbors=knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((a, b_i))

    return match

# l2-norm
# def normalize(data: np.float32) -> np.float32:
#     norm = np.sqrt((data * data).sum(axis=1, keepdims=True))
#     dnorm = data / norm

#     return dnorm# / np.array([np.sqrt(np.sum(np.square(norm), axis=1))]).T

# Approximate nearest neighbors using locality sensitive hashing.
def nn_approx(ds1, ds2, norm=True, knn=KNN, metric='manhattan', n_trees=10):
    if norm:
        ds1 = normalize(ds1)
        ds2 = normalize(ds2) 

    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    return ind

# def nn_approx(ds1, ds2, knn=10, metric='manhattan', n_trees=10, ind_only=False):
#     # Build index.
#     a = AnnoyIndex(ds2.shape[1], metric=metric)
#     for i in range(ds2.shape[0]):
#         a.add_item(i, ds2[i, :])
#     a.build(n_trees)

#     # Search index.
#     ind = []
#     for i in range(ds1.shape[0]):
#         ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
#     ind = np.array(ind)

#     if ind_only:
#         return ind

#     # Match.
#     match = set()
#     for a, b in zip(range(ds1.shape[0]), ind):
#         for b_i in b:
#             match.add((a, b_i))

#     return match

def mnn(ds1, ds2, knn=10, approx=True):
    # Find nearest neighbors in first direction.
    if approx:
        match1 = nn_approx(ds1, ds2, knn=knn)
    else:
        match1 = nn(ds1, ds2, knn=knn)

    # Find nearest neighbors in second direction.
    if approx:
        match2 = nn_approx(ds2, ds1, knn=knn)
    else:
        match2 = nn(ds2, ds1, knn=knn)

    # Compute mutual nearest neighbors.
    mutual = match1 & set([ (b, a) for a, b in match2 ])

    return mutual

# def mnn_approx(X, Y, k=10, norm=False, metric='manhattan'):
#     if norm:
#         X = normalize(X)
#         Y = normalize(Y) 

#     f = X.shape[1]
#     t1 = AnnoyIndex(f, metric)
#     t2 = AnnoyIndex(f, metric)
#     for i in range(len(X)):
#         t1.add_item(i, X[i])
#     for i in range(len(Y)):
#         t2.add_item(i, Y[i])
#     t1.build(10)
#     t2.build(10)

#     # knn(x) in y
#     mnn_mat12 = np.zeros((len(X), len(Y)))
#     sorted_mat = np.array([t2.get_nns_by_vector(item, k) for item in X])
#     for i in range(len(sorted_mat)):
#         mnn_mat12[i,sorted_mat[i]] = 1
#     # knn(y) in x
#     mnn_mat21 = np.zeros((len(X), len(Y)))
#     sorted_mat = np.array([t1.get_nns_by_vector(item, k) for item in Y])
#     for i in range(len(sorted_mat)):
#         mnn_mat21[sorted_mat[i],i] = 1
#     # mnn
#     mnn_mat = (mnn_mat12 + mnn_mat21) >= 2
#     # mnn_mat = np.triu(mnn_mat, k=1)   # exclude self and making pairs are X -> Y
#     # pairs = np.vstack(np.where(mnn_mat)).T 
#     return mnn_mat

def mnn_approx(X, Y, norm=False, knn=10, metric='manhattan'):
    if norm:
        X = normalize(X)
        Y = normalize(Y)

    f = X.shape[1]
    t1 = AnnoyIndex(f, metric)
    t2 = AnnoyIndex(f, metric)
    for i in range(len(X)):
        t1.add_item(i, X[i])
    for i in range(len(Y)):
        t2.add_item(i, Y[i])
    t1.build(10)
    t2.build(10)

    mnn_mat = np.bool8(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t2.get_nns_by_vector(item, knn) for item in X])[:, 1:]
    for i in range(len(sorted_mat)):
        mnn_mat[i,sorted_mat[i]] = True
    _ = np.bool8(np.zeros((len(X), len(Y))))
    sorted_mat = np.array([t1.get_nns_by_vector(item, knn) for item in Y])[:, 1:]
    for i in range(len(sorted_mat)):
        _[sorted_mat[i],i] = True
    mnn_mat = np.logical_and(_, mnn_mat)
    pairs = np.array([[x, y] for x, y in zip(*np.where(mnn_mat>0))])
    return pairs

# get mnn pairs for every pair of datasets within a dataset list
# for quickly recomputing MNN pairs
# only support low-dim input here

### datasets: list, each ele is a array in low-dim
### uni_cnames: list, each ele is a array | global cell index, here
def computeMNNs(datasets, uni_cnames, knn, norm=False, metric='manhattan'):
    mnn_pairs = []
    n_ds = len(datasets)
    for i in range(n_ds):
        for j in range(i+1, n_ds):   # 我tm怎么会之前在这里写个range(i, n_ds) ??
            mnn_ij = mnn_approx(datasets[i], datasets[j], knn=knn, norm=norm, metric=metric)
            bi_names = uni_cnames[i][mnn_ij[:, 0]]
            bj_names = uni_cnames[j][mnn_ij[:, 1]]
            mnn_pairs.append(np.vstack([bi_names, bj_names]).T)

    mnn_pairs = np.vstack(mnn_pairs)
    return mnn_pairs



# exactly the same as procedures used in iMAP
### nns: N * k array, within-batch knn index for each sample
### pairs:   list of tuple, (dataset1.index, dataset2.index)
### steps:   how many steps walking around each sample
def random_walk1(nns, pairs, steps=10):
    pairs_plus = []
    for p in pairs:
        x, y = p[0], p[1]
        pairs_plus.append((x, y))

        for i in range(steps):
            nx = np.random.choice(nns[x])
            ny = np.random.choice(nns[y])
            pairs_plus.append((nx, ny))

    # keep only unique pairs
    pairs_plus = [[p[0], p[1]] for p in set(pairs_plus)]
    pairs_plus = np.asarray(pairs_plus)  # to array 

    return pairs_plus




import time
import faiss


DEFAULT_SEED = 1234

def run_kmeans(x, nmb_clusters, verbose=False, seed=DEFAULT_SEED):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 20
    clus.max_points_per_centroid = 10000000
    clus.seed = seed
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)

    return [int(n[0]) for n in I]