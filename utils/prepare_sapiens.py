import scanpy as sc
import anndata as adata
from sklearn.utils import issparse

def preprocess_rna(
        adata,
        min_features: int = 600,
        min_cells: int = 3,
        target_sum: int = 10000,
        n_top_features=4000,  # or gene list
        chunk_size: int = 20000,
        is_hvg = True,
        batch_key = 'batch',
        log=True
):
    if min_features is None: min_features = 600
    if n_top_features is None: n_top_features = 40000

    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    adata = adata[:, [gene for gene in adata.var_names
                      if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]

    cells_subset, _ = sc.pp.filter_cells(adata, min_genes=min_features, inplace=False)
    adata = adata[cells_subset, :]

    # print(cells_subset)
    # print(adata)

    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    
    if is_hvg == True:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key=batch_key, inplace=True, subset=True)

    print('Processed dataset shape: {}'.format(adata.shape))
    return adata, cells_subset

batch_key = '_scvi_batch'
label_key = 'cell_type'

adata_RNA = sc.read_h5ad('Olga_Data/10df7690-6d10-4029-a47e-0f071bb2df83.h5ad')

adata_RNA, cells_subset = preprocess_rna(adata_RNA, min_features = 0, is_hvg=True, batch_key=batch_key)

tmp = adata_RNA[(adata_RNA.obs[batch_key] == '26') 
                | (adata_RNA.obs[batch_key] == '2')
                | (adata_RNA.obs[batch_key] == '3')
                | (adata_RNA.obs[batch_key] == '13')
                | (adata_RNA.obs[batch_key] == '30')
                | (adata_RNA.obs[batch_key] == '0')].obs["broad_cell_class"]

# print(tmp)
# print(tmp.value_counts())
# print(adata_RNA.obs["broad_cell_class"].value_counts())

adata_RNA = adata_RNA[(adata_RNA.obs["broad_cell_class"] != "melanocyte") 
            & (adata_RNA.obs["broad_cell_class"] != "connective tissue cell") 
            & (adata_RNA.obs["broad_cell_class"] != "retinal pigment epithelial cell")
            & (adata_RNA.obs["broad_cell_class"] != "vestibular dark cell")
            & (adata_RNA.obs["broad_cell_class"] != "female germ cell")
            & (adata_RNA.obs["broad_cell_class"] != "fat cell")
            & (adata_RNA.obs["broad_cell_class"] != "follicle")]


adata_RNA.obs['cell_type_l1'] = adata_RNA.obs['broad_cell_class']
adata_RNA.write_h5ad(f'/home/baunsgaard/scBench/scButterfly/Olga_Data/sapiens.h5ad')
