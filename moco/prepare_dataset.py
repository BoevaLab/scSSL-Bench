import torch
from torch.utils.data import Dataset

import os
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sps

from os.path import join

from moco.utils import py_read_data, load_meta_txt, load_meta_txt7
from moco.config import Config

configs = Config()


def prepare_MouseCellAtlas(data_root):
    data_name = 'filtered_total_batch1_seqwell_batch2_10x'

    sps_x, gene_name, cell_name = py_read_data(data_root, data_name)
    df_meta = load_meta_txt(join(data_root, 'filtered_total_sample_ext_organ_celltype_batch.txt'))
    df_meta['CellType'] = df_meta['ct']

    df_meta[configs.batch_key] = df_meta[configs.batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[configs.label_key].astype('category')

    return sps_x, gene_name, cell_name, df_meta

def prepare_Pancreas(data_root):
    data_name = 'myData_pancreatic_5batches'

    sps_x, gene_name, cell_name = py_read_data(data_root, data_name)
    df_meta = load_meta_txt(join(data_root, 'mySample_pancreatic_5batches.txt'))
    df_meta['CellType'] = df_meta['celltype']

    df_meta[configs.batch_key] = df_meta[configs.batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[configs.label_key].astype('category')

    return sps_x, gene_name, cell_name, df_meta

def prepare_PBMC(data_root):
    sps_x1, gene_name1, cell_name1 = py_read_data(data_root, 'b1_exprs')
    sps_x2, gene_name2, cell_name2 = py_read_data(data_root, 'b2_exprs')

    sps_x = sps.hstack([sps_x1, sps_x2])
    cell_name = np.hstack((cell_name1, cell_name2))

    assert np.all(gene_name1 == gene_name2), 'gene order not match'
    gene_name = gene_name1

    df_meta1 = load_meta_txt(join(data_root, 'b1_celltype.txt'))
    df_meta2 = load_meta_txt(join(data_root, 'b2_celltype.txt'))
    df_meta1['batchlb'] = 'Batch1'
    df_meta2['batchlb'] = 'Batch2'

    df_meta = pd.concat([df_meta1, df_meta2])

    df_meta[configs.batch_key] = df_meta[configs.batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[configs.label_key].astype('category')

    return sps_x, gene_name, cell_name, df_meta

def prepare_PBMC_Full(data_root):
    adata = sc.read_h5ad(join(data_root, 'pbmc.h5ad'))
    
    X = adata.layers['counts'].T

    gene_name = adata.var_names.values
    cell_name = adata.obs_names.values
    df_meta = adata.obs[["batch", "CellType"]].copy()

    # To unify technologies in PBMCFull, uncomment the line below.
    # df_meta.obs['batch'] = [item[6:9] if item[6] == '1' else item[6:] for item in list(adata.obs['batch'])]

    df_meta["batchlb"] = df_meta["batch"].astype('category')
    df_meta["CellType"] = df_meta["CellType"].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_CellLine(data_root):
    b1_exprs_filename = "b1_exprs"
    b2_exprs_filename = "b2_exprs"
    b3_exprs_filename = "b3_exprs"
    b1_celltype_filename = "b1_celltype.txt"
    b2_celltype_filename = "b2_celltype.txt"
    b3_celltype_filename = "b3_celltype.txt"

    # data_name = 'b1_exprs'
    batch_key = 'batchlb'
    label_key = 'CellType'

    expr_mat1, g1, c1 = py_read_data(data_root, b1_exprs_filename)
    metadata1 = pd.read_csv(join(data_root, b1_celltype_filename), sep="\t", index_col=0)

    # expr_mat2 = pd.read_csv(join(data_dir, b2_exprs_filename), sep="\t", index_col=0).T
    expr_mat2, g2, c2 = py_read_data(data_root, b2_exprs_filename)
    metadata2 = pd.read_csv(join(data_root, b2_celltype_filename), sep="\t", index_col=0)

    expr_mat3, g3, c3 = py_read_data(data_root, b3_exprs_filename)
    metadata3 = pd.read_csv(join(data_root, b3_celltype_filename), sep="\t", index_col=0)

    metadata1['batchlb'] = 'Batch_1'
    metadata2['batchlb'] = 'Batch_2'
    metadata3['batchlb'] = 'Batch_3'

    assert np.all(g1 == g2), 'gene name not match'

    cell_name = np.hstack([c1, c2, c3])
    gene_name = g1

    df_meta = pd.concat([metadata1, metadata2, metadata3])
    sps_x = sps.hstack([expr_mat1, expr_mat2, expr_mat3])

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return sps_x, gene_name, cell_name, df_meta 

def prepare_MouseRetina(data_root):
    b1_exprs_filename = "b1_exprs"
    b2_exprs_filename = "b2_exprs"
    b1_celltype_filename = "b1_celltype.txt"
    b2_celltype_filename = "b2_celltype.txt"

    # data_name = 'b1_exprs'
    batch_key = 'batchlb'
    label_key = 'CellType'

    expr_mat1, g1, c1 = py_read_data(data_root, b1_exprs_filename)
    metadata1 = pd.read_csv(join(data_root, b1_celltype_filename), sep="\t", index_col=0)

    # expr_mat2 = pd.read_csv(join(data_root, b2_exprs_filename), sep="\t", index_col=0).T
    expr_mat2, g2, c2 = py_read_data(data_root, b2_exprs_filename)
    metadata2 = pd.read_csv(join(data_root, b2_celltype_filename), sep="\t", index_col=0)

    metadata1['batchlb'] = 'Batch_1'
    metadata2['batchlb'] = 'Batch_2'

    assert np.all(g1 == g2), 'gene name not match'

    cell_name = np.hstack([c1, c2])
    gene_name = g1

    df_meta = pd.concat([metadata1, metadata2])
    sps_x = sps.hstack([expr_mat1, expr_mat2])

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return sps_x, gene_name, cell_name, df_meta 

def prepare_Simulation(data_root):
    # data_root: /home/.../Data/dataset3/simul1_dropout_005_b1_500_b2_900
    batch_key = 'Batch'
    label_key = 'Group'

    # manually switch to counts_all.txt
    # ensure row is gene
    X = pd.read_csv(join(data_root, 'counts.txt'), sep='\t',header=0, index_col=0)  # row is cell
    X = X.T   # to gene

    metadata = pd.read_csv(join(data_root, 'cellinfo.txt'), header=0, index_col=0, sep='\t')
    metadata[configs.batch_key] = metadata[batch_key]
    metadata[configs.label_key] = metadata[label_key]

    return X, X.index.values, X.columns.values, metadata

def prepare_Lung(data_root):
    # data_root: /home/.../Data/dataset3/simul1_dropout_005_b1_500_b2_900
    batch_key = 'batch'
    label_key = 'cell_type'

    # ensure row is gene
    adata = sc.read_h5ad(join(data_root, 'Lung.h5ad'))

    X = adata.layers['counts'].A.T  # gene by cell

    gene_name = adata.var_names
    cell_name = adata.obs_names.values
    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_ImmHuman(data_root):
    batch_key = 'batch'
    label_key = 'final_annotation'
    pseudo_key = 'dpt_pseudotime'

    # ensure row is gene
    adata = sc.read_h5ad(join(data_root, 'Immune_ALL_human.h5ad'))

    X = sps.csr_matrix(adata.layers['counts'].T)  # gene by cell

    gene_name = adata.var_names
    cell_name = adata.obs_names.values
    df_meta = adata.obs[[batch_key, label_key, pseudo_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_ImmHumanMouse(data_root):
    batch_key = 'batch'
    label_key = 'final_annotation'
    pseudo_key = 'dpt_pseudotime'

    # ensure row is gene
    adata = sc.read_h5ad(join(data_root, 'Immune_ALL_hum_mou_filter.h5ad'))

    X = sps.csr_matrix(adata.layers['counts'].T)  # gene by cell

    gene_name = adata.var_names
    cell_name = adata.obs_names.values
    df_meta = adata.obs[[batch_key, label_key, pseudo_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Mixology(data_root):
    batch_key = 'batch'
    label_key = 'cell_line_demuxlet'

    # ensure row is gene
    adata = sc.read_h5ad(join(data_root, 'sc_mixology.h5ad'))
    adata = adata[:, adata.var.Selected.values.astype('bool')].copy()  # use selected hvg, 2000

    # X = sps.csr_matrix(adata.layers['norm_data'].T)  # gene by cell
    X = sps.csr_matrix(adata.X.T)  # gene by cell

    gene_name = adata.var_names
    cell_name = adata.obs_names.values
    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Muris(data_root):
    batch_key = 'batch'
    label_key = 'cell_ontology_class'

    adata = sc.read_h5ad(join(data_root, 'muris_sample_filter.h5ad'))  # 6w cells

    X = sps.csr_matrix(adata.layers['counts'].T)

    gene_name = adata.var_names
    cell_name = adata.obs_names.values

    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Muris_2000(data_root):
    batch_key = 'batch'
    label_key = 'cell_ontology_class'

    adata = sc.read_h5ad(join(data_root, 'muris_sample_2000.h5ad'))  # 6w cells

    X = sps.csr_matrix(adata.layers['counts'].T)

    gene_name = adata.var_names
    cell_name = adata.obs_names.values

    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Muris_4000(data_root):
    batch_key = 'batch'
    label_key = 'cell_ontology_class'

    adata = sc.read_h5ad(join(data_root, 'muris_sample_4000.h5ad'))  # 6w cells

    X = sps.csr_matrix(adata.layers['counts'].T)

    gene_name = adata.var_names
    cell_name = adata.obs_names.values

    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Muris_8000(data_root):
    batch_key = 'batch'
    label_key = 'cell_ontology_class'

    adata = sc.read_h5ad(join(data_root, 'muris_sample_8000.h5ad'))  # 6w cells

    X = sps.csr_matrix(adata.layers['counts'].T)

    gene_name = adata.var_names
    cell_name = adata.obs_names.values

    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Muris_16000(data_root):
    batch_key = 'batch'
    label_key = 'cell_ontology_class'

    adata = sc.read_h5ad(join(data_root, 'muris_sample_16000.h5ad'))  # 6w cells

    X = sps.csr_matrix(adata.layers['counts'].T)

    gene_name = adata.var_names
    cell_name = adata.obs_names.values

    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Muris_30000(data_root):
    batch_key = 'batch'
    label_key = 'cell_ontology_class'

    adata = sc.read_h5ad(join(data_root, 'muris_sample_30000.h5ad'))  # 6w cells

    X = sps.csr_matrix(adata.layers['counts'].T)

    gene_name = adata.var_names
    cell_name = adata.obs_names.values

    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Muris_60000(data_root):
    batch_key = 'batch'
    label_key = 'cell_ontology_class'

    adata = sc.read_h5ad(join(data_root, 'muris_sample_60000.h5ad'))  # 6w cells

    X = sps.csr_matrix(adata.layers['counts'].T)

    gene_name = adata.var_names
    cell_name = adata.obs_names.values

    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Muris_120000(data_root):
    batch_key = 'batch'
    label_key = 'cell_ontology_class'

    adata = sc.read_h5ad(join(data_root, 'muris_sample_120000.h5ad'))  # 6w cells

    X = sps.csr_matrix(adata.layers['counts'].T)

    gene_name = adata.var_names
    cell_name = adata.obs_names.values

    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Neo(data_root):
    # batch_key = 'batch'
    label_key = 'grouping'

    import h5py
    filename = join(data_root, 'mouse_brain_merged.h5')

    with h5py.File(filename, "r") as f:
        # List all groups
        cell_name = list(map(lambda x:x.decode('utf-8'), f['cell_ids'][...]))
        gene_name = list(map(lambda x:x.decode('utf-8'), f['gene_names'][...]))
        
        X = sps.csr_matrix(f['count'][...].T)  # transpose to (genes, cells)
        types = list(map(lambda x:x.decode('utf-8'), f['grouping'][...]))

    df_meta = pd.DataFrame(types, index=cell_name, columns=[configs.label_key])
    df_meta[configs.batch_key] = 'Batch_B'
    df_meta.iloc[:10261, -1] = 'Batch_A'     
    return X, gene_name, cell_name, df_meta

def prepare_PBMCMultome(data_root):
    label_key = 'seurat_annotations'

    adata_rna = sc.read_h5ad(join(data_root, 'RNA/adata_rna.h5ad'))
    adata_atac = sc.read_h5ad(join(data_root, 'ATAC_GAM/adata_atac_gam.h5ad'))

    share_gene = np.intersect1d(adata_rna.var_names, adata_atac.var_names)
    adata_rna = adata_rna[:, share_gene]
    adata_atac = adata_atac[:, share_gene]

    X = sps.vstack([adata_rna.X, adata_atac.X]).T
    X = sps.csr_matrix(X)

    meta1 = pd.read_csv(join(data_root, 'metadata.csv'), index_col=0)
    meta1[configs.label_key] = meta1[label_key]
    meta1[configs.batch_key] = 'RNA'
    meta2 = meta1.copy()
    meta2[configs.batch_key] = 'ATAC'

    meta1.index = [f'{_}_reference' for _ in meta1.index] 
    meta2.index = [f'{_}_query' for _ in meta2.index]
    meta = pd.concat([meta1, meta2])

    cname = np.array(meta.index)
    return X, share_gene, cname, meta

def prepare_PBMC_Multimodal(data_root):
    adata_RNA = sc.read_h5ad(join(data_root, 'adata_RNA_full.h5ad'))
    adata_Protein = sc.read_h5ad(join(data_root, 'adata_Protein_full.h5ad'))

    train_idx = (adata_RNA.obs["batch"] != "P5") & (adata_RNA.obs["batch"] != "P8")
    # test_idx = (train_idx != 1)

    X_RNA = adata_RNA.X.A.T  # gene by cell
    X_Protein = adata_Protein.X.A.T  # gene by cell

    X = np.vstack([X_RNA, X_Protein])

    gene_name = np.concatenate([adata_RNA.var_names.values, adata_Protein.var_names.values])
    cell_name = adata_RNA.obs_names.values

    batch_key, label_key, query_reference_key = "batch", "cell_type_l1", "query_reference"
    df_meta = adata_RNA.obs[[batch_key, label_key]].copy()
    df_meta[query_reference_key] = train_idx

    modality = ["RNA"] * X_RNA.shape[0] + ["Protein"] * X_Protein.shape[0]

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, [df_meta, modality]

def prepare_Neurips_cite_Multimodal(data_root):
    adata_RNA = sc.read_h5ad(join(data_root, 'adata_neurips_GEX_full.h5ad'))
    adata_Protein = sc.read_h5ad(join(data_root, 'adata_neurips_ADT_full.h5ad'))

    X_RNA = adata_RNA.X.A.T  # gene by cell
    X_Protein = adata_Protein.X.A.T  # gene by cell

    X = np.vstack([X_RNA, X_Protein])

    gene_name = np.concatenate([adata_RNA.var_names.values, adata_Protein.var_names.values])
    cell_name = adata_RNA.obs_names.values

    batch_key, label_key,  = "batch", "cell_type_l1"
    df_meta = adata_RNA.obs[[batch_key, label_key]].copy()

    modality = ["RNA"] * X_RNA.shape[0] + ["Protein"] * X_Protein.shape[0]

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')
    
    return X, gene_name, cell_name, [df_meta, modality]

# Function taken from concerto
def preprocessing_rna(
        adata,
        min_features: int = 600,
        min_cells: int = 3,
        target_sum: int = 10000,
        n_top_features=2000,  # or gene list
        chunk_size: int = 20000,
        is_hvg = False,
        batch_key = 'batch',
        log=True
):
    if min_features is None: min_features = 600
    if n_top_features is None: n_top_features = 40000

    if not sps.issparse(adata.X):
        adata.X = sps.csr_matrix(adata.X)

    # adata = adata[:, [gene for gene in adata.var_names
    #                   if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]

    sc.pp.filter_cells(adata, min_genes=min_features)

    #sc.pp.filter_genes(adata, min_cells=min_cells)

    sc.pp.normalize_total(adata, target_sum=target_sum)

    sc.pp.log1p(adata)
    if is_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key=batch_key, inplace=False, subset=True)

    print('Processed dataset shape: {}'.format(adata.shape))
    return adata

def prepare_SimulatedConcerto(data_root):
    '''
      return:
         X:         scipy.sparse.csr_matrix, row = feature, column = cell
         gene_name: array of feature (gene) names
         cell_name: array of cell (barcodes) names
         df_meta:   metadata of dataset, columns include 'batchlb'(batch column), 'CellType'(optional)
    '''
    # ===========
    # example

    # label_key = 'final_annotation'

    adata = sc.read(join(data_root, 'expBatch1_woGroup2.loom'))  # read simulated dataset
    adata = preprocessing_rna(adata, n_top_features=2000, is_hvg=True, batch_key='Batch')
    # X is already sparse after preprocessingRNA function.
    X = adata.X.T # must be gene by cell TODO: true?

    gene_name = adata.var_names.values
    cell_name = adata.obs_names.values
    df_meta = adata.obs[["Batch", "Group"]].copy()

    df_meta["batchlb"] = df_meta["Batch"].astype('category')
    df_meta["CellType"] = df_meta["Group"].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Neftel(data_root):
    '''
      return:
         X:         scipy.sparse.csr_matrix, row = feature, column = cell
         gene_name: array of feature (gene) names
         cell_name: array of cell (barcodes) names
         df_meta:   metadata of dataset, columns include 'batchlb'(batch column), 'CellType'(optional)
    '''
    # The adata object read below contains:
    #   -   the sample (27 in total; one seems to be missing - there should
    #       be 28 as the study had 28 participants) => adata.obs['sample']
    #   -   the assigned cancer-subtype-label => 4 labels => adata.obs['celltype']
    #   -   the subclones; I'm guessing it's the subclones as discovered by hierarchical clustering 
    #   -   on the inferred CNAs => adata.obs['subclones']
    #   -   the phase of the cell => adata.obs['phase']; there are also obs['G2M_score'] and obs['S_score'].

    # Aditionally it contains the inferred CNV's (obsm['X_cnv'])
    # The counts are located in the layers['counts'] object. TODO: how can we check if the data is normalized?
    # The max value in the counts array is 488071. This seems a lot for a normalized matrix..

    # There are 6855 cells in the dataset. In the original there were >17K, but the authors dropped the non-cancerous
    # cells, leaving roughly 6.5K cells (this fits).
    adata = sc.read_h5ad(join(data_root, 'neftel_ss2.h5ad'))
    #adata = preprocessing_rna(adata, n_top_features=2000, is_hvg=True, batch_key='Batch')
    
    # Filtering out cy cycling cells (G2M). TODO: Do the S-Phase cells also need to be filtered out? If not: use .concatenate(adata[adata.obs['phase']=="S"])
    adata = adata[adata.obs['phase']=="G1"]
    X = adata.layers['counts'].T

    gene_name = adata.var_names.values
    cell_name = adata.obs_names.values
    df_meta = adata.obs[["sample", "celltype"]].copy()

    df_meta["batchlb"] = df_meta["sample"].astype('category')
    df_meta["CellType"] = df_meta["celltype"].astype('category')

    return X, gene_name, cell_name, [df_meta, adata.obsm['X_cnv'], adata.uns['cnv']]

def prepare_Neftel_10X(data_root):
    '''
      return:
         X:         scipy.sparse.csr_matrix, row = feature, column = cell
         gene_name: array of feature (gene) names
         cell_name: array of cell (barcodes) names
         df_meta:   metadata of dataset, columns include 'batchlb'(batch column), 'CellType'(optional)
    '''
    # The adata object read below contains:
    #   -   the sample (27 in total; one seems to be missing - there should
    #       be 28 as the study had 28 participants) => adata.obs['sample']
    #   -   the assigned cancer-subtype-label => 4 labels => adata.obs['celltype']
    #   -   the subclones; I'm guessing it's the subclones as discovered by hierarchical clustering 
    #   -   on the inferred CNAs => adata.obs['subclones']
    #   -   the phase of the cell => adata.obs['phase']; there are also obs['G2M_score'] and obs['S_score'].

    # Aditionally it contains the inferred CNV's (obsm['X_cnv'])
    # The counts are located in the layers['counts'] object. TODO: how can we check if the data is normalized?
    # The max value in the counts array is 488071. This seems a lot for a normalized matrix..

    # There are 6855 cells in the dataset. In the original there were >17K, but the authors dropped the non-cancerous
    # cells, leaving roughly 6.5K cells (this fits).
    adata = sc.read_h5ad(join(data_root, 'neftel_10x.h5ad'))
    #adata = preprocessing_rna(adata, n_top_features=2000, is_hvg=True, batch_key='Batch')
    
    # Filtering out cy cycling cells (G2M). TODO: Do the S-Phase cells also need to be filtered out? If not: use .concatenate(adata[adata.obs['phase']=="S"])
    adata = adata[adata.obs['phase']=="G1"]
    X = adata.layers['counts'].T

    gene_name = adata.var_names.values
    cell_name = adata.obs_names.values
    df_meta = adata.obs[["sample", "celltype"]].copy()

    df_meta["batchlb"] = df_meta["sample"].astype('category')
    df_meta["CellType"] = df_meta["celltype"].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Neftel_ss2_10X(data_root):
    '''
      return:
         X:         scipy.sparse.csr_matrix, row = feature, column = cell
         gene_name: array of feature (gene) names
         cell_name: array of cell (barcodes) names
         df_meta:   metadata of dataset, columns include 'batchlb'(batch column), 'CellType'(optional)
    '''
    # The adata object read below contains:
    #   -   the sample (27 in total; one seems to be missing - there should
    #       be 28 as the study had 28 participants) => adata.obs['sample']
    #   -   the assigned cancer-subtype-label => 4 labels => adata.obs['celltype']
    #   -   the subclones; I'm guessing it's the subclones as discovered by hierarchical clustering 
    #   -   on the inferred CNAs => adata.obs['subclones']
    #   -   the phase of the cell => adata.obs['phase']; there are also obs['G2M_score'] and obs['S_score'].

    # Aditionally it contains the inferred CNV's (obsm['X_cnv'])
    # The counts are located in the layers['counts'] object. TODO: how can we check if the data is normalized?
    # The max value in the counts array is 488071. This seems a lot for a normalized matrix..

    # There are 6855 cells in the dataset. In the original there were >17K, but the authors dropped the non-cancerous
    # cells, leaving roughly 6.5K cells (this fits).
    adata = sc.read_h5ad(join(data_root, 'Neftel_10X_ss2.h5ad'))
    #adata = preprocessing_rna(adata, n_top_features=2000, is_hvg=True, batch_key='Batch')
    
    # Filtering out cy cycling cells (G2M). TODO: Do the S-Phase cells also need to be filtered out? If not: use .concatenate(adata[adata.obs['phase']=="S"])
    adata = adata[adata.obs['phase']=="G1"]
    X = adata.layers['counts'].T

    gene_name = adata.var_names.values
    cell_name = adata.obs_names.values
    df_meta = adata.obs[["sample", "celltype"]].copy()

    df_meta["batchlb"] = df_meta["sample"].astype('category')
    df_meta["CellType"] = df_meta["celltype"].astype('category')

    return X, gene_name, cell_name, df_meta


def prepare_ImmuneAtlas(data_root):
    '''
      return:
         X:         scipy.sparse.csr_matrix, row = feature, column = cell
         gene_name: array of feature (gene) names
         cell_name: array of cell (barcodes) names
         df_meta:   metadata of dataset, columns include 'batchlb'(batch column), 'CellType'(optional)
    '''
    
    adata = sc.read_h5ad(join(data_root, 'Conde.h5ad'))
    #adata = preprocessing_rna(adata, n_top_features=2000, is_hvg=True, batch_key='Batch')
    
    # Filtering out cy cycling cells (G2M). TODO: Do the S-Phase cells also need to be filtered out? If not: use .concatenate(adata[adata.obs['phase']=="S"])
    #adata = adata[adata.obs['phase']=="G1"]

    X = adata.raw.X.T

    gene_name = adata.var_names.values
    cell_name = adata.obs_names.values
    df_meta = adata.obs[["donor_id", "cell_type", "assay",]].copy()

    df_meta["batchlb"] = df_meta["assay"].astype('category')
    df_meta["CellType"] = df_meta["cell_type"].astype('category')

    return X, gene_name, cell_name, df_meta


def prepare_HCLA_Core(data_root):
    '''
      return:
         X:         scipy.sparse.csr_matrix, row = feature, column = cell
         gene_name: array of feature (gene) names
         cell_name: array of cell (barcodes) names
         df_meta:   metadata of dataset, columns include 'batchlb'(batch column), 'CellType'(optional)
    '''
    
    adata = sc.read_h5ad(join(data_root, 'Luecken_Core.h5ad'))
    #adata = preprocessing_rna(adata, n_top_features=2000, is_hvg=True, batch_key='Batch')
    
    # Filtering out cy cycling cells (G2M). TODO: Do the S-Phase cells also need to be filtered out? If not: use .concatenate(adata[adata.obs['phase']=="S"])
    #adata = adata[adata.obs['phase']=="G1"]

    X = adata.raw.X.T

    gene_name = adata.var_names.values
    cell_name = adata.obs_names.values
    df_meta = adata.obs[["donor_id", "cell_type", "study"]].copy()

    df_meta["batchlb"] = df_meta["study"].astype('category')
    df_meta["CellType"] = df_meta["cell_type"].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Ji(data_root):
    '''
      return:
         X:         scipy.sparse.csr_matrix, row = feature, column = cell
         gene_name: array of feature (gene) names
         cell_name: array of cell (barcodes) names
         df_meta:   metadata of dataset, columns include 'batchlb'(batch column), 'CellType'(optional)
    '''
    # ===========
    # example

    # label_key = 'final_annotation'

    adata = sc.read_h5ad(join(data_root, 'Ji_skin.h5ad'))  # read simulated dataset
    
    # /cluster/work/boeva/tomap/CLAIRE-data/Ji/Ji_skin.h5ad
    # adata = preprocessing_rna(adata, n_top_features=2000, is_hvg=True, batch_key='Batch')
    # X is already sparse after preprocessingRNA function.
    # X = adata.X.T # must be gene by cell TODO: true?
    adata = adata[adata.obs['phase']=="G1"]
    X = adata.layers['counts'].T

    gene_name = adata.var_names.values
    cell_name = adata.obs_names.values
    df_meta = adata.obs[["sample", "celltype"]].copy()

    df_meta["batchlb"] = df_meta["sample"].astype('category')
    df_meta["CellType"] = df_meta["celltype"].astype('category')

    return X, gene_name, cell_name, df_meta


def prepare_Uhlitz(data_root):
    '''
      return:
         X:         scipy.sparse.csr_matrix, row = feature, column = cell
         gene_name: array of feature (gene) names
         cell_name: array of cell (barcodes) names
         df_meta:   metadata of dataset, columns include 'batchlb'(batch column), 'CellType'(optional)
    '''
    
    adata = sc.read_h5ad(join(data_root, 'uhlitz_S_G2M_removed.h5ad'))  # read simulated dataset
    
    # /cluster/work/boeva/tomap/CLAIRE-data/Ji/Ji_skin.h5ad
    # adata = preprocessing_rna(adata, n_top_features=2000, is_hvg=True, batch_key='Batch')
    # X is already sparse after preprocessingRNA function.
    # X = adata.X.T # must be gene by cell TODO: true?
    adata = adata[adata.obs['origin']=="tumor"]
    # adata = adata[adata.obs['phase']=="G1"]
    adata.layers['counts'] = adata.X
    X = adata.layers['counts'].T

    gene_name = adata.var_names.values
    cell_name = adata.obs_names.values
    # TODO: get celltypes!
    df_meta = adata.obs[["patient", "celltype"]].copy()

    df_meta["batchlb"] = df_meta["patient"].astype('category')
    df_meta["CellType"] = df_meta["celltype"].astype('category')

    return X, gene_name, cell_name, df_meta


def prepare_Lee(data_root):
    '''
      return:
         X:         scipy.sparse.csr_matrix, row = feature, column = cell
         gene_name: array of feature (gene) names
         cell_name: array of cell (barcodes) names
         df_meta:   metadata of dataset, columns include 'batchlb'(batch column), 'CellType'(optional)
    '''
    
    #adata = sc.read_h5ad(join(data_root, 'GBM_smallest_GSE154795.h5ad'))  # read simulated dataset
    adata = sc.read_h5ad(join(data_root, 'GBM_IMM_10X_GSE154795.h5ad'))  # read simulated dataset
    
    # /cluster/work/boeva/tomap/CLAIRE-data/Ji/Ji_skin.h5ad
    # adata = preprocessing_rna(adata, n_top_features=2000, is_hvg=True, batch_key='Batch')
    # X is already sparse after preprocessingRNA function.
    # X = adata.X.T # must be gene by cell TODO: true?

    #adata = adata[adata.obs['origin']=="tumor"]
    # adata = adata[adata.obs['phase']=="G1"]

    adata.layers['counts'] = adata.X
    X = adata.layers['counts'].T

    gene_name = adata.var_names.values
    cell_name = adata.obs_names.values
    # TODO: get celltypes!
    df_meta = adata.obs[["ID", "annotation_minor"]].copy()

    df_meta["batchlb"] = df_meta["ID"].astype('category')
    df_meta["CellType"] = df_meta["annotation_minor"].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_NewDataset(data_root):
    '''
      return:
         X:         scipy.sparse.csr_matrix, row = feature, column = cell
         gene_name: array of feature (gene) names
         cell_name: array of cell (barcodes) names
         df_meta:   metadata of dataset, columns include 'batchlb'(batch column), 'CellType'(optional)
    '''
    # ===========
    # example

    # batch_key = 'batch'
    # label_key = 'final_annotation'

    # adata = sc.read_h5ad(join(data_root, 'Immune_ALL_human.h5ad'))  # read ImmHuman dataset
    # X = sps.csr_matrix(adata.layers['counts'].T)  # gene by cell

    # gene_name = adata.var_names.values
    # cell_name = adata.obs_names.values
    # df_meta = adata.obs[[batch_key, label_key]].copy()

    # df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    # df_meta[configs.label_key] = df_meta[label_key].astype('category')

    # return X, gene_name, cell_name, df_meta
    # ===========

def prepare_ImmHuman_our(data_root):
    batch_key = 'batch'
    label_key = 'CellType'

    # ensure row is gene
    adata = sc.read_h5ad('/home/baunsgaard/scBench/scButterfly/Olga_Data/ImmHuman.h5ad')

    X = sps.csr_matrix(adata.layers['counts'].T)  # gene by cell

    gene_name = adata.var_names
    cell_name = adata.obs_names.values
    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_PBMC_our(data_root):
    batch_key = 'batch'
    label_key = 'CellType'

    # ensure row is gene
    adata = sc.read_h5ad('/home/baunsgaard/scBench/scButterfly/Olga_Data/PBMC.h5ad')

    X = adata.layers['counts'].A.T  # gene by cell

    gene_name = adata.var_names
    cell_name = adata.obs_names.values
    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_sapiens(data_root):
    batch_key = '_scvi_batch'
    label_key = 'broad_cell_class'

    # ensure row is gene
    adata = sc.read_h5ad('/home/baunsgaard/scBench/scButterfly/Olga_Data/sapiens.h5ad')

    X = adata.X.T  # gene by cell

    gene_name = adata.var_names
    cell_name = adata.obs_names.values
    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Pancreas_our(data_root):
    batch_key = 'batchlb'
    label_key = 'celltype'

    # ensure row is gene
    adata = sc.read_h5ad('/home/baunsgaard/scBench/scButterfly/Olga_Data/Pancreas.h5ad')

    X = adata.layers['counts'].A.T  # gene by cell

    gene_name = adata.var_names
    cell_name = adata.obs_names.values
    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_ImmuneAtlas_our(data_root):
    batch_key = 'batchlb'
    label_key = 'cell_type'

    # ensure row is gene
    adata = sc.read_h5ad('/home/baunsgaard/scBench/scButterfly/Olga_Data/ImmuneAtlas.h5ad')

    X = adata.layers['counts'].A.T  # gene by cell

    gene_name = adata.var_names
    cell_name = adata.obs_names.values
    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_MCA_our(data_root):
    batch_key = 'batch'
    label_key = 'CellType'

    # ensure row is gene
    adata = sc.read_h5ad('/home/baunsgaard/scBench/scButterfly/Olga_Data/MCA.h5ad')

    X = adata.layers['counts'].A.T  # gene by cell

    gene_name = adata.var_names
    cell_name = adata.obs_names.values
    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta

def prepare_Lung_our(data_root):
    batch_key = 'batch'
    label_key = 'cell_type'

    # ensure row is gene
    adata = sc.read_h5ad('/home/baunsgaard/scBench/scButterfly/Olga_Data/Lung.h5ad')

    X = adata.layers['counts'].A.T  # gene by cell

    gene_name = adata.var_names
    cell_name = adata.obs_names.values
    df_meta = adata.obs[[batch_key, label_key]].copy()

    df_meta[configs.batch_key] = df_meta[batch_key].astype('category')
    df_meta[configs.label_key] = df_meta[label_key].astype('category')

    return X, gene_name, cell_name, df_meta


def prepare_dataset(data_dir):
    dataset_name = data_dir.split('/')[-1]
    func_dict = {
                    'MouseCellAtlas': prepare_MouseCellAtlas, 
                    # 'Pancreas': prepare_Pancreas, 
                    # 'PBMC': prepare_PBMC,
                    'Neurips_cite_multimodal': prepare_Neurips_cite_Multimodal,
                    'PBMC_multimodal': prepare_PBMC_Multimodal, 
                    'PBMCFull': prepare_PBMC_Full, 
                    'CellLine': prepare_CellLine, 
                    'MouseRetina': prepare_MouseRetina, 
                    # 'Lung': prepare_Lung,
                    # 'ImmHuman': prepare_ImmHuman,
                    # 'ImmuneAtlas': prepare_ImmuneAtlas,
                    'Muris': prepare_Muris,
                    'Neocortex': prepare_Neo,
                    'Muris_2000': prepare_Muris_2000,
                    'Muris_4000': prepare_Muris_4000,
                    'Muris_8000': prepare_Muris_8000,
                    'Muris_16000': prepare_Muris_16000,
                    'Muris_30000': prepare_Muris_30000,
                    'Muris_60000': prepare_Muris_60000,
                    'Muris_120000': prepare_Muris_120000,
                    'PBMCMultome': prepare_PBMCMultome,
                    'Simulated': prepare_SimulatedConcerto,
                    'Neftel': prepare_Neftel,
                    'Neftel-Non-Cycling': prepare_Neftel,
                    'Neftel10X': prepare_Neftel_10X,
                    'NeftelMulti': prepare_Neftel_ss2_10X,
                    'Ji': prepare_Ji,
                    'Uhlitz': prepare_Uhlitz,
                    'Lee': prepare_Lee,
                    'HLCACore': prepare_HCLA_Core,
                    'new_dataset': prepare_NewDataset,

                    'ImmHuman': prepare_ImmHuman_our,
                    'Pancreas': prepare_Pancreas_our,
                    'ImmuneAtlas': prepare_ImmuneAtlas_our,
                    'Lung': prepare_Lung_our,
                    'MCA': prepare_MCA_our,
                    'PBMC': prepare_PBMC_our,
                    'sapiens': prepare_sapiens,
    }

    # dataset 3 
    return func_dict.get(dataset_name, prepare_Simulation)(data_dir)