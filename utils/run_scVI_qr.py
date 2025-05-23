import argparse
import scanpy as sc
import scvi
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


def train_qr_scVI(adata_ref, adata_query, batch_key, cell_type):
    scvi.model.SCVI.setup_anndata(adata_ref, layer="counts", batch_key=batch_key)
    model = scvi.model.SCVI(adata_ref, n_layers=8, n_latent=30, gene_likelihood="nb")

    model.train(accelerator="gpu")
    # model.train()

    SCVI_LATENT_KEY = "X_scVI"
    embedding = model.get_latent_representation()

    # Query
    scvi.model.SCVI.prepare_query_anndata(adata_query, model, return_reference_var_names=True)
    scvi_query = scvi.model.SCVI.load_query_data(adata_query, model)

    scvi_query.train(max_epochs=100, plan_kwargs={"weight_decay": 0.0})
    embedding_test = scvi_query.get_latent_representation()

    # Evaluate
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(embedding, adata_ref.obs[cell_type].tolist())
    cat_preds = knn.predict(embedding_test)

    cell_types_list = pd.unique(adata_query.obs[cell_type]).tolist()
    acc = accuracy_score(adata_query.obs[cell_type].to_list(), cat_preds)
    f1 = f1_score(adata_query.obs[cell_type].to_list(), cat_preds, labels=cell_types_list, average=None)
    f1_weighted = f1_score(adata_query.obs[cell_type].to_list(), cat_preds, labels=cell_types_list, average='weighted')
    f1_macro = f1_score(adata_query.obs[cell_type].to_list(), cat_preds, labels=cell_types_list, average='macro')
    f1_median = np.median(f1)
    
    print(f"Per class {cell_types_list} F1 {f1}")
    print('Accuracy {:.3f}, F1 median {:.3f}, F1 macro {:.3f}, F1 weighted {:.3f} '.format(acc, f1_median, f1_macro, f1_weighted),)

    return acc, f1_macro


# # ImmuneAtlas
# adata_path = "/home/oovcharenko/Olga_Data/ImmuneAtlas.h5ad"
# batch_key = "batchlb"
# cell_type = "cell_type"

# accs, macro_f1s = [], []
# adata = sc.read(adata_path)
# for i in range(0, 2):
#     adata_ref = adata[adata.obs["batchlb"] != "10x 5' v2",].copy()
#     adata_query = adata[adata.obs["batchlb"] == "10x 5' v2",].copy()
#     acc, macro_f1 = train_qr_scVI(adata_ref=adata_ref, adata_query=adata_query, batch_key=batch_key, cell_type=cell_type)
#     accs.append(acc)
#     macro_f1s.append(macro_f1)

# print(f"{adata_path.split('/')[-1]} avg Accuracy {np.mean(accs).round(3)}, macro F1 {np.mean(macro_f1s).round(3)}")
# print(f"{adata_path.split('/')[-1]} std Accuracy {np.std(accs).round(3)}, macro F1 {np.std(macro_f1s).round(3)}")


# Pancreas
adata_path = "/home/oovcharenko/Olga_Data/Pancreas.h5ad"
batch_key = "batchlb"
cell_type = "celltype"

accs, macro_f1s = [], []
batches = ['Baron_b1', 'Mutaro_b2', 'Segerstolpe_b3', 'Wang_b4', 'Xin_b5']
adata = sc.read(adata_path)
for batch in batches:
    adata_ref = adata[adata.obs["batchlb"] != batch,].copy()
    adata_query = adata[adata.obs["batchlb"] == batch,].copy()
    for i in range(0, 2):
        acc, macro_f1 = train_qr_scVI(adata_ref=adata_ref, adata_query=adata_query, batch_key=batch_key, cell_type=cell_type)
        accs.append(acc)
        macro_f1s.append(macro_f1)

    print(f"{adata_path.split('/')[-1]} {batch} avg Accuracy {np.mean(accs).round(3)}, macro F1 {np.mean(macro_f1s).round(3)}")
    print(f"{adata_path.split('/')[-1]} {batch} std Accuracy {np.std(accs).round(3)}, macro F1 {np.std(macro_f1s).round(3)}")
