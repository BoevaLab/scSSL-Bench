import os
import torch
import numpy as np
import pandas as pd
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import scanpy as sc
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


_BIO_METRICS = BioConservation(isolated_labels=True, 
                               nmi_ari_cluster_labels_leiden=True, 
                               nmi_ari_cluster_labels_kmeans=False, 
                               silhouette_label=True, 
                               clisi_knn=True
                               )
_BATCH_METRICS = BatchCorrection(graph_connectivity=True, 
                                 kbet_per_label=True, 
                                 ilisi_knn=True, 
                                 pcr_comparison=True, 
                                 silhouette_batch=True
                                 )

def infer_embedding(model, val_loader, dsbn: bool = False):
    outs = []
    for x in val_loader:
        with torch.no_grad():
            if dsbn:
                outs.append(model.predict_dsbn(x[0], x[2])) 
            else:
                outs.append(model.predict(x[0]))
    
    embedding = torch.concat(outs)
    embedding = np.array(embedding)
    return embedding

def infer_embedding_separate(model, val_loader):
    outs, rnas, proteins = [], [], []
    for x in val_loader:
        with torch.no_grad():
            out, rna, protein = model.predict_separate(x[0])
            outs.append(out)
            rnas.append(rna)
            proteins.append(protein)
    
    embedding = torch.concat(outs)
    embedding = np.array(embedding)

    embedding_rna = torch.concat(rnas)
    embedding_rna = np.array(embedding_rna)

    embedding_protein = torch.concat(proteins)
    embedding_protein = np.array(embedding_protein)
    return embedding, embedding_rna, embedding_protein

def infer_projector_embedding(model, val_loader):
    outs = []
    for x in val_loader:
        with torch.no_grad():
            outs.append(model(x[0]))
    
    embedding = torch.concat(outs)
    embedding = np.array(embedding)
    return embedding


def evaluate_model(model, adata, dataset, batch_size, num_workers, logger, embedding_save_path,
                   batch_key="batchlb", cell_type_label="CellType", umap_plot="", dsbn=False):
    val_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    drop_last=False)
    embedding = infer_embedding(model, val_loader, dsbn=dsbn)
    np.savez_compressed(embedding_save_path, embedding)

    logger.info(f"Inferred embedding of shape {embedding.shape}")
    adata.obsm["Embedding"] = embedding

    sc.pp.neighbors(adata, use_rep="Embedding", metric="cosine")
    sc.tl.umap(adata, min_dist=0.1)
    sc.pl.umap(adata, color=["CellType", "batch"], legend_fontweight='light') 
    plt.savefig(umap_plot)
    try:
        bm = Benchmarker(
                    adata,
                    batch_key=batch_key,
                    label_key=cell_type_label,
                    embedding_obsm_keys=["Embedding"],
                    bio_conservation_metrics=_BIO_METRICS,
                    batch_correction_metrics=_BATCH_METRICS,
                    n_jobs=num_workers,
                )
        bm.benchmark()
        a = bm.get_results(False, True)
        results = a[:1].astype(float).round(4)
    except Exception as error:
        results = None
        logger.info(".. An exception occured while evaluating:", error)

    return results, embedding

def recalculate_results(adata, embedding, num_workers,
                   batch_key="batchlb", cell_type_label="CellType",):
    adata.obsm["Embedding"] = embedding
    bm = Benchmarker(
                adata,
                batch_key=batch_key,
                label_key=cell_type_label,
                embedding_obsm_keys=["Embedding"],
                bio_conservation_metrics=_BIO_METRICS,
                batch_correction_metrics=_BATCH_METRICS,
                n_jobs=num_workers,
            )
    bm.benchmark()
    a = bm.get_results(False, True)
    results = a[:1]

    return results.astype(float).round(4)


def plot_umap(adata, embedding, results_dir):
    adata.obsm['Embedding'] = embedding
    sc.pp.neighbors(adata, use_rep="Embedding")
    sc.tl.umap(adata)
    return sc.pl.umap(adata, show=False, color=['CellType', 'batchlb'],)


def collect_runs(project_root = "/local/home/tomap/scAugmentBench", dirname = "architecture-ablation",
                 dname = "ImmHuman", n_runs = 5):
    model_names = os.listdir(os.path.join(project_root, dirname, dname))
    
    for mname in model_names:
        tmp = os.path.join(project_root, dirname, dname, mname)
        
        if os.path.exists(os.path.join(tmp, "mean_result.csv")):
            os.remove(os.path.join(tmp, "mean_result.csv"))
            os.remove(os.path.join(tmp, "std_result.csv"))
        
        assert len(os.listdir(tmp))==n_runs, f"Number of runs does not match number of seeds @ {tmp}!"
        
        metrics = [pd.read_csv(os.path.join(tmp, seed, "evaluation_metrics.csv")) for seed in os.listdir(tmp)]
        mean = pd.DataFrame(pd.concat(metrics).mean(0).round(4), columns=[mname]).T
        std = pd.DataFrame(pd.concat(metrics).std(0).round(4), columns=[mname]).T
        mean.to_csv(os.path.join(tmp, "mean_result.csv"))
        std.to_csv(os.path.join(tmp, "std_result.csv"))


def unify_table(project_root = "/local/home/tomap/scAugmentBench", dirname = "architecture-ablation",
                dname = "ImmHuman", n_runs = 5):
    model_names = os.listdir(os.path.join(project_root, dirname, dname))
    model_means = []
    model_stds = []

    for mname in model_names:
        tmp = os.path.join(project_root, dirname, dname, mname)
        assert os.path.exists(os.path.join(tmp, "mean_result.csv")), f"There is no file for the mean-metrics @ {tmp}."
        model_means.append(pd.read_csv(os.path.join(tmp, "mean_result.csv"), index_col=0))
        model_stds.append(pd.read_csv(os.path.join(tmp, "std_result.csv"), index_col=0))
    
    means = pd.concat(model_means)
    stds = pd.concat(model_stds)
    
    return means, stds


def scale_table(df):
    biometrics = np.array(['Isolated labels', 'Leiden', 'KMeans', 'Silhouette label', 'cLISI'])
    batchmetrics = np.array(['Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity', 'PCR comparison'])

    biometrics = list(biometrics[list(_BIO_METRICS.__dict__.values())])
    tmp = biometrics[1]
    biometrics[1] = tmp + " ARI"
    biometrics.append(tmp + " NMI")
    batchmetrics = list(batchmetrics[list(_BATCH_METRICS.__dict__.values())])

    scaled= pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns, index=df.index)
    scaled['Batch correction'] = scaled[batchmetrics].mean(1)
    scaled['Bio conservation'] = scaled[biometrics].mean(1)
    scaled['Total'] = 0.6 * scaled['Bio conservation'] + 0.4 * scaled['Batch correction']
    df['Batch correction'] = scaled['Batch correction'].copy()
    df['Bio conservation'] = scaled['Bio conservation'].copy()
    df['Total'] = scaled['Total'].copy()

    return df


def get_bigger_table(project_root = "/local/home/tomap/scAugmentBench", dirname = "architecture-ablation",
                 dname = "ImmHuman"):
    root = os.path.join(project_root, dirname, dname)
    model_names = os.listdir(root)
    if os.path.exists(os.path.join(root, "final_collected.csv")):
            os.remove(os.path.join(root, "final_collected.csv"))
        
    for mname in model_names:
        tmp = os.path.join(project_root, dirname, dname, mname)
        
        if os.path.exists(os.path.join(tmp, "mean_result_collected.csv")):
            os.remove(os.path.join(tmp, "mean_result_collected.csv"))
        if os.path.exists(os.path.join(tmp, "std_result_collected.csv")):
            os.remove(os.path.join(tmp, "std_result_collected.csv"))
        
        n_seeds = [len(os.listdir(os.path.join(tmp, param_config))) for param_config in os.listdir(os.path.join(tmp))]
        print(f"Min num seeds: {min(n_seeds)}.\nMax num seeds: {max(n_seeds)}.")

        # get mean and std per parameter-config:
        for param in os.listdir(tmp):
            if "mean_result.csv" in os.listdir(os.path.join(tmp, param)):
                os.remove(os.path.join(tmp, param, "mean_result.csv"))
            if "std_result.csv" in os.listdir(os.path.join(tmp, param)):
                os.remove(os.path.join(tmp, param, "std_result.csv"))
            metrics = [pd.read_csv(os.path.join(tmp, param, seed, "evaluation_metrics.csv")) for seed in os.listdir(os.path.join(tmp, param))]
            df = pd.DataFrame(pd.concat(metrics).round(4), columns=[seed for seed in os.listdir(os.path.join(tmp, param))]) #columns=["-".join([mname, param])]).T
            #std = pd.DataFrame(pd.concat(metrics).std(0).round(4), columns=[mname + "-" + param]).T
            #mean.to_csv(os.path.join(tmp, param, "mean_result.csv"), index=True)
            #std.to_csv(os.path.join(tmp, param, "std_result.csv"), index=True)
            print(mean)
        
        # collect across param-configs.
        """params = os.listdir(tmp)
        means = [pd.read_csv(os.path.join(tmp, param, "mean_result.csv"), index_col=0) for param in params]
        std = [pd.read_csv(os.path.join(tmp, param, "std_result.csv"), index_col=0) for param in params]
        mean = pd.DataFrame(pd.concat(means).round(4))
        std = pd.DataFrame(pd.concat(std).round(4))
        params = [mname + "-" + p for p in params]
        mean.to_csv(os.path.join(tmp, "mean_result_collected.csv"))
        std.to_csv(os.path.join(tmp, "std_result_collected.csv"))
    
    means = [pd.read_csv(os.path.join(root, mname, "mean_result_collected.csv"), index_col=0) for mname in model_names]
    stds = [pd.read_csv(os.path.join(root, mname, "std_result_collected.csv"), index_col=0) for mname in model_names]
    final_means = pd.DataFrame(pd.concat(means).round(4))
    final_stds = pd.DataFrame(pd.concat(stds).round(4))
    return final_means, final_stds"""



def get_best_params(project_root = "/local/home/tomap/scAugmentBench", dirname = "architecture-ablation",
                 dname = "ImmHuman", n_runs = 5):
    root = os.path.join(project_root, dirname, dname)
    model_names = os.listdir(root)
    if os.path.exists(os.path.join(root, "final_collected.csv")):
            os.remove(os.path.join(root, "final_collected.csv"))
        
    for mname in model_names:
        tmp = os.path.join(project_root, dirname, dname, mname)
        
        if os.path.exists(os.path.join(tmp, "mean_result_collected.csv")):
            os.remove(os.path.join(tmp, "mean_result_collected.csv"))
        if os.path.exists(os.path.join(tmp, "std_result_collected.csv")):
            os.remove(os.path.join(tmp, "std_result_collected.csv"))
        
        n_seeds = [len(os.listdir(os.path.join(tmp, param_config))) for param_config in os.listdir(os.path.join(tmp))]
        print(f"Min num seeds: {min(n_seeds)}.\nMax num seeds: {max(n_seeds)}.")

        # get mean and std per parameter-config:
        for param in os.listdir(tmp):
            if "mean_result.csv" in os.listdir(os.path.join(tmp, param)):
                os.remove(os.path.join(tmp, param, "mean_result.csv"))
            if "std_result.csv" in os.listdir(os.path.join(tmp, param)):
                os.remove(os.path.join(tmp, param, "std_result.csv"))
            metrics = [pd.read_csv(os.path.join(tmp, param, seed, "evaluation_metrics.csv")) for seed in os.listdir(os.path.join(tmp, param))]
            mean = pd.DataFrame(pd.concat(metrics).mean(0).round(4), columns=[mname + "-" + param]).T
            std = pd.DataFrame(pd.concat(metrics).std(0).round(4), columns=[mname + "-" + param]).T
            mean.to_csv(os.path.join(tmp, param, "mean_result.csv"), index=True)
            std.to_csv(os.path.join(tmp, param, "std_result.csv"), index=True)
        
        # collect across param-configs.
        params = os.listdir(tmp)
        means = [pd.read_csv(os.path.join(tmp, param, "mean_result.csv"), index_col=0) for param in params]
        std = [pd.read_csv(os.path.join(tmp, param, "std_result.csv"), index_col=0) for param in params]
        mean = pd.DataFrame(pd.concat(means).round(4))
        std = pd.DataFrame(pd.concat(std).round(4))
        params = [mname + "-" + p for p in params]
        mean.to_csv(os.path.join(tmp, "mean_result_collected.csv"))
        std.to_csv(os.path.join(tmp, "std_result_collected.csv"))
    
    means = [pd.read_csv(os.path.join(root, mname, "mean_result_collected.csv"), index_col=0) for mname in model_names]
    stds = [pd.read_csv(os.path.join(root, mname, "std_result_collected.csv"), index_col=0) for mname in model_names]
    final_means = pd.DataFrame(pd.concat(means).round(4))
    final_stds = pd.DataFrame(pd.concat(stds).round(4))
    return final_means, final_stds