from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection
import pandas as pd
import pickle
from data.dataset_collection import *
import numpy as np
import scanpy as sc


# TODO: Handle multi-run case, as was done in the master thesis.
# WARNING! At the moment, only the single-run-case can be handled.
"""
This class provides a wrapper for our evaluation.

INFO: Further functionality (multi-run-evaluation, normalization and plotting) will be added.
"""
class EvaluationModule():

    def __init__(self, db, adata, model_indices, augmentation_name=None):
        self.ad = adata
        self.num_models = len(model_indices)
        #embeddings = db.embeddings[model_indices]
        self.model_names = list(db.configs.iloc[model_indices]['model_name'])
        self.emb_keys = [db.configs.iloc[i]['model_name'] + f"-{db.configs.iloc[i]['run_i']}" for i in model_indices]
        print(self.emb_keys)
        for i in model_indices:
            self.ad.obsm[db.configs.iloc[i]['model_name'] + f"-{db.configs.iloc[i]['run_i']}"] = db.embeddings[i].toarray()
            #self.ad.obsm[db.configs.iloc[i]['model_name']] = db.embeddings[i].toarray()
            #emb_keys.append(db.configs.iloc[i]['model_name'])
        self.augname = db.augname
        self.dname = db.dname

        self.biometrics = BioConservation(isolated_labels=True, nmi_ari_cluster_labels_leiden=True, nmi_ari_cluster_labels_kmeans=False, silhouette_label=False, clisi_knn=False)
        self.batchmetrics = BatchCorrection(graph_connectivity=True, kbet_per_label=True, ilisi_knn=False, pcr_comparison=False, silhouette_batch=False)
        
        self.results = None
        self.means = None
        self.stds = None

    def run_evaluation(self):
        bm = Benchmarker(
                self.ad,
                batch_key="batchlb",
                label_key="CellType",
                embedding_obsm_keys=self.emb_keys,
                bio_conservation_metrics=self.biometrics,
                batch_correction_metrics=self.batchmetrics,
            )
        bm.benchmark()
        a = bm.get_results(False, True)
        self.results = a[:self.num_models]#len(self.db.configs)]
        #train_metrics.index = pd.Index([e_id], name="Embedding")

    def unify_runs(self):
        q = self.display_table()
        q["Model"] = self.model_names
        self.means = q.groupby(["Model"]).mean()
        self.stds = q.groupby(["Model"]).std()
    
    def print_latex_table(self):
        assert self.means is not None and self.stds is not None, "Must call .unify_runs() before printing unified tables."
        print("Metrics:")
        print(self.means.sort_index().round(3).to_latex(index=True, float_format="${:.3f}$".format,))
        print("-----")
        print("STD:")
        print(self.stds.sort_index().round(3).to_latex(index=True, float_format="${:.3f}$".format,))
    
    def print_combined_latex_table(self):
        assert self.means is not None and self.stds is not None, "Must call .unify_runs() before printing unified tables."
        # Combine means and stds into a single DataFrame with the format "mean ± std"
        combined = self.means.sort_index().round(3).astype(str) + " ± " + self.stds.sort_index().round(3).astype(str)        
        # Print the combined DataFrame in LaTeX format
        print("Metrics (Mean ± STD):")
        print(combined.to_latex(index=True, escape=False))
    
    def plot_emb_umaps(self):
        """
        Get the model name (and the dataset-name?) and then plot the umaps of all embeddings
        """
        n_cells = len(self.ad)
        for model_name in self.emb_keys:
            ad = self.ad.copy()
            sc.pp.neighbors(ad, use_rep=model_name)
            sc.tl.umap(ad)
            sc.pl.umap(ad, show=False, color=['CellType', 'batchlb'], save=f"_{self.dname}_{model_name}_{self.augname}.png", size = 90000 / n_cells)
            sc.pl.umap(ad, show=False, color=['CellType', 'batchlb'], save=f"_{self.dname}_{model_name}_{self.augname}.svg", size = 90000 / n_cells)
    
    def plot_leiden(self):
        """
        if args.leiden:
            random_seed, res = get_best_leiden_config(tmp_emb[idx])
            sc.tl.leiden(tmp_emb[idx], resolution=res, random_state=random_seed)
            sc.pl.umap(tmp_emb[idx], show=False, color='leiden', ax=axes[2], size=size)
        """
        pass

    def display_table(self):
        assert self.results is not None, "Results are missing. Must call run_evaluation() first."
        return self.results.round(3)


#em = EvaluationModule(db, val_data, model_indices=[0,1])
#a = AugCombiner("test_database", "NeftelMulti")
