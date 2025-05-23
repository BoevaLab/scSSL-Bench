from scipy.sparse import csr_matrix
import pandas as pd
import os
import warnings
import pickle
from scipy.sparse import save_npz, load_npz

from data.dataset_collection import *


"""
This class defines a database which stores necessary information for embeddings:
model info, model run, augmentation information.
"""
class Embedding_Database():

    def __init__(self, data_location=None,):
        assert data_location is not None, "Embedding_Database: data location must be specified!"
        self.keys = ['model_name', 'run_i', 
                    'n_epochs', 'hidden_dim_1', 'hidden_dim_2', 'lat_dim', # model config
                    'mask_percentage', 'apply_mask_prob', 
                    'noise_percentage', 'sigma', 'apply_noise_prob', 
                    'swap_percentage', 'apply_swap_prob', 
                    'cross_percentage', 'apply_cross_prob',
                    'change_percentage', 'apply_mutation_prob',
                    'apply_bbknn', 'bbknn_alpha', 'bbknn_neighbors',
                    'apply_mnn', 'mnn_alpha','mnn_neighbors']
        self.augname = data_location.split("_")[-2] # index -1 is dname
        self.dname = data_location.split("_")[-1]
        if os.path.exists(data_location) and len(os.listdir(data_location)) >= 2:
            warnings.warn("loading existing embedding database")
            self.path = data_location
            self.load_data()
        else:
            self.embeddings = []
            self.configs = pd.DataFrame(columns=self.keys)
            os.makedirs(data_location, exist_ok=True)
            self.path = data_location

    def load_data(self):
        #with open(os.path.join(self.path, "configs.idx"), 'wb') as handle:
        self.configs = pd.read_csv(os.path.join(self.path, "configs.csv"), index_col=0)
        n_embeddings = len(self.configs)
        self.embeddings = []
        for i in range(n_embeddings):
            self.embeddings.append(load_npz(os.path.join(self.path, f"emb_{i}.npz")))
    

    def store_database(self):
        #with open(os.path.join(self.path, "configs.idx"), 'wb') as handle:
        #    pickle.dump(self.configs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.configs.to_csv(os.path.join(self.path, "configs.csv"))
        for i, e in zip(range(len(self.embeddings)), self.embeddings):
            save_npz(os.path.join(self.path, f"emb_{i}.npz"), e)


    def add_entry(self, model_name, run_index, model_config, aug_config, eval_metrics, embedding):
        row = {'model_name': model_name, 'run_i': run_index}
        row.update(model_config)
        if aug_config is not None:
            row.update(aug_config)
        row = pd.DataFrame.from_dict([row])
        self.configs = pd.concat([
                self.configs,
                row]#, columns=self.keys)]
                #pd.DataFrame([model_name, run_index, model_config, aug_config, eval_metrics], columns=self.keys)]
           ).reset_index(drop=True)
        embedding = csr_matrix(embedding)
        self.embeddings.append(embedding)

#db = Embedding_Database("test_database/")


"""
WARNING: Might be confusing, but this function combines actual datasets, not embedding datasets.
TODO: Move into a preprocessing_module.py file at a later stage.
"""
def create_dataset_combination(dname, hvg, augmentation_args_naive, augmentation_args_graph_based):
    naive_flag = True in [a for a in augmentation_args_naive.values() if type(a) == bool]
    mnn_flag = augmentation_args_graph_based['apply_mnn']
    bbknn_flag = augmentation_args_graph_based['apply_bbknn']

    val_data = ClaireDataset(data_dir=f"../own_model/CLAIRE-data/{dname}", mode='val',select_hvg=hvg,
                     knn=5, alpha=0.5, augment_set=['int', 'exc'], bbknn_exchange=True, bbknn_fair=True)
    
    datasets = []
    if naive_flag:
        data = NaiveDataset(
            adata=val_data.adata,
            obs_label_colname='CellType',
            transform=True,
            args_transformation=augmentation_args_naive
            )
        datasets.append(data)
    if mnn_flag:
        data = ClaireDataset(data_dir=f"../own_model/CLAIRE-data/{dname}", mode='train',select_hvg=hvg,
                     knn=augmentation_args_graph_based['mnn_neighbors'], alpha=augmentation_args_graph_based['mnn_alpha'], augment_set=['int', 'exc'], bbknn_exchange=False, anchor_path=f"../own_model/CLAIRE-data/{dname}")
        datasets.append(data)
    if bbknn_flag:
        data = ClaireDataset(data_dir=f"../own_model/CLAIRE-data/{dname}", mode='train',select_hvg=hvg, anchor_path=f"../own_model/CLAIRE-data/{dname}",
                     knn=augmentation_args_graph_based['bbknn_neighbors'], alpha=augmentation_args_graph_based['bbknn_alpha'], augment_set=['int', 'exc'], bbknn_exchange=True, bbknn_fair=True)
        datasets.append(data)
    
    datasets = OwnCollection('train', None, datasets)
    
    return datasets, val_data