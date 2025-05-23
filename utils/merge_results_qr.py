import os
from glob import glob
import pandas as pd
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler

experiment = 'augmentation-ablation-vol2'
data = ['PBMC-Multiome', 'Neurips-Multiome']
models_list = ['SimCLR', 'MoCo', 'SimSiam', 'NNCLR', 'BYOL', 'VICReg', 'BarlowTwins']
pipeline = 'base'

combine = ['add', 'concat', 'clip']
projection = ['True', 'False']


def compare_model_architectures(projection, combine, data):
    assert len(projection) == 1

    metrics = ['Macro-F1', 'Accuracy']

    cols = []
    for c in combine:
        for p in projection:
            for r, onlyRNA in zip(['', '-onlyRNA'], [False, True]):
                for d in data:
                    for metric in metrics:
                        cols.append(f"{d}_{c}_{p}_{metric}_{onlyRNA}")
    result = pd.DataFrame(index=models_list, columns=cols)
    std = pd.DataFrame(index=models_list, columns=cols)

    for model in models_list:
        for c in combine:
            for p in projection:
                for r, onlyRNA in zip(['', '-onlyRNA'], [False, True]):
                    for d in data:
                        PATH = experiment + '/' + d + '/' + model + '_qr' + '/' + pipeline + '/' + f"integrate_{c}_RNA_False_projection_{p}/"
                        result_paths = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], f'qr{r}-results.csv'))]
                        
                        tmp_df = pd.DataFrame(columns=metrics)
                        for i, seed_path in enumerate(result_paths):
                            with open(seed_path, newline='') as csvfile:
                                reader = csv.DictReader(csvfile)
                                vals = []
                                for row in reader:
                                    vals.append(float(row['0']))
                                tmp_df.loc[i] = [vals[0], vals[1]]
                                
                        for k, v in tmp_df.mean(axis=0).to_dict().items():
                            result.loc[model, f'{d}_{c}_{p}_{k}_{onlyRNA}'] = v
                        
                        for k, v in tmp_df.std(axis=0).to_dict().items():
                            std.loc[model, f'{d}_{c}_{p}_{k}_{onlyRNA}'] = v
    result, std = result.astype(float).round(3), std.astype(float).round(3)
    print(result)
    return result, std


def compare_model_architectures_new(projection, combine, data):
    assert len(projection) == 1

    metrics = ['Macro-F1', 'Accuracy']

    cols = []
    for c in combine:
        for p in projection:
            for r, onlyRNA in zip(['', '-onlyRNA'], [False, True]):
                for d in data:
                    for metric in metrics:
                        cols.append(f"{d}_{c}_{p}_{metric}_{onlyRNA}")
    result = pd.DataFrame(index=models_list, columns=cols)
    std = pd.DataFrame(index=models_list, columns=cols)

    for model in models_list:
        for c in combine:
            for p in projection:
                for r, onlyRNA in zip(['', '-onlyRNA'], [False, True]):
                    for d in data:
                        PATH = experiment + '/' + d + '/' + model + '_qr' + '/' + pipeline + '/' + f"integrate_{c}_RNA_False_projection_{p}/"
                        result_paths = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], f'qr{r}-results.csv'))]
                        
                        tmp_df = pd.DataFrame(columns=metrics)
                        for i, seed_path in enumerate(result_paths):
                            with open(seed_path, newline='') as csvfile:
                                reader = csv.DictReader(csvfile)
                                vals = []
                                for row in reader:
                                    vals.append(float(row['0']))
                                tmp_df.loc[i] = [vals[0], vals[1]]
                                
                        for k, v in tmp_df.mean(axis=0).to_dict().items():
                            result.loc[model, f'{d}_{c}_{p}_{k}_{onlyRNA}'] = v
                        
    result = result.astype(float).round(3)
    print(result)
    return result

plot = 1
if plot == 1:   
    # Compare backbone concat for only RNA and both modalities 
    averages, stds = compare_model_architectures(projection = ['False'], combine=['concat'], data=['PBMC-Multiome', 'Neurips-Multiome'])
    dir_path = "results/qr/qr_backbone_concat/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")
    # averages.to_csv(f"{dir_path}avg.csv")
    # stds.to_csv(f"{dir_path}std.csv")
    averages.to_csv(f"results/qr/new/avg.csv")
    stds.to_csv(f"results/qr/new/std.csv")

elif plot == 2:   
    # Compare backbone concat for only RNA and both modalities 
    averages = compare_model_architectures_new(projection = ['False'], combine=['concat'], data=['PBMC-Multiome', 'Neurips-Multiome'])
    dir_path = "results/qr/qr_backbone_concat_other/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")
    averages.to_csv(f"{dir_path}avg.csv")

