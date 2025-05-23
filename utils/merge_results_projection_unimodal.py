import os
from glob import glob
import pandas as pd
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler

experiment = 'projection'
data = ['ImmHuman', 'ImmuneAtlas', 'Lung', 'MCA', 'Pancreas', 'PBMC']
models_list = ['SimCLR', 'MoCo', 'SimSiam', 'NNCLR', 'BYOL', 'VICReg', 'BarlowTwins']
pipeline = 'base'
projection = ['True']

def compare_model_architectures(projection, data):
    metrics = ['Isolated labels', 'Leiden NMI', 'Leiden ARI', 'Silhouette label', 'cLISI', 'Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity', 'PCR comparison']

    cols = []
    for p in projection:
        for d in data:
            for metric in metrics:
                cols.append(f"{d}_{p}_{metric}")
                    
    df_list = [pd.DataFrame(index=models_list, columns=cols), pd.DataFrame(index=models_list, columns=cols), pd.DataFrame(index=models_list, columns=cols), pd.DataFrame(index=models_list, columns=cols), pd.DataFrame(index=models_list, columns=cols)]
    for model in models_list:
        for p in projection:
            for d in data:
                PATH = experiment + '/' + d + '/' + model + '/' + pipeline + '/' + f"temp_{p}/"
                result_paths = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.csv'))]

                for i, seed_path in enumerate(result_paths):
                    with open(seed_path, newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            row.pop('Bio conservation')
                            row.pop('Batch correction')
                            row.pop('Total')

                            for k, v in row.items():
                                df_list[i].loc[model, f'{d}_{p}_{k}'] = float(v)
    
    # for i, df in enumerate(df_list):
    #     df.to_csv(f"results/bc/projection/bc_{i}.csv")

    df_final = []
    for df in df_list:
        df_final.append(scale_result(df, projection, data))


    averages = pd.concat([each.stack() for each in df_final],axis=1)\
             .apply(lambda x:x.mean(),axis=1)\
             .unstack()
    stds = pd.concat([each.stack() for each in df_final],axis=1)\
             .apply(lambda x:x.std(),axis=1)\
             .unstack()
    
    remove_cols = list(filter(lambda x: ("Bio conservation" not in x) and ("Batch correction" not in x) and ("Total" not in x), list(averages.columns)))
    averages = averages.drop(remove_cols, axis=1).round(3)
    stds = stds.drop(remove_cols, axis=1).round(3)
    return averages, stds

def scale_result(df, projection, data):
    scaled= pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns, index=df.index)
    for p in projection:
        for d in data:
            biometrics = [f"{d}_{p}_{i}" for i in ['Isolated labels', 'Leiden NMI', 'Leiden ARI', 'Silhouette label', 'cLISI']]
            batchmetrics = [f"{d}_{p}_{i}" for i in ['Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity', 'PCR comparison']]
            scaled[f"{d}_{p}_Batch correction"] = scaled[batchmetrics].mean(1)
            scaled[f"{d}_{p}_Bio conservation"] = scaled[biometrics].mean(1)
            scaled[f"{d}_{p}_Total"] = 0.6 * scaled[f"{d}_{p}_Bio conservation"] + 0.4 * scaled[f"{d}_{p}_Batch correction"]
            df[f"{d}_{p}_Bio conservation"] = scaled[f"{d}_{p}_Bio conservation"].copy()
            df[f"{d}_{p}_Batch correction"] = scaled[f"{d}_{p}_Batch correction"].copy()
            df[f"{d}_{p}_Total"] = scaled[f"{d}_{p}_Total"].copy()
    return df


averages, stds = compare_model_architectures(projection = ['True'], data=data)
dir_path = "results/bc/unimodal_projection/"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    print("Directory created successfully!")
else:
    print("Directory already exists!")
averages.to_csv(f"{dir_path}avg.csv")
stds.to_csv(f"{dir_path}std.csv")
