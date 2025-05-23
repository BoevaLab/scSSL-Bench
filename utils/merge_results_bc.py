import os
from glob import glob
import pandas as pd
import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler

experiment = 'augmentation-ablation-vol1'
data = ['PBMC-Multiome', 'Neurips-Multiome']
models_list = ['SimCLR', 'MoCo', 'SimSiam', 'NNCLR', 'BYOL', 'VICReg', 'BarlowTwins']
pipeline = 'base'

combine = ['add', 'concat', 'clip']
projection = ['True', 'False']

def compare_model_architectures(projection, combine, data):
    metrics = ['Isolated labels', 'Leiden NMI', 'Leiden ARI', 'Silhouette label', 'cLISI', 'Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity', 'PCR comparison']

    cols = []
    for c in combine:
        for p in projection:
            for d in data:
                for metric in metrics:
                    cols.append(f"{d}_{c}_{p}_{metric}")
                    
    df_list = [pd.DataFrame(index=models_list, columns=cols), pd.DataFrame(index=models_list, columns=cols), pd.DataFrame(index=models_list, columns=cols), pd.DataFrame(index=models_list, columns=cols), pd.DataFrame(index=models_list, columns=cols)]
    for model in models_list:
        for c in combine:
            for p in projection:
                for d in data:
                    PATH = experiment + '/' + d + '/' + model + '/' + pipeline + '/' + f"integrate_{c}_RNA_False_projection_{p}/"
                    result_paths = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.csv'))]

                    for i, seed_path in enumerate(result_paths):
                        with open(seed_path, newline='') as csvfile:
                            reader = csv.DictReader(csvfile)
                            for row in reader:
                                row.pop('Bio conservation')
                                row.pop('Batch correction')
                                row.pop('Total')

                                for k, v in row.items():
                                    df_list[i].loc[model, f'{d}_{c}_{p}_{k}'] = float(v)
    
    for i, df in enumerate(df_list):
        df.to_csv(f"results/bc/new/bc_{i}.csv")

    exit()

    df_final = []
    for df in df_list:
        df_final.append(scale_result(df, projection, combine, data))


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

def compare_model_architectures_new(projection, combine, data):
    metrics = ['Isolated labels', 'Leiden NMI', 'Leiden ARI', 'Silhouette label', 'cLISI', 'Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity', 'PCR comparison']

    cols = []
    for c in combine:
        for p in projection:
            for d in data:
                for metric in metrics:
                    cols.append(f"{d}_{c}_{p}_{metric}")
                    
    df = pd.DataFrame(index=models_list+["Concerto", "CLIP + Teacher-Student Between Modalities", "CLIP + Teacher-Student All Pairs", "scCLIP", "totalVI"], columns=cols)
    for model in models_list:
        for c in combine:
            for p in projection:
                for d in data:
                    PATH = experiment + '/' + d + '/' + model + '/' + pipeline + '/' + f"integrate_{c}_RNA_False_projection_{p}/"
                    result_paths = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.csv'))]

                    for i, seed_path in enumerate(result_paths):
                        with open(seed_path, newline='') as csvfile:
                            reader = csv.DictReader(csvfile)
                            for row in reader:
                                row.pop('Bio conservation')
                                row.pop('Batch correction')
                                row.pop('Total')

                                for k, v in row.items():
                                    df.loc[model, f'{d}_{c}_{p}_{k}'] = float(v)

    df.loc["CLIP + Teacher-Student Between Modalities", :] = {'PBMC-Multiome_concat_False_Isolated labels':0.6286,
                                                              'PBMC-Multiome_concat_False_Leiden NMI': 0.82,
                                                              'PBMC-Multiome_concat_False_Leiden ARI': 0.82,
                                                              'PBMC-Multiome_concat_False_Silhouette label': 0.5483,
                                                              'PBMC-Multiome_concat_False_cLISI': 0.9997,
                                                              'PBMC-Multiome_concat_False_Silhouette batch': 0.9446,
                                                              'PBMC-Multiome_concat_False_iLISI': 0.3044, 
                                                              'PBMC-Multiome_concat_False_KBET': 0.4541,
                                                              'PBMC-Multiome_concat_False_Graph connectivity': 0.8616,
                                                              'PBMC-Multiome_concat_False_PCR comparison': 0.6175,
                                                              'Neurips-Multiome_concat_False_Isolated labels':0.5305215,
                                                              'Neurips-Multiome_concat_False_Leiden NMI': 0.7824953161931057,
                                                              'Neurips-Multiome_concat_False_Leiden ARI': 0.8222508350229699,
                                                              'Neurips-Multiome_concat_False_Silhouette label': 0.5757077932357788,
                                                              'Neurips-Multiome_concat_False_cLISI': 0.9989798367023468,
                                                              'Neurips-Multiome_concat_False_Silhouette batch': 0.8892552,
                                                              'Neurips-Multiome_concat_False_iLISI': 0.2234760197726163, 
                                                              'Neurips-Multiome_concat_False_KBET': 0.12197307115574,
                                                              'Neurips-Multiome_concat_False_Graph connectivity': 0.8352386086964091,
                                                              'Neurips-Multiome_concat_False_PCR comparison': 0.4810632929918455}
    
    df.loc["CLIP + Teacher-Student All Pairs", :] = {'PBMC-Multiome_concat_False_Isolated labels':0.6332726,
                                                     'PBMC-Multiome_concat_False_Leiden NMI': 0.8333278259672753,
                                                     'PBMC-Multiome_concat_False_Leiden ARI': 0.7514766240518961,
                                                     'PBMC-Multiome_concat_False_Silhouette label': 0.6412781178951263,
                                                     'PBMC-Multiome_concat_False_cLISI': 0.9999999829701016,
                                                     'PBMC-Multiome_concat_False_Silhouette batch': 0.912345,
                                                     'PBMC-Multiome_concat_False_iLISI': 0.3291448865618024, 
                                                     'PBMC-Multiome_concat_False_KBET': 0.1146408776896053,
                                                     'PBMC-Multiome_concat_False_Graph connectivity': 0.9374905267461542,
                                                     'PBMC-Multiome_concat_False_PCR comparison': 0.3406798481878996,
                                                     'Neurips-Multiome_concat_False_Isolated labels':0.5391785,
                                                     'Neurips-Multiome_concat_False_Leiden NMI': 0.7907218873261689,
                                                     'Neurips-Multiome_concat_False_Leiden ARI': 0.8309767907974978,
                                                     'Neurips-Multiome_concat_False_Silhouette label': 0.593246266245842,
                                                     'Neurips-Multiome_concat_False_cLISI': 0.9991274575392404,
                                                     'Neurips-Multiome_concat_False_Silhouette batch': 0.9053739,
                                                     'Neurips-Multiome_concat_False_iLISI': 0.2082948684692382, 
                                                     'Neurips-Multiome_concat_False_KBET': 0.0887305529885838,
                                                     'Neurips-Multiome_concat_False_Graph connectivity': 0.8336686173111079,
                                                     'Neurips-Multiome_concat_False_PCR comparison': 0.4046049628208698}
    
    df.loc["scCLIP", :] = {'PBMC-Multiome_concat_False_Isolated labels':0.5464631,
                           'PBMC-Multiome_concat_False_Leiden NMI': 0.6241284265995152,
                           'PBMC-Multiome_concat_False_Leiden ARI': 0.4847607574302772,
                           'PBMC-Multiome_concat_False_Silhouette label': 0.5605474896728992,
                           'PBMC-Multiome_concat_False_cLISI': 0.9982563427516392,
                           'PBMC-Multiome_concat_False_Silhouette batch': 0.9534004,
                           'PBMC-Multiome_concat_False_iLISI': 0.2786969797951834, 
                           'PBMC-Multiome_concat_False_KBET': 0.0023292275564002,
                           'PBMC-Multiome_concat_False_Graph connectivity': 0.8200502492970527,
                           'PBMC-Multiome_concat_False_PCR comparison': 0.0,
                           'Neurips-Multiome_concat_False_Isolated labels':0.5223073,
                           'Neurips-Multiome_concat_False_Leiden NMI': 0.5485740416253094,
                           'Neurips-Multiome_concat_False_Leiden ARI': 0.3532760138981519,
                           'Neurips-Multiome_concat_False_Silhouette label': 0.5594623684883118,
                           'Neurips-Multiome_concat_False_cLISI': 0.9963764548301696,
                           'Neurips-Multiome_concat_False_Silhouette batch': 0.8235544,
                           'Neurips-Multiome_concat_False_iLISI': 0.0777017203244296, 
                           'Neurips-Multiome_concat_False_KBET': 0.0270451084496028,
                           'Neurips-Multiome_concat_False_Graph connectivity': 0.6331223790167401,
                           'Neurips-Multiome_concat_False_PCR comparison': 0.0}
    
    df.loc["totalVI", :] = {'PBMC-Multiome_concat_False_Isolated labels':0.5658823,
                            'PBMC-Multiome_concat_False_Leiden NMI': 0.7365020961355562,
                            'PBMC-Multiome_concat_False_Leiden ARI': 0.4974151424137608,
                            'PBMC-Multiome_concat_False_Silhouette label': 0.5722888633608818,
                            'PBMC-Multiome_concat_False_cLISI': 0.9999999829701016,
                            'PBMC-Multiome_concat_False_Silhouette batch': 0.91766095,
                            'PBMC-Multiome_concat_False_iLISI': 0.0872393335614885, 
                            'PBMC-Multiome_concat_False_KBET': 0.0056887169965836,
                            'PBMC-Multiome_concat_False_Graph connectivity': 0.9497261795679192,
                            'PBMC-Multiome_concat_False_PCR comparison': 0.0,
                            'Neurips-Multiome_concat_False_Isolated labels':0.5546399,
                            'Neurips-Multiome_concat_False_Leiden NMI': 0.6437890935175837,
                            'Neurips-Multiome_concat_False_Leiden ARI': 0.3718398391693006,
                            'Neurips-Multiome_concat_False_Silhouette label': 0.5697183907032013,
                            'Neurips-Multiome_concat_False_cLISI': 0.999999980131785,
                            'Neurips-Multiome_concat_False_Silhouette batch': 0.8752989,
                            'Neurips-Multiome_concat_False_iLISI': 0.0079804117029363, 
                            'Neurips-Multiome_concat_False_KBET': 0.0012519703151398,
                            'Neurips-Multiome_concat_False_Graph connectivity': 0.8717301881984453,
                            'Neurips-Multiome_concat_False_PCR comparison': 0.0}
           
    df.loc["Concerto", :] = {'PBMC-Multiome_concat_False_Isolated labels':0.5441334,
                            'PBMC-Multiome_concat_False_Leiden NMI': 0.8687780145290134,
                            'PBMC-Multiome_concat_False_Leiden ARI': 0.8968530240789663,
                            'PBMC-Multiome_concat_False_Silhouette label': 0.5570280030369759,
                            'PBMC-Multiome_concat_False_cLISI': 0.999999965940203,
                            'PBMC-Multiome_concat_False_Silhouette batch': 0.9451909,
                            'PBMC-Multiome_concat_False_iLISI': 0.2727957453046526, 
                            'PBMC-Multiome_concat_False_KBET': 0.0360299387423919,
                            'PBMC-Multiome_concat_False_Graph connectivity': 0.9245870397819798,
                            'PBMC-Multiome_concat_False_PCR comparison': 0.0,
                            'Neurips-Multiome_concat_False_Isolated labels':0.50804216,
                            'Neurips-Multiome_concat_False_Leiden NMI': 0.510512082794047,
                            'Neurips-Multiome_concat_False_Leiden ARI': 0.5206091149425239,
                            'Neurips-Multiome_concat_False_Silhouette label': 0.5128912050276995,
                            'Neurips-Multiome_concat_False_cLISI': 0.9687945346037546,
                            'Neurips-Multiome_concat_False_Silhouette batch': 0.9616242,
                            'Neurips-Multiome_concat_False_iLISI': 0.2122701731595126, 
                            'Neurips-Multiome_concat_False_KBET': 0.0852899166299665,
                            'Neurips-Multiome_concat_False_Graph connectivity': 0.7643347148772965,
                            'Neurips-Multiome_concat_False_PCR comparison': 0.6034181271850003}

    print(df)
    averages = scale_result(df, projection, combine, data)
    
    remove_cols = list(filter(lambda x: ("Bio conservation" not in x) and ("Batch correction" not in x) and ("Total" not in x), list(averages.columns)))
    averages = averages.drop(remove_cols, axis=1).round(3)
    return averages

def scale_result(df, projection, combine, data):
    scaled= pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns, index=df.index)
    for c in combine:
        for p in projection:
            for d in data:
                biometrics = [f"{d}_{c}_{p}_{i}" for i in ['Isolated labels', 'Leiden NMI', 'Leiden ARI', 'Silhouette label', 'cLISI']]
                batchmetrics = [f"{d}_{c}_{p}_{i}" for i in ['Silhouette batch', 'iLISI', 'KBET', 'Graph connectivity', 'PCR comparison']]
                scaled[f"{d}_{c}_{p}_Batch correction"] = scaled[batchmetrics].mean(1)
                scaled[f"{d}_{c}_{p}_Bio conservation"] = scaled[biometrics].mean(1)
                scaled[f"{d}_{c}_{p}_Total"] = 0.6 * scaled[f"{d}_{c}_{p}_Bio conservation"] + 0.4 * scaled[f"{d}_{c}_{p}_Batch correction"]
                df[f"{d}_{c}_{p}_Bio conservation"] = scaled[f"{d}_{c}_{p}_Bio conservation"].copy()
                df[f"{d}_{c}_{p}_Batch correction"] = scaled[f"{d}_{c}_{p}_Batch correction"].copy()
                df[f"{d}_{c}_{p}_Total"] = scaled[f"{d}_{c}_{p}_Total"].copy()
    return df

plot = 2
if plot == 1:   
    # Compare CLIP, Add, Concat for backbone  
    averages, stds = compare_model_architectures(projection = ['False'], combine=combine, data=['Neurips-Multiome'])
    dir_path = "results/bc/avg_compare_integration_projection_False/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")
    averages.to_csv(f"{dir_path}avg.csv")
    stds.to_csv(f"{dir_path}std.csv")

elif plot == 2:
    # Compare Concat for backbone and projection
    averages, stds = compare_model_architectures(projection = ['False'], combine=['concat'], data=data)
    dir_path = "results/bc/avg_compare_backbone_projection/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")
    averages.to_csv(f"{dir_path}avg.csv")
    stds.to_csv(f"{dir_path}std.csv")

elif plot == 3:
    # Compare Concat for backbone and projection
    averages = compare_model_architectures_new(projection = ['False'], combine=['concat'], data=data)
    dir_path = "results/bc/avg_compare_backbone_projection_with_other/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")
    averages.to_csv(f"{dir_path}avg.csv")

# result2 = compare_integrations(projection = ['True'])
