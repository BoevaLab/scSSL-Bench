import logging
import os

import pandas as pd

"""import torch
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose"""
import numpy as np
import random
#import lightning as pl

import yaml
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from evaluator import collect_runs, unify_table, scale_table


parser = argparse.ArgumentParser(description='Re-Evaluate with SCIB')
# "/local/home/tomap/scAugmentBench//augmentation-ablation-vol9/MCA"
parser.add_argument('--dname_root', default='./', type=str,
                        help='Where to evaluate.')
parser.add_argument('--dataset', default='immune', type=str,
                        help='yaml-dataset fname.')
parser.add_argument('--dname', default='MCA', type=str,
                        help='actual dataset dirname.')
parser.add_argument('--mname', default='simclr', type=str,
                        help='name of model (lower case), if needed for functionality.')


args = parser.parse_args()

_LOGGER = logging.getLogger(__name__)
with open("/local/home/tomap/scAugmentBench/conf/augmentation/clear_pipeline.yaml") as stream:
    cfg_aug = yaml.safe_load(stream)
#with open("/local/home/tomap/scAugmentBench//conf/data/pbmc.yaml") as stream:
with open(f"/local/home/tomap/scAugmentBench/conf/data/{args.dataset}.yaml") as stream:
    cfg_data = yaml.safe_load(stream)
with open(f"/local/home/tomap/scAugmentBench/conf/model/{args.mname}.yaml") as stream:
    cfg_model = yaml.safe_load(stream)


cfg = {}
cfg['data'] = cfg_data
cfg['data']['n_hvgs'] = 4000
cfg['model'] = cfg_model
cfg['model']['in_dim'] = 4000
cfg["data"]["holdout_batch"] = None
cfg['augmentation'] = cfg_aug

sns.set_theme(style="ticks", rc={"axes.labelsize": 20,   # Axis label font size
                                 "xtick.labelsize": 18,  # X-axis tick label font size
                                 "ytick.labelsize": 18,  # Y-axis tick label font size
                                 "legend.fontsize": 20,   # Legend font size
                                 "font.size": 20, 
                                 "axes.titlesize": 20,
                                 "font.family": "serif"})

from main import load_data

train, val, adata = load_data(cfg)


###########################################################################3
###########################################################################3
###########################################################################3
###########################################################################3
###########################################################################3


######################################################
# Representation-Dim Figures
######################################################

import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from evaluator import *


def get_results(path):
    if "mean_result.csv" in os.listdir(path):
        os.remove(os.path.join(path, "mean_result.csv"))
    if "std_result.csv" in os.listdir(path):
        os.remove(os.path.join(path, "std_result.csv"))
    seeds = os.listdir(path)
    metrics = []
    for seed in seeds:
        if os.path.exists(os.path.join(path, seed, "evaluation_metrics.csv")):
            metrics.append(pd.read_csv(os.path.join(path, seed, "evaluation_metrics.csv")))
    return metrics

def get_bigger_table(project_root = "/cluster/work/boeva/tomap/scAugmentBench-main", dirname = "architecture-ablation",
                 dname = "MCA"):
    root = os.path.join(project_root, dirname, dname)
    model_names = os.listdir(root)

    if os.path.exists(os.path.join(root, "final_collected.csv")):
            os.remove(os.path.join(root, "final_collected.csv"))
    
    ls = []
    for mname in model_names:
        tmp = os.path.join(project_root, dirname, dname, mname)
        
        if os.path.exists(os.path.join(tmp, "mean_result_collected.csv")):
            os.remove(os.path.join(tmp, "mean_result_collected.csv"))
        if os.path.exists(os.path.join(tmp, "std_result_collected.csv")):
            os.remove(os.path.join(tmp, "std_result_collected.csv"))
        
        n_seeds = [len(os.listdir(os.path.join(tmp, param_config))) for param_config in os.listdir(os.path.join(tmp))]
        print(f"Min num seeds: {min(n_seeds)}.\nMax num seeds: {max(n_seeds)}.")

        # get mean and std per parameter-config:
        representation_dims = os.listdir(tmp)
        hyperparam_strings = []
        for param in representation_dims:
            if mname in ["BarlowTwins", "SimSiam", "NNCLR", "BYOL", "SimCLR", "MoCo"]:
                for param_2 in os.listdir(os.path.join(tmp, param)):
                    metrics = []
                    metrics.extend(get_results(os.path.join(tmp, param, param_2)))
                    hyperparam_strings.append("-".join([param, param_2]))
            elif mname == "VICReg":
                for param_2 in os.listdir(os.path.join(tmp, param)):
                    for lamda in os.listdir(os.path.join(tmp, param, param_2)):
                        metrics = []
                        metrics.extend(get_results(os.path.join(tmp, param, param_2, lamda)))
            else:
                tmp = os.path.join(tmp, param)
                print(tmp)
                metrics = get_results(os.path.join(tmp))
                print(metrics)
                hyperparam_strings.append(param)
            
            if len(metrics) > 0:
                for i in range(len(metrics)):
                    metrics[i]['Model-Architecture'] = mname
                    print(param)
                    metrics[i]['hidden_dim'] = int(param)
                
                df = pd.DataFrame(pd.concat(metrics).round(4))#, columns=[seed for seed in os.listdir(os.path.join(tmp, param))])
                ls.append(df)
            else:
                print(f"Metrics were empty for {mname, param}")
    return pd.concat(ls)

print("RUNNING df-getter")

df = get_bigger_table(project_root="/local/home/tomap/scAugmentBench/ablation-db-2", dirname="architecture-ablation",
                               dname=args.dname)

print("FINISHED df-getter")


def plot_bigger_table_dim_total(collected_df, key="Batch correction", key_x="hidden_dim", x_label="alpha", palette=None, categories=None):
    tmp = collected_df.copy()
    collected_df_scaled = scale_table(tmp.drop(columns=[key_x, 'Model-Architecture']))
    collected_df_scaled[key_x] = collected_df[key_x]
    collected_df_scaled['Model-Architecture'] = collected_df['Model-Architecture']

    plt.clf()
    #ax = sns.catplot(x="alpha", y=key, hue="Model-Architecture", kind="point", capsize=.15, data=collected_df, aspect=1.5, errwidth=0.5,).set(title=f'{name} augmentation')
    ax = sns.catplot(x=key_x, y=key, hue="Model-Architecture", kind="point", capsize=.15, data=collected_df_scaled, aspect=1.5, errwidth=0.5,palette=palette, hue_order=categories).set(xlabel=x_label)#,title=f'Hidden-Dim-Choice'
    ax._legend.set_title("")
    ax._legend.remove()
    ax.savefig(f"/local/home/tomap/scAugmentBench/representation-dimension-ablation-{key}.png")
    ax.savefig(f"/local/home/tomap/scAugmentBench/representation-dimension-ablation-{key}.svg")
    return ax


def plot_bigger_table_scatter(collected_df, name):
    tmp = collected_df.copy()
    collected_df_scaled = scale_table(tmp.drop(columns=['alpha', 'Model-Architecture']))
    collected_df_scaled['alpha'] = collected_df['alpha']
    collected_df_scaled['Model-Architecture'] = collected_df['Model-Architecture']

    ax = sns.scatterplot(data=collected_df, x="Batch correction", y="Bio conservation", hue="alpha", style="Model-Architecture",)
    ax.set(title=f'{name} augmentation')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
    ax.savefig(f"/cluster/work/boeva/tomap/output/{name}-alphas-scatter.png")

"""df = get_bigger_table(dirname="dimension-ablation-vol2",
                               dname="MCA")"""


import glasbey
labels = np.unique(np.array(df["Model-Architecture"]))
categories=labels
#palette = sns.color_palette("tab10",len(labels))
palette = glasbey.create_palette(palette_size=len(labels), colorblind_safe=True, cvd_severity=50)
print(palette)
label_colors = dict(zip(labels, palette))

"""
df = df[~pd.isna(df['hidden_dim2'])]
df = df[df['hidden_dim']==128]
df['hidden_dim2'] = df['hidden_dim2'].astype(int)
"""
key_x = "hidden_dim"


#df = df.loc[(df['hidden_dim'] > 8) & (df['hidden_dim'] <= 64)]
df = df.loc[(df['hidden_dim'] <= 64)]

ax = plot_bigger_table_dim_total(df, key="Bio conservation", key_x=key_x, x_label="Representation Dimension", palette=palette, categories=categories)
ax = plot_bigger_table_dim_total(df, key="Batch correction", key_x=key_x, x_label="Representation Dimension", palette=palette, categories=categories)
ax = plot_bigger_table_dim_total(df, key="Total", key_x=key_x, x_label="Representation Dimension", palette=palette, categories=categories)


###########################################################################3
###########################################################################3
###########################################################################3
###########################################################################3
###########################################################################3

##
##              Projector Ablation
##

def get_bigger_table_cell_2(project_root = "/local/home/tomap/scAugmentBench/ablation-db-2", dirname = "architecture-ablation",
                 dname = "MCA"):
    root = os.path.join(project_root, dirname, dname)
    model_names = os.listdir(root)
    if os.path.exists(os.path.join(root, "final_collected.csv")):
            os.remove(os.path.join(root, "final_collected.csv"))
    
    ls = []
    for mname in model_names:
        tmp = os.path.join(project_root, dirname, dname, mname)
        
        if os.path.exists(os.path.join(tmp, "mean_result_collected.csv")):
            os.remove(os.path.join(tmp, "mean_result_collected.csv"))
        if os.path.exists(os.path.join(tmp, "std_result_collected.csv")):
            os.remove(os.path.join(tmp, "std_result_collected.csv"))
        
        n_seeds = [len(os.listdir(os.path.join(tmp, param_config))) for param_config in os.listdir(os.path.join(tmp))]
        print(f"Min num seeds: {min(n_seeds)}.\nMax num seeds: {max(n_seeds)}.")

        # get mean and std per parameter-config:
        representation_dims = os.listdir(os.path.join(tmp, "64"))
        hyperparam_strings = []
        for param in representation_dims:
            print(param)
            if mname == "VICReg":
                for lamda in os.listdir(os.path.join(tmp, "64", param)):
                    metrics = []
                    metrics.extend(get_results(os.path.join(tmp, "64", param, lamda)))
            else:
                metrics = get_results(os.path.join(tmp, "64", param))
                hyperparam_strings.append(param)
            
            if len(metrics) != 0:
                for i in range(len(metrics)):
                    metrics[i]['Model-Architecture'] = mname
                    metrics[i]['Projector Output'] = int(param)
                #if metrics is not None and len(metrics) != 0:
                df = pd.DataFrame(pd.concat(metrics).round(4))#, columns=[seed for seed in os.listdir(os.path.join(tmp, param))])
                ls.append(df)
            else:
                print(f"Metrics were empty for {mname, param}")
    return pd.concat(ls)


def just_scale_total(collected_df, key_x):
    tmp = collected_df.copy()
    collected_df_scaled = scale_table(tmp.drop(columns=[key_x, 'Model-Architecture']))
    collected_df_scaled[key_x] = collected_df[key_x]
    collected_df_scaled['Model-Architecture'] = collected_df['Model-Architecture']
    return collected_df_scaled

df = get_bigger_table_cell_2(project_root="/local/home/tomap/scAugmentBench/ablation-db-2", dirname="architecture-ablation",
                               dname="MCA")

print(df)

def plot_bigger_table_dim_total(collected_df, key="Batch correction", key_x="hidden_dim", x_label="alpha", palette=None, categories=None):
    tmp = collected_df.copy()
    collected_df_scaled = tmp.drop(columns=[key_x, 'Model-Architecture'])
    collected_df_scaled[key_x] = collected_df[key_x]
    collected_df_scaled['Model-Architecture'] = collected_df['Model-Architecture']

    plt.clf()
    #ax = sns.catplot(x="alpha", y=key, hue="Model-Architecture", kind="point", capsize=.15, data=collected_df, aspect=1.5, errwidth=0.5,).set(title=f'{name} augmentation')
    ax = sns.catplot(x=key_x, y=key, hue="Model-Architecture", kind="point", capsize=.15, data=collected_df_scaled, aspect=1.5, errwidth=0.5,palette=palette, hue_order=categories).set(xlabel=x_label)#,title=f'Hidden-Dim-Choice'
    ax._legend.set_title("")
    ax._legend.remove()
    ax.savefig(f"/local/home/tomap/scAugmentBench/projection-scaler-ablation-{key}.png")
    ax.savefig(f"/local/home/tomap/scAugmentBench/projection-scaler-ablation-{key}.svg")
    return ax

key_x = "Projector Output"

df_scaled = just_scale_total(df, key_x)

#factor_scalers = ["BYOL", "BarlowTwins", "SimSiam", "VICReg"]
#fixed_scaler = ["MoCo", "SimCLR"]

os.makedirs("tables", exist_ok=True)

for mname in os.listdir("./ablation-db-2/architecture-ablation/MCA"):
    df_scaled[df_scaled['Model-Architecture'] == mname].drop(columns=["Model-Architecture"]).groupby("Projector Output").mean().round(4).sort_values(by="Total", ascending=False).to_csv(f"tables/projector_ablation_means_{mname}.csv")
    df_scaled[df_scaled['Model-Architecture'] == mname].drop(columns=["Model-Architecture"]).groupby("Projector Output").std().round(4).sort_values(by="Total", ascending=False).to_csv(f"tables/projector_ablation_stds_{mname}.csv")

ax = plot_bigger_table_dim_total(df_scaled, key="Bio conservation", key_x=key_x, x_label="Scale Factor", palette=palette, categories=categories)
ax = plot_bigger_table_dim_total(df_scaled, key="Batch correction", key_x=key_x, x_label="Scale Factor", palette=palette, categories=categories)
ax = plot_bigger_table_dim_total(df_scaled, key="Total", key_x=key_x, x_label="Scale Factor", palette=palette, categories=categories)

###########################################################################3
###########################################################################3
###########################################################################3
###########################################################################3
###########################################################################3

##
##              VICReg Ablation
##

print("!!! WARNING !!!\nThis code likely uses 64 as the vicreg representation dim.\n\nPossibly change it, if not chosen rep. dim.")

def get_bigger_table(project_root = "/local/home/tomap/scAugmentBench/ablation-db-2", dirname = "architecture-ablation",
                 dname = "MCA"):
    root = os.path.join(project_root, dirname, dname)
    model_names = os.listdir(root)
    if os.path.exists(os.path.join(root, "final_collected.csv")):
            os.remove(os.path.join(root, "final_collected.csv"))
    
    ls = []
    for mname in ["VICReg"]:
        tmp = os.path.join(project_root, dirname, dname, mname)
        
        if os.path.exists(os.path.join(tmp, "mean_result_collected.csv")):
            os.remove(os.path.join(tmp, "mean_result_collected.csv"))
        if os.path.exists(os.path.join(tmp, "std_result_collected.csv")):
            os.remove(os.path.join(tmp, "std_result_collected.csv"))
        
        n_seeds = [len(os.listdir(os.path.join(tmp, param_config))) for param_config in os.listdir(os.path.join(tmp))]
        print(f"Min num seeds: {min(n_seeds)}.\nMax num seeds: {max(n_seeds)}.")

        # get mean and std per parameter-config:
        for lamda in os.listdir(os.path.join(tmp, "64", "2")):
            print(os.listdir(os.path.join(tmp, "64", "2")))
            metrics = []
            metrics.extend(get_results(os.path.join(tmp, "64", "2", lamda)))

            
            if len(metrics) != 0:
                for i in range(len(metrics)):
                    metrics[i]['Model-Architecture'] = mname
                    metrics[i]['Lamda'] = float(lamda)
                
                df = pd.DataFrame(pd.concat(metrics).round(4))#, columns=[seed for seed in os.listdir(os.path.join(tmp, param))])
                ls.append(df)
    return pd.concat(ls)


def just_scale_total(collected_df, key_x):
    tmp = collected_df.copy()
    collected_df_scaled = scale_table(tmp.drop(columns=[key_x, 'Model-Architecture']))
    collected_df_scaled[key_x] = collected_df[key_x]
    collected_df_scaled['Model-Architecture'] = collected_df['Model-Architecture']
    return collected_df_scaled

df = get_bigger_table(project_root="/local/home/tomap/scAugmentBench/ablation-db-2", dirname="architecture-ablation",
                               dname="MCA")

key_x = "Lamda"

df_scaled = just_scale_total(df, key_x)

df_scaled.sort_values(by="Total")

df_scaled.sort_values(by="Total").drop(columns=["Model-Architecture"]).groupby("Lamda").mean().round(4).to_csv(f"tables/vicreg_ablation_means_{mname}.csv")
df_scaled.sort_values(by="Total").drop(columns=["Model-Architecture"]).groupby("Lamda").std().round(4).to_csv(f"tables/vicreg_ablation_stds_{mname}.csv")


#################################################
#################################################
#################################################

#
#   Augmentation Ablation
#

def get_bigger_table_cell_2(project_root = "/local/home/tomap/scAugmentBench/ablation-db-2", dirname = "augmentation-ablation",
                 dname = "MCA"):
    root = os.path.join(project_root, dirname, dname)
    aug_names = os.listdir(root)
    if os.path.exists(os.path.join(root, "final_collected.csv")):
            os.remove(os.path.join(root, "final_collected.csv"))
    
    ls = []
    for aname in aug_names:
        tmp = os.path.join(project_root, dirname, dname, aname)
        
        if os.path.exists(os.path.join(tmp, "mean_result_collected.csv")):
            os.remove(os.path.join(tmp, "mean_result_collected.csv"))
        if os.path.exists(os.path.join(tmp, "std_result_collected.csv")):
            os.remove(os.path.join(tmp, "std_result_collected.csv"))
        
        """n_seeds = [len(os.listdir(os.path.join(tmp, param_config))) for param_config in os.listdir(os.path.join(tmp))]
        print(f"Min num seeds: {min(n_seeds)}.\nMax num seeds: {max(n_seeds)}.")"""

        # get mean and std per parameter-config:
        alphas = os.listdir(tmp)
        for param in alphas:
            print(param)
            if aname == "mnn" or aname == "bbknn" or aname == "gauss":
                knns = []
                for knn in os.listdir(os.path.join(tmp, param)):
                    metrics = []
                    for mname in os.listdir(os.path.join(tmp, param, knn)):
                        metrics.extend(get_results(os.path.join(tmp, param, knn, mname)))
                        knns.extend(knn)
            else:
                for mname in os.listdir(os.path.join(tmp, param)):
                    metrics = get_results(os.path.join(tmp, param, mname))
            
            if len(metrics) != 0:
                """if aname == "mnn" or aname == "bbknn":
                        metrics['knn'] = knns"""
                for i in range(len(metrics)):
                    metrics[i]['Augmentation'] = aname
                    metrics[i]['alpha'] = float(param)
                #if metrics is not None and len(metrics) != 0:
                df = pd.DataFrame(pd.concat(metrics).round(4))#, columns=[seed for seed in os.listdir(os.path.join(tmp, param))])
                ls.append(df)
            else:
                print(f"Metrics were empty for {aname, param}")
    return pd.concat(ls)


def just_scale_total(collected_df, key_x):
    tmp = collected_df.copy()
    print(collected_df)
    print("Above was non-scaled.")
    collected_df_scaled = scale_table(tmp.drop(columns=[key_x, 'Augmentation'])) # 'knn'
    print(collected_df_scaled) 
    collected_df_scaled[key_x] = collected_df[key_x]
    collected_df_scaled['Augmentation'] = collected_df['Augmentation']
    return collected_df_scaled

df = get_bigger_table_cell_2(project_root="/local/home/tomap/scAugmentBench/ablation-db-2", dirname="augmentation-ablation",
                               dname="MCA")

print(df)

def plot_bigger_table_dim_total(collected_df, key="Batch correction", key_x="hidden_dim", x_label="alpha", palette=None, categories=None):
    tmp = collected_df.copy()
    collected_df_scaled = tmp.drop(columns=[key_x, 'Augmentation'])
    collected_df_scaled[key_x] = collected_df[key_x]
    collected_df_scaled['Augmentation'] = collected_df['Augmentation']

    plt.clf()
    #ax = sns.catplot(x="alpha", y=key, hue="Model-Architecture", kind="point", capsize=.15, data=collected_df, aspect=1.5, errwidth=0.5,).set(title=f'{name} augmentation')
    ax = sns.catplot(x=key_x, y=key, hue="Augmentation", kind="point", capsize=.15, data=collected_df_scaled, aspect=1.5, errwidth=0.5,palette=palette, hue_order=categories).set(xlabel=x_label)#,title=f'Hidden-Dim-Choice'
    ax._legend.set_title("")
    ax.savefig(f"/local/home/tomap/scAugmentBench/alpha-ablation-{key}.png")
    ax.savefig(f"/local/home/tomap/scAugmentBench/alpha-ablation-{key}.svg")
    return ax

key_x = "alpha"

df_scaled = just_scale_total(df.loc[~((df['Augmentation'] == "mnn") | (df['Augmentation'] == "bbknn") | (df['Augmentation'] == "gauss"))], key_x)


labels = np.unique(np.array(df["Augmentation"]))
categories=labels
#palette = sns.color_palette("tab10",len(labels))
palette = glasbey.create_palette(palette_size=len(labels), colorblind_safe=True, cvd_severity=50)
print(palette)
label_colors = dict(zip(labels, palette))

categories = [c for c in labels if c in np.unique(np.array(df_scaled["Augmentation"]))]
ax = plot_bigger_table_dim_total(df_scaled, key="Bio conservation", key_x=key_x, x_label=key_x, palette=palette, categories=categories)
ax = plot_bigger_table_dim_total(df_scaled, key="Batch correction", key_x=key_x, x_label=key_x, palette=palette, categories=categories)
ax = plot_bigger_table_dim_total(df_scaled, key="Total", key_x=key_x, x_label=key_x, palette=palette, categories=categories)


#################################################
#################################################
#################################################

#
#   Augmentation Ablation -- Gauss & KNN
#

def get_bigger_table_cell_2(project_root = "/local/home/tomap/scAugmentBench/ablation-db-2", dirname = "augmentation-ablation",
                 dname = "MCA"):
    root = os.path.join(project_root, dirname, dname)
    aug_names = os.listdir(root)
    if os.path.exists(os.path.join(root, "final_collected.csv")):
            os.remove(os.path.join(root, "final_collected.csv"))
    
    ls = []
    for aname in aug_names:
        tmp = os.path.join(project_root, dirname, dname, aname)
        
        if os.path.exists(os.path.join(tmp, "mean_result_collected.csv")):
            os.remove(os.path.join(tmp, "mean_result_collected.csv"))
        if os.path.exists(os.path.join(tmp, "std_result_collected.csv")):
            os.remove(os.path.join(tmp, "std_result_collected.csv"))
        
        """n_seeds = [len(os.listdir(os.path.join(tmp, param_config))) for param_config in os.listdir(os.path.join(tmp))]
        print(f"Min num seeds: {min(n_seeds)}.\nMax num seeds: {max(n_seeds)}.")"""

        # get mean and std per parameter-config:
        alphas = os.listdir(tmp)
        for param in alphas:
            print(param)
            if aname == "mnn" or aname == "bbknn":
                knns = []
                sigmas = []
                for knn in os.listdir(os.path.join(tmp, param)):
                    metrics = []
                    for mname in os.listdir(os.path.join(tmp, param, knn)):
                        metrics.extend(get_results(os.path.join(tmp, param, knn, mname)))
                        
                        if len(metrics) != 0:
                            
                            for i in range(len(metrics)):
                                metrics[i]['Augmentation'] = aname
                                metrics[i]['alpha'] = float(param)
                                metrics[i]['knn'] = int(knn)
                                metrics[i]['sigma'] = None
                            df = pd.DataFrame(pd.concat(metrics).round(4))#, columns=[seed for seed in os.listdir(os.path.join(tmp, param))])
                            ls.append(df)

            elif aname == "gauss":
                for sigma in os.listdir(os.path.join(tmp, param)):
                    metrics = []
                    for mname in os.listdir(os.path.join(tmp, param, sigma)):
                        metrics.extend(get_results(os.path.join(tmp, param, sigma, mname)))
                        
                        if len(metrics) != 0:    
                            for i in range(len(metrics)):
                                metrics[i]['Augmentation'] = aname
                                metrics[i]['alpha'] = float(param)
                                metrics[i]['knn'] = None
                                metrics[i]['sigma'] = float(sigma)
                            #if metrics is not None and len(metrics) != 0:
                            df = pd.DataFrame(pd.concat(metrics).round(4))#, columns=[seed for seed in os.listdir(os.path.join(tmp, param))])
                            ls.append(df)
            else:
                print(f"Metrics were empty for {aname, param}")
    return pd.concat(ls)


def just_scale_total(collected_df, key_x):
    tmp = collected_df.copy()
    print(collected_df)
    print("Above was non-scaled.")
    collected_df_scaled = scale_table(tmp.drop(columns=[key_x, 'knn', 'sigma', 'Augmentation'])) # 'knn'
    print(collected_df_scaled) 
    collected_df_scaled[key_x] = collected_df[key_x]
    collected_df_scaled[[key_x, 'knn', 'sigma', 'Augmentation']] = collected_df[[key_x, 'knn', 'sigma', 'Augmentation']]
    return collected_df_scaled

df = get_bigger_table_cell_2(project_root="/local/home/tomap/scAugmentBench/ablation-db-2", dirname="augmentation-ablation",
                               dname="MCA")

print(df)

def plot_bigger_table_dim_total(collected_df, aparam, key="Batch correction", key_x="hidden_dim", x_label="alpha", palette=None, categories=None):
    tmp = collected_df.copy()
    collected_df_scaled = tmp.drop(columns=[key_x, 'Augmentation'])
    collected_df_scaled[key_x] = collected_df[key_x]
    collected_df_scaled['Augmentation'] = collected_df['Augmentation']

    collected_df_scaled[aparam] = collected_df['Augmentation'].astype(str) + collected_df_scaled[aparam].astype(str)
    plt.clf()
    #ax = sns.catplot(x="alpha", y=key, hue="Model-Architecture", kind="point", capsize=.15, data=collected_df, aspect=1.5, errwidth=0.5,).set(title=f'{name} augmentation')
    ax = sns.catplot(x=key_x, y=key, hue=aparam, kind="point", capsize=.15, data=collected_df_scaled, aspect=1.5, errwidth=0.5,).set(xlabel=x_label)#,title=f'Hidden-Dim-Choice'
    ax._legend.set_title("")
    ax._legend.remove()
    ax.savefig(f"/local/home/tomap/scAugmentBench/{aparam}-ablation-{key}.png")
    ax.savefig(f"/local/home/tomap/scAugmentBench/{aparam}-ablation-{key}.svg")
    return ax


key_x = "alpha"
df_scaled = just_scale_total(df.loc[(df['Augmentation'] == "mnn") | (df['Augmentation'] == "bbknn")], key_x)

print(df_scaled)

ax = plot_bigger_table_dim_total(df_scaled, aparam="knn", key="Bio conservation", key_x=key_x, x_label=key_x, palette=palette, categories=categories)
ax = plot_bigger_table_dim_total(df_scaled, aparam="knn", key="Batch correction", key_x=key_x, x_label=key_x, palette=palette, categories=categories)
ax = plot_bigger_table_dim_total(df_scaled, aparam="knn", key="Total", key_x=key_x, x_label=key_x, palette=palette, categories=categories)

import scanpy as sc
import numpy as np

def plot_umap(adata, embedding, path, mname):
    adata.obsm['Embedding'] = embedding
    sc.pp.neighbors(adata, use_rep="Embedding")
    sc.tl.umap(adata)
    sc.pl.umap(adata, show=False, color=['CellType', 'batchlb'], save=f"_{mname}_emb.png")

"""path = "/local/home/tomap/scAugmentBench/ablation-db-4/augmentation-ablation/MCA/mnn/0.5/3/BarlowTwins/20/embedding.npz"
embedding = np.load(path)['arr_0']
print(embedding.shape)
plot_umap(adata, embedding, "/local/home/tomap/scAugmentBench/", "mnn")

path = "/local/home/tomap/scAugmentBench/ablation-db-4/augmentation-ablation/MCA/bbknn/0.3/3/BarlowTwins/20/embedding.npz"
embedding = np.load(path)['arr_0']
print(embedding.shape)
plot_umap(adata, embedding, "/local/home/tomap/scAugmentBench/", "bbknn")

df_scaled = just_scale_total(df.loc[(df['Augmentation'] == "gauss")], key_x)
print(df_scaled)


ax = plot_bigger_table_dim_total(df_scaled, aparam="sigma", key="Bio conservation", key_x=key_x, x_label=key_x, palette=palette, categories=categories)
ax = plot_bigger_table_dim_total(df_scaled, aparam="sigma", key="Batch correction", key_x=key_x, x_label=key_x, palette=palette, categories=categories)
ax = plot_bigger_table_dim_total(df_scaled, aparam="sigma", key="Total", key_x=key_x, x_label=key_x, palette=palette, categories=categories)"""