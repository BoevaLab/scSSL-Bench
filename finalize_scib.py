import shutil
from evaluator import recalculate_results
import numpy as np
from main import reset_random_seeds, load_data
import argparse
import os
import yaml


def run_scib(tmp, seed, adata):
    if not os.path.exists(os.path.join(tmp, seed,"evaluation_metrics.csv")):
        print(os.path.join(tmp, seed,))
        reset_random_seeds(int(seed))
        # recalculate scib-metrics.
        if os.path.exists(os.path.join(tmp, seed, "embedding.npz")):
            print(f"Calculate @ {os.path.join(tmp, seed)}")
            emb = np.load(os.path.join(tmp, seed, "embedding.npz"))['arr_0']
            results = recalculate_results(adata, emb, 12)
            results.to_csv(os.path.join(tmp, seed, 'evaluation_metrics.csv'), index=None)


parser = argparse.ArgumentParser(description='Re-Evaluate with SCIB')
# "/local/home/tomap/scAugmentBench/augmentation-ablation-vol9/ImmHuman"
parser.add_argument('--dname_root', default='./', type=str,
                        help='Where to evaluate.')
parser.add_argument('--dataset', default='immune', type=str,
                        help='Where to evaluate.')
parser.add_argument('--project_directory', default='immune', type=str,
                        help='Where to evaluate.')


args = parser.parse_args()
dname_root = args.dname_root


with open(f"{args.project_directory}/conf/data/{args.dataset}.yaml") as stream:
    cfg_data = yaml.safe_load(stream)
with open(f"{args.project_directory}/conf/augmentation/base.yaml") as stream:
    cfg_aug = yaml.safe_load(stream)

cfg = {}
cfg['data'] = cfg_data
cfg['data']['n_hvgs'] = 4000
cfg["data"]["holdout_batch"] = None
cfg['augmentation'] = cfg_aug
train, val, adata = load_data(cfg)

for mname in os.listdir(dname_root):
    for param in os.listdir(os.path.join(dname_root, mname)):
        try:
            for param2 in os.listdir(os.path.join(dname_root, mname, param)):
                for seed in  os.listdir(os.path.join(dname_root, mname, param, param2)):
                    tmp = os.path.join(dname_root, mname, param, param2)
                    run_scib(tmp, seed, adata)
        except:
            try:
                for param2 in os.listdir(os.path.join(dname_root, mname, param)):
                    for param3 in os.listdir(os.path.join(dname_root, mname, param, param2)):
                        for seed in  os.listdir(os.path.join(dname_root, mname, param, param2, param3)):
                            tmp = os.path.join(dname_root, mname, param, param2, param3)
                            run_scib(tmp, seed, adata)
            except:
                for seed in  os.listdir(os.path.join(dname_root, mname, param)):
                    tmp = os.path.join(dname_root, mname, param)
                    run_scib(tmp, seed, adata)