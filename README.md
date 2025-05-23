# scSSL-Bench: Benchmarking Self-Supervised Learning for Single-Cell Data

Self-supervised learning (SSL) has emerged as a powerful paradigm for extracting biologically meaningful representations from single-cell data, yet systematic guidelines for single-cell SSL applications remain lacking. Here, we present a comprehensive benchmark, scSSL-Bench, evaluating twelve SSL methods across eight datasets and three critical downstream tasks: batch correction, cell-type annotation, and missing modality prediction. We furthermore systematically assess various data augmentation strategies. Our analysis reveals task-specific trade-offs: the specialized single-cell frameworks, scVI and CLAIRE, excel at uni-modal batch correction, while general SSL methods, VICReg and SimCLR, demonstrate superior performance in cell-typing task and multi-modal data integration. Random masking emerges as the most effective augmentation technique across all tasks. Notably, our results indicate the need for a specialized single-cell multi-modal data integration framework. scSSL-Bench provides a standardized evaluation platform and concrete recommendations for applying SSL to single-cell analysis, advancing the convergence of deep learning and single-cell genomics.


## Scope

We evaluate augmentations across seven generic model architectures. All architectures share the same encoder architecture. They differ in various details, such as:
- the employed loss function
- usage of a memory bank
- implementation of nearest-neighbor-embeddings
- usage of a projector
- usage of a predictor

Additionally, we evaluate specialized self-suupervised methods, CLAIRE, CLEAR, Concerto, and scCLIP, together with generative metods such as scGPT, scBERT, Geneformer, scTEL, scButterfly, and scVI/totalVI.

See the manuscript for details.

## Implementation Details

To install a conda / miniconda / mamba environment for reproducibility, call `conda create --name <env> --file requirements.txt`.

You might need to run `conda install conda-forge::python-annoy`.

We use hydra to schedule experiments (see _conf_ folder) and lightly to define the neural networks (see _model_ folder).
Model training is performed with pytorch-lightning, the ADAM optimizer and a constant learning rate 1e-4.

To schedule experiments from the _conf_ folder, define the data_path in the corresponding file of the conf/data directory. Models and augmentations, as well as the dataset, are defined in the experiment yaml-file. 

To train the model(s), run

`python main.py --multirun +experiment=<experiment_name>`

To schedule [multiple runs with slurm](run.sh), use

`python main.py --multirun +experiment=<experiment_name> +cluster=slurm`

To schedule [multiple runs with slurm and detach the process](run_nohup.sh), use

`nohup python main.py --multirun +experiment=<experiment_name> +cluster=slurm >> <output>.log 2>&1 &`

### Availability of Augmentations
This work evaluates various single-cell augmentations. To use the augmentations in another project:

```python
from main import load_data

train_dataset, val_dataset, adata = load_data(config)
```

where config is required to be a dictionary (e.g. stemming from a .yaml-file) with entries in config["augmentation"].