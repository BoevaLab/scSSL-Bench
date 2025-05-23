# Benchmarking Single-Cell Augmentations for Contrastive SSL

This repository implements a benchmark of augmentations for single-cell RNA-seq data with contrastive learning.

We evaluate augmentations across three common model architectures.

All architectures share the same encoder architecture. They differ in various details, such as:
- the employed loss function
- usage of a memory bank
- implementation of nearest-neighbor-embeddings
- usage of a projector
- usage of a predictor

## Scope

Goal of this project is to advertise further research on contrastive self-supervised learning for cell representation learning.
Our work shows that current methods are able to correct for batch effects, improving performance on downstream tasks.

## Limitations

This work is a first step towards a wider application of CL for cell representation learning. We note that there are parameters and architectural
choices that were not considered during this work due to computational constraints. For the presented models, there are many parameters (and hyperparameters)
that could be improved upon. This is left as future work.

## Implementation Details

To install a conda / miniconda / mamba environment for reproducibility, call `conda create --name <env> --file requirements.txt`.

You might need to install `conda install conda-forge::python-annoy`.

We use hydra to schedule experiments (see _conf_ folder) and lightly to define the neural networks (see _model_ folder).
Model training is performed with pytorch-lightning, the ADAM optimizer and a constant learning rate 1e-4.

To schedule experiments from the _conf_ folder, define the data_path in the corresponding file of the conf/data directory. Models and augmentations, as well as the dataset, are defined in the experiment yaml-file. 

To train the model(s), run

`python main.py --multirun +experiment=<experiment_name>`

To schedule multiple runs with slurm, use

`python main.py --multirun +experiment=<experiment_name> +cluster=slurm`

### Availability of Augmentations
This work evaluates various single-cell augmentations. To use the augmentations in another project:

```python
from main import load_data

train_dataset, val_dataset, adata = load_data(config)
```

where config is required to be a dictionary (e.g. stemming from a .yaml-file) with entries in config["augmentation"].