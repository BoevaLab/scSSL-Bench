import numpy as np

class Config(object):
    # data directory root
    data_root = '/home/yxh/gitrepo/Batch-effect-removal-benchmarking-master/Script/sapling/GLOBE/data'
    out_root  = '/home/yxh/gitrepo/clMining/scmoco/CLAIRE-release'

    # model configs
    n_hvgs = in_dim = 2000  # n_hvgs

    min_cells = 0
    scale_factor = 1e4
    n_pcs = 50
    n_neighbors = 15

    verbose = 1

    batch_key = 'batchlb'
    label_key = 'CellType'
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
    


