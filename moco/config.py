import numpy as np
#from moco.trainval import get_args

class Config(object):

    # data directory root
    data_root = '/local/home/tomap/own_model/CLAIRE-data'

    #args = get_args()
    out_root = '/local/home/tomap/own_model/CLAIRE-original-outputs'

    # model configs
    n_hvgs = in_dim = 6000  # n_hvgs

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