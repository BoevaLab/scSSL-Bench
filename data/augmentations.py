import torch.nn as nn
import torch
import numpy as np
from functools import partial
from torchvision.transforms import Compose
import torch
import torch.nn as nn


#from scipy.stats import bernoulli
from torch import bernoulli, rand, normal # rand is the uniform distribution [0, 1) 


def get_augmentation_list(config, X, nns=None, mnn_dict=None, input_shape=None):
    # TODO: Implement possibility for reordering of augmentations.
    print(input_shape)
    if input_shape is None:
        input_shape = (1, X.shape[1])
    if nns is None:
        return [Mask_Augment(input_shape=input_shape, **config['mask']), # in original CLEAR pipeline this comes first.
                Gauss_Augment(input_shape=input_shape, **config['gauss']),
                InnerSwap_Augment(input_shape=input_shape, **config['innerswap']),
                CrossOver_Augment(X=X, input_shape=input_shape, **config['crossover']),]
    elif mnn_dict is not None:
        return [Mnn_Augment(X=X, mnn_dict=mnn_dict, nns=nns, **config['mnn']),
                Gauss_Augment(input_shape=input_shape, **config['gauss']),
                InnerSwap_Augment(input_shape=input_shape, **config['innerswap']),
                Mask_Augment(input_shape=input_shape, **config['mask']),]
    else:
        return [Bbknn_Augment(X=X, nns=nns, **config['bbknn']),
                Gauss_Augment(input_shape=input_shape, **config['gauss']),
                InnerSwap_Augment(input_shape=input_shape, **config['innerswap']),
                CrossOver_Augment(X=X, input_shape=input_shape, **config['crossover']),
                Mask_Augment(input_shape=input_shape,**config['mask']),]

def get_transforms(transform_list):
    return Compose(transform_list)

# x_bar = x*lambda + x_p*(1-lambda)
def interpolation(x, x_p, alpha):
    #lamda = (alpha - 1.0) * rand(1) + 1
    x = alpha * x + (1 - alpha) * x_p
    return x

# x_bar = x^lambda + x_p^(1-lambda)
def geo_interpolation(x, x_p, alpha):
    lamda = (alpha - 1.0) * rand(1) + 1
    x = (x**lamda) * (x_p**(1-lamda))
    return x

# x_bar = x * ber_vector + x_p * (1-ber_vector)
def binary_switch(x, x_p, alpha):
    bernou_p = bernoulli(alpha*torch.ones(len(x)))
    x = x * bernou_p + x_p * (1-bernou_p)
    return x

class Mnn_Augment(nn.Module):
    
    def __init__(self, X, mnn_dict, nns, alpha, augment_set, apply_prob=0.9, nsize=1, **kwargs):
        super().__init__()
        self.nns = {int(k): nns[k] for k in nns.keys()}
        # TODO: This needs to be done during constr. of mnn_dict.
        self.apply_thresh = apply_prob
        self.mnn_dict = {int(k): mnn_dict[k] for k in mnn_dict.keys()}
        self.alpha = alpha
        print(f"---------- alpha : {alpha} ------------")
        self.nsize = nsize
        self.X = torch.tensor(X.toarray())
        self.anchor_keys = list(self.mnn_dict.keys())
        self.augment_set = []
        for ai in augment_set:
            if ai=='int':
                self.augment_set.append(partial(interpolation, alpha=alpha))
            elif ai=='geo':
                self.augment_set.append(partial(geo_interpolation, alpha=alpha))
            elif ai=='exc':
                #self.augment_set.append(partial(binary_switch_mv, alpha=alpha))
                self.augment_set.append(partial(binary_switch, alpha=alpha))

    def aug_fn(self, x, x_p):
        opi = torch.randint(0, len(self.augment_set), (1,))
        return self.augment_set[opi](x, x_p)
    
    def augment_intra(self, x, cell_id):
        nns = self.nns[cell_id]
        n_intra = [cell_id] if len(nns) == 0 else torch.multinomial(torch.ones(len(nns)), self.nsize, replacement=False)
        x_n = self.X[n_intra]
        return self.aug_fn(x, x_n)

    def augment_inter(self, x, cell_id):
        pos_anchors = self.mnn_dict[cell_id]
        anchor_id = [cell_id] if len(pos_anchors) == 0 else pos_anchors[torch.multinomial(torch.ones(len(pos_anchors)), 1, replacement=False)]

        nns_inter = self.nns[anchor_id]
        n_inter = [anchor_id] if len(nns_inter) == 0 else nns_inter[torch.multinomial(torch.ones(len(nns_inter)), self.nsize, replacement=False)]

        x_p = self.X[anchor_id]
        x_p_n = self.X[n_inter]

        return self.aug_fn(x_p, x_p_n)

    def forward(self, *input):
        view_1, view_2, cell_ids = input[0]['x1'], input[0]['x2'], input[0]['cell_ids']
        
        s = rand(1)
        if s < self.apply_thresh:
            view_1 = self.augment_intra(view_1, int(cell_ids))

            view_2 = self.augment_inter(view_2, int(cell_ids))
        
            """            
            view_1.append(self.aug_fn(self.X[cell_id], x_n))
            view_2.append(self.aug_fn(x_p, x_p_n))"""
        """view_1 = torch.cat(view_1)
        view_2 = torch.cat(view_2)"""

        #return {'x1': view_1.unsqueeze(0), 'x2': view_2.unsqueeze(0), 'cell_ids': cell_ids}
        return {'x1': view_1, 'x2': view_2, 'cell_ids': cell_ids}


class Gauss_Augment(nn.Module):
    
    def __init__(self, noise_percentage: float=0.2, sigma: float=0.5, apply_prob: float=0.3, input_shape=(1, 4000)):
        super().__init__()
        self.apply_thresh = apply_prob
        self.noise_percentage = noise_percentage
        self.sigma = sigma
        self.input_shape=input_shape
    
    def augment(self, x):
        """
        TODO: This is, at the moment, applying the same transform to all cells in the batch. CHANGE!
        """
        #application_tensor = torch.rand(x.shape[1]) <= self.noise_percentage
        num_masked = int(self.noise_percentage*self.input_shape[1])
        mask = torch.cat([torch.ones(num_masked, dtype=torch.bool), 
                      torch.zeros(self.input_shape[1] - num_masked, dtype=torch.bool)])
        mask = mask[torch.randperm(mask.size(0))]

        #application_tensor = torch.rand(x.shape[1], device="cuda") <= self.noise_percentage
        #torch.normal(mean=torch.zeros(x.shape[1]), std=0.5*torch.ones(x.shape[1]))
        #return x + mask * torch.normal(mean=torch.zeros(x.shape[1]), std=0.5*torch.ones(x.shape[1]))
        return x + mask * torch.normal(mean=torch.zeros(self.input_shape[1]), std=self.sigma*torch.ones(self.input_shape[1]))

    def forward(self, *input):
        view_1, view_2, cell_ids = input[0]['x1'], input[0]['x2'], input[0]['cell_ids']
        """view_1 = view_1.cuda()
        view_2 = view_2.cuda()"""

        s = rand(1)
        if s<self.apply_thresh:
            #print("Apply Gauss 1")
            view_1 = self.augment(view_1)
        s = rand(1)
        if s<self.apply_thresh:
            #print("Apply Gauss 2")
            view_2 = self.augment(view_2)
        return {'x1': view_1, 'x2': view_2, 'cell_ids': cell_ids}


"""
Crossing over with any cell, rather than similar ones. This is basically what mnn augment etc. do.
"""
class CrossOver_Augment(nn.Module):
    
    def __init__(self, X, cross_percentage: float=0.25, apply_prob: float=0.4,input_shape=(1, 2000)):
        super().__init__()
        self.apply_thresh = apply_prob
        self.cross_percentage = cross_percentage
        self.X = torch.tensor(X.toarray())
        self.input_shape=input_shape
        
    def augment(self, x):
        cross_idx = torch.randint(0, len(self.X), (1,))
        cross_instance = self.X[cross_idx]
        
        num_masked = int(self.cross_percentage*self.input_shape[1])
        mask = torch.cat([torch.ones(num_masked, dtype=torch.bool), 
                      torch.zeros(self.input_shape[1] - num_masked, dtype=torch.bool)])
        mask = mask[torch.randperm(mask.size(0))]
        antimask = mask == 0
        return x*antimask + cross_instance*mask
        #x[:,mask] = cross_instance[:,mask]
        #return x

    def forward(self, *input):
        #print(input[0])
        view_1, view_2, cell_ids = input[0]['x1'], input[0]['x2'], input[0]['cell_ids']
        s = rand(1)
        if s<self.apply_thresh:
            view_1 = self.augment(view_1)
        s = rand(1)
        if s<self.apply_thresh:
            view_2 = self.augment(view_2)
        return {'x1': view_1, 'x2': view_2, 'cell_ids': cell_ids}


class InnerSwap_Augment(nn.Module):
    
    def __init__(self, swap_percentage: float=0.1, apply_prob: float=0.5, input_shape=(1, 2000)):
        super().__init__()
        self.apply_thresh = apply_prob
        self.swap_percentage = swap_percentage
        self.input_shape = input_shape
    
    def augment(self, x):
        n_swaps = int(self.input_shape[1]*self.swap_percentage//2)
        swap_pair = torch.randint(self.input_shape[1], size=(n_swaps, 2))
        #swap_indices = torch.multinomial(torch.ones(2*x.shape[0], x.shape[1]), n_swaps)
        x[:,swap_pair[:,0]], x[:,swap_pair[:,1]] = x[:,swap_pair[:,1]], x[:,swap_pair[:, 0]]
        return x
    
    def forward(self, *input):
        view_1, view_2, cell_ids = input[0]['x1'], input[0]['x2'], input[0]['cell_ids']
        s = rand(1)
        if s<self.apply_thresh:
            view_1 = self.augment(view_1)
        s = rand(1)
        if s<self.apply_thresh:
            view_2 = self.augment(view_2)
        return {'x1': view_1, 'x2': view_2, 'cell_ids': cell_ids}
    

class Mask_Augment(nn.Module):
    
    def __init__(self, mask_percentage: float = 0.15, apply_prob: float = 0.5, input_shape=(1, 2000)):
        super().__init__()
        self.apply_thresh=apply_prob
        self.mask_percentage=mask_percentage
        self.input_shape=input_shape

    def augment(self, x):
        num_masked = int(self.mask_percentage*self.input_shape[1])
        mask = torch.cat([torch.ones(num_masked, dtype=torch.bool), 
                      torch.zeros(self.input_shape[1] - num_masked, dtype=torch.bool)])
        mask = mask[torch.randperm(mask.size(0))]
        return x*mask

    def forward(self, *input):
        view_1, view_2, cell_ids = input[0]['x1'], input[0]['x2'], input[0]['cell_ids']
        s = rand(1)
        if s<self.apply_thresh:
            view_1 = self.augment(view_1)
        s = rand(1)
        if s<self.apply_thresh:
            view_2 = self.augment(view_2)
        return {'x1': view_1, 'x2': view_2, 'cell_ids': cell_ids}


class Bbknn_Augment(nn.Module):
    
    def __init__(self, X, nns, alpha, augment_set, apply_prob=0.9, nsize=1, **kwargs):
        super().__init__()
        self.apply_thresh = apply_prob
        if nns is not None:
            self.nns = {int(k): nns[k] for k in nns.keys()}
        self.alpha = alpha
        # Number of neighbors to use as basis for mutation.
        self.nsize = nsize
        self.X = torch.tensor(X.toarray())
        if nns is not None:
            self.anchor_keys = list(self.nns.keys())
        self.augment_set = []
        for ai in augment_set:
            if ai=='int':
                self.augment_set.append(partial(interpolation, alpha=alpha))
            elif ai=='geo':
                self.augment_set.append(partial(geo_interpolation, alpha=alpha))
            elif ai=='exc':
                self.augment_set.append(partial(binary_switch, alpha=alpha))

    def aug_fn(self, x, x_p):
        opi = torch.randint(0, len(self.augment_set), (1,))
        return self.augment_set[opi](x, x_p)
    
    def augment(self, x, cell_id):
        nns = self.nns[cell_id]
        neighbor_index = [cell_id] if len(nns) == 0 else torch.multinomial(torch.ones(len(nns)), self.nsize, replacement=False)
        x_n = self.X[neighbor_index]
        return self.aug_fn(x, x_n)
    
    def forward(self, *input):
        view_1, view_2, cell_ids = input[0]['x1'].squeeze(), input[0]['x2'].squeeze(), input[0]['cell_ids']
        
        s = rand(1)
        if s < self.apply_thresh:
            view_1 = self.augment(view_1, int(cell_ids))
        s = rand(1)
        if s < self.apply_thresh:
            view_2 = self.augment(view_2, int(cell_ids))
        
        #return {'x1': view_1.unsqueeze(0), 'x2': view_2.unsqueeze(0), 'cell_ids': cell_ids}
        return {'x1': view_1, 'x2': view_2, 'cell_ids': cell_ids}