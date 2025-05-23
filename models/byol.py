import torch
import torch.nn as nn

import lightly
from lightly.loss.sym_neg_cos_sim_loss import SymNegCosineSimilarityLoss
from lightly.models._momentum import _MomentumEncoderMixin

from utils.train_utils import *
from models.model_utils import get_backbone

import lightning as pl


def _get_byol_mlp(num_ftrs: int, hidden_dim: int, out_dim: int):
    """Returns a 2-layer MLP with batch norm on the hidden layer.

    Reference (12.03.2021)
    https://arxiv.org/abs/2006.07733

    """
    modules = [
        nn.Linear(num_ftrs, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim)
    ]
    return nn.Sequential(*modules)


class BYOL(nn.Module, _MomentumEncoderMixin):
    """Implementation of the BYOL architecture.

    Attributes:
        backbone:
            Backbone model to extract features from images.
        num_ftrs:
            Dimension of the embedding (before the projection mlp).
        hidden_dim:
            Dimension of the hidden layer in the projection and prediction mlp.
        out_dim:
            Dimension of the output (after the projection/prediction mlp).
        m:
            Momentum for the momentum update of encoder.

    """

    def __init__(self,
                 # TODO adapt parameters according to paper
                 backbone: nn.Module,
                 num_ftrs: int = 512,
                 hidden_dim: int = 4096,
                 out_dim: int = 256,
                 m: float = 0.999,
                 **kwargs):

        super(BYOL, self).__init__()

        self.backbone = backbone
        self.projection_head = _get_byol_mlp(num_ftrs, hidden_dim, out_dim)
        self.prediction_head = _get_byol_mlp(out_dim, hidden_dim, out_dim)
        self.momentum_backbone = None
        self.momentum_projection_head = None

        self._init_momentum_encoder()
        self.m = m

    def _forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor = None,
                return_features: bool = False):
        
        self._momentum_update(self.m)

        # forward pass of first input x0
        f0 = self.backbone(x0).squeeze()
        z0 = self.projection_head(f0)
        out0 = self.prediction_head(z0)

        # append features if requested
        if return_features:
            out0 = (out0, f0)

        if x1 is None:
            return out0

        # forward pass of second input x1
        with torch.no_grad():

            f1 = self.momentum_backbone(x1).squeeze()
            out1 = self.momentum_projection_head(f1)
        
            if return_features:
                out1 = (out1, f1)
        
        return out0, out1

    def forward(self,
                x0: torch.Tensor,
                x1: torch.Tensor = None,
                return_features: bool = False
                ):
        """Symmetrizes the forward pass (see _forward).

        Performs two forward passes, once where x0 is passed through the encoder
        and x1 through the momentum encoder and once the other way around.

        Args:
            x0:
                Tensor of shape bsz x channels x W x H.
            x1:
                Tensor of shape bsz x channels x W x H.

        Returns: TODO


        """
        p0, z1 = self._forward(x0, x1, return_features=return_features)
        p1, z0 = self._forward(x1, x0, return_features=return_features)

        return (z0, p0), (z1, p1)


class BYOLModule(pl.LightningModule):

    def __init__(self, in_dim, hidden_dim, out_dim, batch_size, **kwargs):
        # create the BYOL-backbone
        super().__init__()
        self.backbone = get_backbone(in_dim, hidden_dim, **kwargs)
        
        # create a simsiam model based on ResNet
        # note that bartontwins has the same architecture
        self.projector_and_predictor = BYOL(
            self.backbone,
            num_ftrs=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            m=0.996
        )
        self.batch_size = batch_size
        self.criterion = SymNegCosineSimilarityLoss()
    
    def training_step(self, batch, batch_idx):
        #(x0, x1), (id0, id1) = batch
        x0, x1 = batch[0]
        y0, y1 = self.projector_and_predictor(x0, x1)
        # TODO: symmetrize the outputs of byol and calculate the loss
        loss = self.criterion(y0, y1)
        self.log('train_loss_ssl', loss)
        return loss
    
    def predict(self, x):
        with torch.no_grad():
            z = self.projector_and_predictor.backbone(x)
            return z
            #return self.projector_and_predictor(x, x)[0][1]
            
    def configure_optimizers(self):
        return optimizer_builder(self)