import copy

import torch
import torchvision
from torch import nn
import lightning as pl

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.loss.sym_neg_cos_sim_loss import SymNegCosineSimilarityLoss

from models.model_utils import *
from utils.train_utils import *
import numpy as np


class BYOL(pl.LightningModule):
    def __init__(self, in_dim, hidden_dim, factor, multimodal, in_dim2=0, integrate=None, only_rna=False, predict_projection=False, **kwargs):
        super().__init__()

        self.multimodal = multimodal
        self.predict_projection = predict_projection

        hidden_dim_2 = factor*hidden_dim
        out_dim = hidden_dim
        
        if self.multimodal:
            self.integrate = integrate
            self.predict_only_rna = only_rna

            self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

            self.backbone = get_backbone_deep(in_dim, hidden_dim, **kwargs)
            self.projection_head = BYOLProjectionHead(hidden_dim, hidden_dim_2, out_dim)
            self.prediction_head = BYOLPredictionHead(out_dim, hidden_dim_2, out_dim)

            self.backbone_momentum = copy.deepcopy(self.backbone)
            self.projection_head_momentum = copy.deepcopy(self.projection_head)

            deactivate_requires_grad(self.backbone_momentum)
            deactivate_requires_grad(self.projection_head_momentum)

            self.backbone2 = get_backbone_deep(in_dim2, hidden_dim, **kwargs)
            self.projection_head2 = BYOLProjectionHead(hidden_dim, hidden_dim_2, out_dim)
            self.prediction_head2 = BYOLPredictionHead(out_dim, hidden_dim_2, out_dim)

            self.backbone_momentum2 = copy.deepcopy(self.backbone2)
            self.projection_head_momentum2 = copy.deepcopy(self.projection_head2)

            deactivate_requires_grad(self.backbone_momentum2)
            deactivate_requires_grad(self.projection_head_momentum2)

            if self.integrate == 'clip':
                self.loss_img = nn.CrossEntropyLoss()
                self.loss_txt = nn.CrossEntropyLoss()
            else:
                self.criterion = SymNegCosineSimilarityLoss()
        else:
            self.backbone = get_backbone_deep(in_dim, hidden_dim, **kwargs)
            self.projection_head = BYOLProjectionHead(hidden_dim, hidden_dim_2, out_dim)
            self.prediction_head = BYOLPredictionHead(out_dim, hidden_dim_2, out_dim)

            self.backbone_momentum = copy.deepcopy(self.backbone)
            self.projection_head_momentum = copy.deepcopy(self.projection_head)

            deactivate_requires_grad(self.backbone_momentum)
            deactivate_requires_grad(self.projection_head_momentum)
            self.criterion = SymNegCosineSimilarityLoss()

    def forward(self, x, bid):
        if self.multimodal:
            y = self.backbone(x[0]).flatten(start_dim=1)
            z = self.projection_head(y)
            p = self.prediction_head(z)

            y2 = self.backbone2(x[1]).flatten(start_dim=1)
            z2 = self.projection_head2(y2)
            p2 = self.prediction_head2(z2)
            return p, p2

        else:
            y = self.backbone(x, bid).flatten(start_dim=1)
            z = self.projection_head(y)
            p = self.prediction_head(z)
            return p

    def forward_momentum(self, x, bid):
        if self.multimodal:
            y = self.backbone_momentum(x[0]).flatten(start_dim=1)
            z = self.projection_head_momentum(y)
            z = z.detach()

            y2 = self.backbone_momentum2(x[1]).flatten(start_dim=1)
            z2 = self.projection_head_momentum2(y2)
            z2 = z2.detach()
            return z, z2
        else:
            y = self.backbone_momentum(x, bid).flatten(start_dim=1)
            z = self.projection_head_momentum(y)
            z = z.detach()
            return z
    
    def training_step(self, batch, batch_idx):
        if self.multimodal:
            x1_0, x1_1, x2_0, x2_1 = batch[0]
            x1 = [x1_0, x2_0]
            x2 = [x1_1, x2_1]

            p1_1, p1_2 = self.forward(x1)
            z1_1, z1_2 = self.forward_momentum(x1)
            p2_1, p2_2 = self.forward(x2)
            z2_1, z2_2 = self.forward_momentum(x2)

            if self.integrate == "add":
                p0 = p1_1 + p1_2
                z0 = z1_1 + z1_2
                p1 = p2_1 + p2_2
                z1 = z2_1 + z2_2
            elif self.integrate == "mean":
                p0 = (p1_1 + p1_2) / 2
                z0 = (z1_1 + z1_2) / 2
                p1 = (p2_1 + p2_2) / 2
                z1 = (z2_1 + z2_2) / 2
            elif self.integrate == "concat":
                p0 = torch.cat((p1_1, p1_2), 1)
                z0 = torch.cat((z1_1, z1_2), 1)
                p1 = torch.cat((p2_1, p2_2), 1)
                z1 = torch.cat((z2_1, z2_2), 1)
            elif self.integrate == "clip":
                # FIXME does it make sense?
                logit_scale = self.temperature.exp()
                loss = clip_loss(p1_1, p1_2, logit_scale, self.loss_img, self.loss_txt) + clip_loss(z1_1, z1_2, logit_scale, self.loss_img, self.loss_txt) + clip_loss(p2_1, p2_2, logit_scale, self.loss_img, self.loss_txt) + clip_loss(z2_1, z2_2, logit_scale, self.loss_img, self.loss_txt)
                return loss
            else:
                raise Exception("Invalid integration method.")

            loss = self.criterion((z0, p0), (z1, p1))
            
        else:
            #(x0, x1), (id0, id1) = batch
            x0, x1 = batch[0]
            bid0, bid1 = batch[2]
            p0 = self.forward(x0, bid0)
            z0 = self.forward_momentum(x0, bid0)
            p1 = self.forward(x1, bid1)
            z1 = self.forward_momentum(x1, bid1)
            # TODO: symmetrize the outputs of byol and calculate the loss
            loss = self.criterion((z0, p0), (z1, p1))
            self.log('train_loss_ssl', loss)
        
        self.log('train_loss', loss)
        return loss
    
    def predict(self, x):
        with torch.no_grad():
            if self.multimodal:
                if self.predict_projection:
                    z1_0, z1_1 = self(x)  
                else:
                    z1_0, z1_1 = self.backbone(x[0]), self.backbone2(x[1])

                if self.predict_only_rna:
                    return z1_0

                if self.integrate == "add":
                    z0 = z1_0 + z1_1
                elif self.integrate == "mean":
                    z0 = (z1_0 + z1_1) / 2
                else:
                    z0 = torch.cat((z1_0, z1_1), 1)
                return z0
            else:
                return self(x) if self.predict_projection else self.backbone(x)
    
    def predict_dsbn(self, x, bid):
        with torch.no_grad():
            return self.backbone(x, bid)

    def predict_separate(self, x):
        with torch.no_grad():
            if self.multimodal:
                if self.predict_projection:
                    z1_0, z1_1 = self(x)  
                else:
                    z1_0, z1_1 = self.backbone(x[0]), self.backbone2(x[1])

                if self.predict_only_rna:
                    raise Exception("Invalid path")

                if self.integrate == "add":
                    z0 = z1_0 + z1_1
                elif self.integrate == "mean":
                    z0 = (z1_0 + z1_1) / 2
                else:
                    z0 = torch.cat((z1_0, z1_1), 1)
                return z0, z1_0, z1_1
            else:
                raise Exception("Invalid path")
    
    def configure_optimizers(self):
        return optimizer_builder(self)