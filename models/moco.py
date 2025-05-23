# although we have the original CLAIRE-code, we can use the MoCo-code from lightly as a comparison.
import copy

import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.moco_transform import MoCoV2Transform
from lightly.utils.scheduler import cosine_schedule

from utils.train_utils import *
from models.model_utils import get_backbone, get_backbone_deep, clip_loss

import lightning as pl


class MoCo(pl.LightningModule):
    def __init__(self, in_dim, hidden_dim, multimodal, out_dim, memory_bank_size, temperature, max_epochs=200, in_dim2=0, integrate=None, only_rna=False, predict_projection=False, **kwargs):
        super().__init__()

        self.multimodal = multimodal
        self.max_epochs = max_epochs
        self.predict_projection = predict_projection
        
        if self.multimodal:
            self.integrate = integrate
            self.predict_only_rna = only_rna
            self.temperature = 1.0 # nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

            self.backbone = get_backbone_deep(in_dim, hidden_dim, **kwargs)
            self.projection_head = MoCoProjectionHead(hidden_dim, hidden_dim, out_dim)

            self.backbone_momentum = copy.deepcopy(self.backbone)
            self.projection_head_momentum = copy.deepcopy(self.projection_head)

            deactivate_requires_grad(self.backbone_momentum)
            deactivate_requires_grad(self.projection_head_momentum)

            self.backbone2 = get_backbone_deep(in_dim2, hidden_dim, **kwargs)
            self.projection_head2 = MoCoProjectionHead(hidden_dim, hidden_dim, out_dim)

            self.backbone_momentum2 = copy.deepcopy(self.backbone2)
            self.projection_head_momentum2 = copy.deepcopy(self.projection_head2)

            deactivate_requires_grad(self.backbone_momentum2)
            deactivate_requires_grad(self.projection_head_momentum2)

            self.criterion = NTXentLoss(temperature=temperature, memory_bank_size=(memory_bank_size, out_dim))
        
        else:
            self.backbone = get_backbone_deep(in_dim, hidden_dim, **kwargs)
            self.projection_head = MoCoProjectionHead(hidden_dim, hidden_dim, out_dim)

            self.backbone_momentum = copy.deepcopy(self.backbone)
            self.projection_head_momentum = copy.deepcopy(self.projection_head)

            deactivate_requires_grad(self.backbone_momentum)
            deactivate_requires_grad(self.projection_head_momentum)

            self.criterion = NTXentLoss(temperature=temperature, memory_bank_size=(memory_bank_size, out_dim))

    def forward(self, x):
        if self.multimodal:
            query = self.backbone(x[0]).flatten(start_dim=1)
            query = self.projection_head(query)
            
            query2 = self.backbone2(x[1]).flatten(start_dim=1)
            query2 = self.projection_head2(query2)
            return query, query2
        else:
            query = self.backbone(x).flatten(start_dim=1)
            query = self.projection_head(query)
            return query
    
    def predict(self, x):
        with torch.no_grad():
            if self.multimodal:
                z1_0, z1_1 = self(x) if self.predict_projection else self.backbone(x[0]), self.backbone2(x[1]) 

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

    def forward_momentum(self, x):
        if self.multimodal:
            key = self.backbone_momentum(x[0]).flatten(start_dim=1)
            key = self.projection_head_momentum(key).detach()

            key2 = self.backbone_momentum2(x[1]).flatten(start_dim=1)
            key2 = self.projection_head_momentum2(key2).detach()
            return key, key2
        else:
            key = self.backbone_momentum(x).flatten(start_dim=1)
            key = self.projection_head_momentum(key).detach()
            return key

    def training_step(self, batch, batch_idx):
        if self.multimodal:
            momentum = cosine_schedule(self.current_epoch, self.max_epochs, 0.996, 1)
            update_momentum(self.backbone, self.backbone_momentum, m=momentum)
            update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
            update_momentum(self.backbone2, self.backbone_momentum2, m=momentum)
            update_momentum(self.projection_head2, self.projection_head_momentum2, m=momentum)
            
            x_query_1, x_key_1, x_query_2, x_key_2 = batch[0]
            x_query = [x_query_1, x_query_2]
            x_key = [x_key_1, x_key_2]

            query_1, query_2 = self.forward(x_query)
            key_1, key_2 = self.forward_momentum(x_key)

            if self.integrate == "add":
                query = query_1 + query_2
                key = key_1 + key_2
            elif self.integrate == "mean":
                query = (query_1 + query_2) / 2
                key = (key_1 + key_2) / 2
            elif self.integrate == "concat":
                query = torch.cat((query_1, query_2), 1)
                key = torch.cat((key_1, key_2), 1)
            elif self.integrate == "clip":
                loss = 0.5 * (clip_loss(query_1, query_2) + clip_loss(key_1, key_2))
                return loss

            else:
                raise Exception("Invalid integration method.")
            
            loss = self.criterion(query, key)
            return loss
        else:
            momentum = cosine_schedule(self.current_epoch, self.max_epochs, 0.996, 1)
            update_momentum(self.backbone, self.backbone_momentum, m=momentum)
            update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)
            x_query, x_key = batch[0]
            query = self.forward(x_query)
            key = self.forward_momentum(x_key)
            # TODO: symmetrize the loss?
            loss = self.criterion(query, key)
            return loss
    
    def configure_optimizers(self):
        return optimizer_builder(self)

"""    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim"""