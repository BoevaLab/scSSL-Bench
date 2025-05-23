import torch
from torch import nn

from lightly.loss.ntx_ent_loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead

from utils.train_utils import *
from models.model_utils import get_backbone, clip_loss

import lightning as pl
import numpy as np

def student_backbone(in_dim: int, out_dim: int, **kwargs):
    """
    Returns a single-layer encoder to extract features from gene expression profiles.

    In our benchmark study, this architecture is shared between all models.
    """
    modules = [
        nn.BatchNorm1d(in_dim),
        nn.Linear(in_dim, out_dim),
        nn.ReLU(),
        nn.BatchNorm1d(out_dim),
    ]
    return nn.Sequential(*modules)


class AttentionWithContext(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        # the u needs to be initialized. 
        self.u = nn.Parameter(torch.Tensor(in_dim))
        nn.init.xavier_uniform_(self.u.unsqueeze(0))

    def forward(self, x):
        #u_t = tanh(W h_t + b)
        u_t = torch.tanh(self.l1(x))
        a_t = torch.matmul(u_t, self.u)
        a = torch.exp(a_t)
        a = a / (torch.sum(a, dim=1, keepdim=True) + 1e-10)
        a = a.unsqueeze(-1)
        weighted_input = x * a
        return weighted_input, a

    

class Concerto(pl.LightningModule):
    
    def __init__(self, in_dim, hidden_dim, multimodal, out_dim, dropout=0.2, vocabulary_size=4000, in_dim2=0, integrate=None, predict_only_rna=False, predict_projection=False, **kwargs):
        super().__init__()

        self.multimodal = multimodal
        self.predict_projection = predict_projection
        
        if self.multimodal:
            self.integrate = integrate
            self.predict_only_rna = predict_only_rna
            self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

            print(f"Attention with vocabulary-size {vocabulary_size}")
            self.student_encoder = student_backbone(in_dim, hidden_dim, **kwargs)
            self.projection_head_student = SimCLRProjectionHead(hidden_dim, hidden_dim, out_dim, num_layers=1)
            self.embedding_layer = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=out_dim)
            self.attention = AttentionWithContext(out_dim, out_dim)
            self.projection_head_teacher = SimCLRProjectionHead(out_dim, out_dim, out_dim, num_layers=1)
            self.bn1 = nn.BatchNorm1d(in_dim)
            self.bn2 = nn.BatchNorm1d(out_dim)
            self.dropout = nn.Dropout(p=dropout)

            self.student_encoder2 = student_backbone(in_dim2, hidden_dim, **kwargs)
            self.projection_head_student2 = SimCLRProjectionHead(hidden_dim, hidden_dim, out_dim, num_layers=1)
            
            self.embedding_layer2 = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=out_dim)
            self.attention2 = AttentionWithContext(out_dim, out_dim)
            self.projection_head_teacher2 = SimCLRProjectionHead(out_dim, out_dim, out_dim, num_layers=1)
            self.bn1_2 = nn.BatchNorm1d(in_dim2)
            self.bn2_2 = nn.BatchNorm1d(out_dim)
            self.dropout2 = nn.Dropout(p=dropout)

            if self.integrate == 'clip':
                self.loss_img = nn.CrossEntropyLoss()
                self.loss_txt = nn.CrossEntropyLoss()
            else:
                self.criterion = NTXentLoss(temperature=0.2)
        else:
            print(f"Attention with vocabulary-size {vocabulary_size}")
            self.student_encoder = student_backbone(in_dim, hidden_dim, **kwargs)
            self.projection_head_student = SimCLRProjectionHead(hidden_dim, hidden_dim, out_dim, num_layers=1)
            
            self.embedding_layer = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=out_dim)
            self.attention = AttentionWithContext(out_dim, out_dim)
            self.projection_head_teacher = SimCLRProjectionHead(out_dim, out_dim, out_dim, num_layers=1)
            self.bn1 = nn.BatchNorm1d(in_dim)
            self.bn2 = nn.BatchNorm1d(out_dim)
            
            self.dropout = nn.Dropout(p=dropout)
            self.criterion = NTXentLoss(temperature=0.2)

    def forward_teacher(self, x):
        if self.multimodal:
            x0, x1 = x[0], x[1]

            x0 = self.bn1(x0)
            torch.nn.functional.relu_(x0)
            x0 = self.embedding_layer(x0.long())
            x0, a = self.attention(x0)
            x0 = torch.tanh(torch.sum(x0, axis=1))
            x0 = self.bn2(x0)
            x0 = self.dropout(x0)
            z0 = self.projection_head_teacher(x0)

            x1 = self.bn1_2(x1)
            torch.nn.functional.relu_(x1)
            x1 = self.embedding_layer2(x1.long())
            x1, a1 = self.attention2(x1)
            x1 = torch.tanh(torch.sum(x1, axis=1))
            x1 = self.bn2_2(x1)
            x1 = self.dropout2(x1)
            z1 = self.projection_head_teacher2(x1)
            return z0, z1
        else:
            # TODO.
            #sparse_value = torch.unsqueeze(x, 2)
            x = self.bn1(x)
            torch.nn.functional.relu_(x)
            x = self.embedding_layer(x.long())
            x, a = self.attention(x)
            # attention returned nans because u was uninitialized...

            x = torch.tanh(torch.sum(x, axis=1))
            x = self.bn2(x)
            x = self.dropout(x)
            z = self.projection_head_teacher(x)
            return z
    
    def forward(self, x):
        x0, x1 = x[0], x[1]
        if self.multimodal:
            x0 = self.student_encoder(x0).flatten(start_dim=1)
            x0 = self.dropout(x0)
            z0 = self.projection_head_student(x0)

            x1 = self.student_encoder2(x1).flatten(start_dim=1)
            x1 = self.dropout2(x1)
            z1 = self.projection_head_student2(x1)
            return z0, z1
        else:
            x = self.student_encoder(x).flatten(start_dim=1)
            x = self.dropout(x)
            z = self.projection_head_student(x)
            return z
    
    def predict(self, x):
        with torch.no_grad():
            if self.multimodal:
                x0, x1 = x[0], x[1]
                x0 = self.bn1(x0)
                torch.nn.functional.relu_(x0)
                x0 = self.embedding_layer(x0.long())
                x0, a = self.attention(x0)
                x0 = torch.tanh(torch.sum(x0, axis=1))
                z0 = self.bn2(x0)
                if self.predict_projection:
                    z0 = self.projection_head_teacher(z0)

                x1 = self.bn1_2(x1)
                torch.nn.functional.relu_(x1)
                x1 = self.embedding_layer2(x1.long())
                x1, a1 = self.attention2(x1)
                x1 = torch.tanh(torch.sum(x1, axis=1))
                z1 = self.bn2_2(x1)
                if self.predict_projection:
                    z1 = self.projection_head_teacher2(z1)

                # Only RNA embedding
                if self.predict_only_rna:
                    return z0

                if self.integrate == "add":
                    z = z0 + z1
                elif self.integrate == "mean":
                    z = (z0 + z1) / 2
                else:
                    z = torch.cat((z0, z1), 1)
                return z
            else:
                x = self.bn1(x)
                torch.nn.functional.relu_(x)
                x = self.embedding_layer(x.long())
                x, a = self.attention(x)
                x = torch.tanh(torch.sum(x, axis=1))
                z = self.bn2(x)
                if self.predict_projection:
                    z = self.projection_head_teacher(z)
                return z

    def predict_separate(self, x):
        with torch.no_grad():
            if self.multimodal:
                x0, x1 = x[0], x[1]
                x0 = self.bn1(x0)
                torch.nn.functional.relu_(x0)
                x0 = self.embedding_layer(x0.long())
                x0, a = self.attention(x0)
                x0 = torch.tanh(torch.sum(x0, axis=1))
                z0 = self.bn2(x0)
                if self.predict_projection:
                    z0 = self.projection_head_teacher(z0)

                x1 = self.bn1_2(x1)
                torch.nn.functional.relu_(x1)
                x1 = self.embedding_layer2(x1.long())
                x1, a1 = self.attention2(x1)
                x1 = torch.tanh(torch.sum(x1, axis=1))
                z1 = self.bn2_2(x1)
                if self.predict_projection:
                    z1 = self.projection_head_teacher2(z1)

                # Only RNA embedding
                if self.predict_only_rna:
                    raise Exception("Invalid path")

                if self.integrate == "add":
                    z = z0 + z1
                elif self.integrate == "mean":
                    z = (z0 + z1) / 2
                else:
                    z = torch.cat((z0, z1), 1)
                return z, z0, z1
            else:
                raise Exception("Invalid path")
    
    def training_step(self, batch, batch_index):
        if self.multimodal:
            x1_1, x2_1, x1_2, x2_2 = batch[0]
            x0 = [x1_1, x1_2]
            x1 = [x2_1, x2_2]

            z0_0, z1_0 = self.forward(x0)
            z0_1, z1_1 = self.forward_teacher(x1)

            if self.integrate == "add":
                z0 = z0_0 + z1_0
                z1 = z0_1 + z1_1
            elif self.integrate == "mean":
                z0 = (z0_0 + z1_0) / 2
                z1 = (z0_1 + z1_1) / 2
            elif self.integrate == "concat":
                z0 = torch.cat((z0_0, z1_0), 1)
                z1 = torch.cat((z0_1, z1_1), 1)
            elif self.integrate == "clip":
                logit_scale = self.temperature.exp()
                loss = clip_loss(z0_0, z1_0, logit_scale, self.loss_img, self.loss_txt) + clip_loss(z0_1, z1_1, logit_scale, self.loss_img, self.loss_txt)
                return loss

            else:
                raise Exception("Invalid integration method.")

            loss = self.criterion(z0, z1)
        else:
            x0, x1 = batch[0]
            z0 = self.forward(x0)
            z1 = self.forward_teacher(x1)
            # TODO: symmetrize the loss?
            loss = self.criterion(z0, z1)
        return loss
    
    def configure_optimizers(self):
        return optimizer_builder(self)