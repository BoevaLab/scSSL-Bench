import torch
from torch import nn

class DomainSpecificBatchNorm(nn.Module):

    def __init__(self, num_features, num_classes):
        super(DomainSpecificBatchNorm, self).__init__()

        dict_batches = {}
        for i in num_classes:
            dict_batches[str(i)] = nn.BatchNorm1d(num_features)
        
        self.bns = nn.ModuleDict(dict_batches)

    def reset_running_stats(self):
        for bn in self.bns.values():
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns.values():
            bn.reset_parameters()

    def forward(self, x, domain_label):
        bn = self.bns[domain_label]
        return bn(x)
    
class DSBNBackbone(nn.Module):
    def __init__(self, in_dim: int, encoder_out_dim: int, dropout:float=0, num_domains=[],):
        super().__init__()
        self.num_domains = num_domains
        self.encoder_out_dim = encoder_out_dim
        self.lin = nn.Linear(in_dim, 128)
        self.bn = DomainSpecificBatchNorm(128, num_domains)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(128, encoder_out_dim)
        self.bn2 = nn.BatchNorm1d(encoder_out_dim)
        self.drop = nn.Dropout(p=dropout)
    
    def reset_running_stats(self):
        self.bn.reset_running_stats()
        self.bn2.reset_running_stats()

    def reset_parameters(self):
        self.bn.reset_parameters()
        self.bn2.reset_parameters()
        
    def forward(self, x, bid):
        results = torch.zeros(x.shape[0], self.encoder_out_dim, device=x.device)
        bid = torch.Tensor(bid).to(x.device)
        for i in self.num_domains:
            mask = bid == i
            new_x = x[mask]

            x1 = self.lin(new_x)
            x2 = self.bn(x1, str(i))
            x3 = self.relu(x2)
            x4 = self.lin2(x3)
            x5 = self.bn2(x4)
            x6 = self.drop(x5)
            # results.index_copy_(0, mask, x6)
            # torch_v = torch.arange(1,n)
            results[mask] = x6
        return results

def get_backbone(in_dim: int, encoder_out_dim: int, dropout:float=0, dsbn=False, num_domains=1, **kwargs):
    pass

def get_backbone_deep(in_dim: int, encoder_out_dim: int, dropout:float=0, dsbn=False, num_domains=1, **kwargs):
    """
    Returns a single-layer encoder to extract features from gene expression profiles.

    In our benchmark study, this architecture is shared between all models.
    """
    print(f"Backbone built with dropout {dropout}")

    if dsbn:
        return DSBNBackbone(in_dim=in_dim, encoder_out_dim=encoder_out_dim, dropout=dropout, num_domains=num_domains)

    else:
        modules = [
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, encoder_out_dim),
            nn.BatchNorm1d(encoder_out_dim),
            nn.Dropout(p=dropout),
        ]
    return nn.Sequential(*modules)

def clip_loss(image_features, text_features, logit_scale, loss_img, loss_txt):
    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ground_truth = torch.arange(len(image_features), dtype=torch.long, device=device)
    total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth))/2

    return total_loss

def cross_entropy(preds, targets):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    return loss
    
