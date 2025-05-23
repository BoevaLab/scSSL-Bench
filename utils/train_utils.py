import torch


def optimizer_builder(model, lr=1e-4, weight_decay=1e-6):
    #optim = torch.optim.SGD(model.parameters(), lr=1e-2)
    optim= torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay,) # momentum=momentum
    scheduler=torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)
    return [optim], [scheduler]

"""
def configure_optimizers(self):
    optimizer = ...
    stepping_batches = self.trainer.estimated_stepping_batches
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=stepping_batches)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
    }

# SEE: https://lightning.ai/docs/pytorch/stable/common/trainer.html#estimated-stepping-batches 
"""