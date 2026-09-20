from typing import Dict
import torch.optim as optim

OPTIMIZER_MAP = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'adamw': optim.AdamW,
}

SCHEDULER_MAP = {
    'cosine_with_warmup': optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR,
}


def get_optimizer(p: Dict, model, cluster_head_only=False):
    """Return optimizer."""
    optimizer_name = p['optimizer']
    if optimizer_name not in OPTIMIZER_MAP:
        raise ValueError(f'Invalid optimizer: {optimizer_name}')
    optimizer_class = OPTIMIZER_MAP[optimizer_name]
    params = model.parameters()
    return optimizer_class(params, **p['optimizer_kwargs'])


def get_scheduler(p: Dict, optimizer):
    """Return scheduler."""
    scheduler_name = p['scheduler']
    if scheduler_name not in SCHEDULER_MAP:
        raise ValueError(f'Invalid scheduler: {scheduler_name}')
    scheduler_class = SCHEDULER_MAP[scheduler_name]
    return scheduler_class(optimizer, **p['scheduler_kwargs'])
