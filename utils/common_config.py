"""Utilities for common config."""
import torch
import numpy as np

try:
    from .dataset_registry import dataset_registry
    from .model_registry import model_registry
    from .criterion_registry import get_criterion
    from .optimizer_registry import get_optimizer, get_scheduler
except ImportError:
    from utils.dataset_registry import dataset_registry
    from utils.model_registry import model_registry
    from utils.criterion_registry import get_criterion
    from utils.optimizer_registry import get_optimizer, get_scheduler


def get_train_dataset(p, imaging_system):
    """Return train dataset."""
    db_name = p['train_db_name']
    if 'data_loader' in p:
        pass
    return dataset_registry.create_train_dataset(db_name, p, imaging_system)


def get_val_dataset(p, imaging_system):
    """Return val dataset."""
    db_name = p['val_db_name']
    return dataset_registry.create_val_dataset(db_name, p, imaging_system)


def loader_batch_size(p):
    """Interpret batch_size_scope explicitly across distributed ranks."""
    import torch.distributed as dist
    batch = int(p['batch_size'])
    if p.get('batch_size_scope', 'per_process') == 'global':
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if batch < world or batch % world:
            raise ValueError('Global batch_size must be divisible by the number of ranks')
        batch //= world
    return batch


def get_train_dataloader(p, dataset):
    """Return train dataloader."""
    import torch.distributed as dist
    is_unpaired = hasattr(dataset, 'is_paired') and hasattr(dataset, 'get_paired_indices')
    if is_unpaired:
        from .dataset_registry import PairedUnpairedBatchSampler
        num_replicas = None
        rank = None
        if dist.is_available() and dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        batch_sampler = PairedUnpairedBatchSampler(
            dataset,
            batch_size=loader_batch_size(p),
            shuffle=True,
            drop_last=True,
            num_replicas=num_replicas,
            rank=rank
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=p['num_workers'],
            pin_memory=True
        )
    else:
        if dist.is_available() and dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=loader_batch_size(p),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=p['num_workers'],
            pin_memory=True,
            drop_last=True
        )


def get_val_dataloader(p, dataset):
    """Return val dataloader."""
    import torch.distributed as dist
    is_unpaired = hasattr(dataset, 'is_paired') and hasattr(dataset, 'get_paired_indices')
    if is_unpaired:
        from .dataset_registry import PairedUnpairedBatchSampler
        num_replicas = None
        rank = None
        if dist.is_available() and dist.is_initialized():
            num_replicas = dist.get_world_size()
            rank = dist.get_rank()
        batch_sampler = PairedUnpairedBatchSampler(
            dataset,
            batch_size=loader_batch_size(p),
            shuffle=False,
            drop_last=False,
            num_replicas=num_replicas,
            rank=rank
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=p['num_workers'],
            pin_memory=True
        )
    else:
        if dist.is_available() and dist.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False
            )
        else:
            sampler = None
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=loader_batch_size(p),
            shuffle=False,
            sampler=sampler,
            num_workers=p['num_workers'],
            pin_memory=True,
            drop_last=False
        )


def get_train_dataloader_LOOCV(p, dataset, leave_out_case, slices_per_case=81):
    """Return train dataloader LOOCV."""
    import torch.distributed as dist
    total_slices = len(dataset)
    num_cases = total_slices // slices_per_case
    assert 0 <= leave_out_case < num_cases, (
        f"leave_out_case must be in [0, {num_cases - 1}]"
    )
    start_idx = leave_out_case * slices_per_case
    end_idx = start_idx + slices_per_case
    train_indices = list(range(0, start_idx)) + list(range(end_idx, total_slices))
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    if dist.is_available() and dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_subset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    return torch.utils.data.DataLoader(
        train_subset,
        batch_size=loader_batch_size(p),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=p['num_workers'],
        pin_memory=True,
        drop_last=True
    )


def get_val_dataloader_LOOCV(p, dataset, leave_out_case, slices_per_case=81):
    """Return val dataloader LOOCV."""
    import torch.distributed as dist
    total_slices = len(dataset)
    num_cases = total_slices // slices_per_case
    assert 0 <= leave_out_case < num_cases, (
        f"leave_out_case must be in [0, {num_cases - 1}]"
    )
    start_idx = leave_out_case * slices_per_case
    end_idx = start_idx + slices_per_case
    val_indices = list(range(start_idx, end_idx))
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    if dist.is_available() and dist.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            val_subset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=False
        )
    else:
        sampler = None
    return torch.utils.data.DataLoader(
        val_subset,
        batch_size=loader_batch_size(p),
        shuffle=False,
        sampler=sampler,
        num_workers=p['num_workers'],
        pin_memory=True,
        drop_last=False
    )


def get_imaging_system(p):
    if p['imaging_system'] == 'PET':
        from physics.pet import PET
        pet_kwargs = p['imaging_system_kwargs']
        return PET(
            del_count=pet_kwargs['del_count'],
            angles_count=pet_kwargs['angles_count'],
            circle_mask=pet_kwargs.get('circle_mask', True),
            device=pet_kwargs.get('device', 'cuda:0')
        )
    elif p['imaging_system'] == 'CT':
        from physics.ct import CT
        return CT(img_width=p['imaging_system_kwargs']['image_size'], radon_view=p['imaging_system_kwargs']['radon_view'], uniform=p['imaging_system_kwargs'].get('uniform', True), circle=p['imaging_system_kwargs'].get('circle', False))
    elif p['imaging_system'] == 'MRI':
        from physics.mri import SinglecoilMRI_real
        from fastmri.data.subsample import create_mask_for_mask_type
        kw = p['imaging_system_kwargs']
        img_width = kw.get('image_size', kw.get('image_size', 256))
        if img_width is None:
            raise ValueError(
                "MRI imaging_system_kwargs must define an image size"
            )
        mask_type_str = kw.get('mask_type', 'equispaced')  # 'random' / 'equispaced' / ...
        center_fractions = kw.get('center_fraction', 0.08)
        accelerations = kw.get('accelerations', kw.get('acc_factor', 4))
        if not isinstance(center_fractions, (list, tuple)):
            center_fractions = [center_fractions]
        if not isinstance(accelerations, (list, tuple)):
            accelerations = [accelerations]
        mask_func = create_mask_for_mask_type(
            mask_type_str=mask_type_str,
            center_fractions=center_fractions,
            accelerations=accelerations,
        )
        return SinglecoilMRI_real(image_size=img_width, mask=mask_func)
    else:
        raise ValueError(f'Invalid imaging system: {p["imaging_system"]}')


def get_model(p, imaging_system=None, tokenizer=None):
    """Return model."""
    backbone_name = p['backbone']
    return model_registry.create_model(backbone_name, p, imaging_system, tokenizer)

__all__ = [
    'get_train_dataset',
    'get_val_dataset',
    'get_train_dataloader',
    'get_val_dataloader',
    'get_train_dataloader_LOOCV',
    'get_val_dataloader_LOOCV',
    'get_imaging_system',
    'get_model',
    'get_criterion',
    'get_optimizer',
    'get_scheduler',
]
