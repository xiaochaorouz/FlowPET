"""Dataset registry for the released pediatric PET experiment."""

from typing import Any, Callable, Dict

import torch


class JointAugmentDataset(torch.utils.data.Dataset):
    """Apply the same spatial transform to all loaded image doses."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, index):
        sample = self.dataset[index]
        image_keys = [
            key for key, value in sample.items()
            if key != 'prior' and torch.is_tensor(value) and value.ndim >= 3
        ]
        if not image_keys:
            return sample
        images = torch.stack([sample[key].clone() for key in image_keys])
        images = self.transform(images)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        for offset, key in enumerate(image_keys):
            sample[key] = images[offset]
        return sample


def get_pet_joint_augment():
    """Build the joint augmentation used for training."""
    from torchvision.transforms.v2 import (
        Compose,
        GaussianBlur,
        RandomAffine,
        RandomApply,
        RandomHorizontalFlip,
        RandomRotation,
        ToDtype,
    )

    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomApply([RandomRotation(degrees=10)], p=0.3),
        RandomApply(
            [RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1))],
            p=0.3,
        ),
        RandomApply([GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
        ToDtype(torch.float32, scale=False),
    ])


class DatasetRegistry:
    """Register dataset factories by split."""

    def __init__(self):
        self._train_registry: Dict[str, Callable] = {}
        self._val_registry: Dict[str, Callable] = {}

    def register_train(self, name):
        def decorator(factory):
            self._train_registry[name] = factory
            return factory
        return decorator

    def register_val(self, name):
        def decorator(factory):
            self._val_registry[name] = factory
            return factory
        return decorator

    def create_train_dataset(self, name, config, imaging_system):
        if name not in self._train_registry:
            raise ValueError(f'Unknown train dataset: {name}')
        dataset = self._train_registry[name](config, imaging_system)
        if config.get('do_joint_augment', False):
            dataset = JointAugmentDataset(dataset, get_pet_joint_augment())
        return dataset

    def create_val_dataset(self, name, config, imaging_system):
        if name not in self._val_registry:
            raise ValueError(f'Unknown val dataset: {name}')
        return self._val_registry[name](config, imaging_system)


dataset_registry = DatasetRegistry()


def _create_pediatric(config: Dict[str, Any], imaging_system, split: str):
    """Create one pediatric PET split from user-supplied LMDB paths."""
    from data.pet_data import multidose_SUVlmdb

    load_keys = config.get('load_keys', ['full'])
    paths = config.get('dataset_paths', {}).get(split, {})
    missing = [key for key in load_keys if not paths.get(key)]
    if missing:
        raise ValueError(
            f"Missing dataset_paths.{split} entries for: " + ", ".join(missing)
        )
    return multidose_SUVlmdb(
        prior_path=paths.get('prior'),
        ultra_ultra_low_path=paths.get('ultra_ultra_low'),
        ultra_low_path=paths.get('ultra_low'),
        low_path=paths.get('low'),
        full_path=paths.get('full'),
        imaging_system=imaging_system,
        original_resolution=256,
        image_size=(
            config.get('image_size', 128) if split == 'train'
            else config.get('val_image_size', config.get('image_size', 128))
        ),
        do_augment=False,
        do_normalize=config.get('do_normalize', True),
        Anscobe_normalize=False,
        minmax_normalize=False,
        SUV_window_threshold=config.get('threshold', 1.0),
        lmdb_zfill=6,
        load_keys=load_keys,
    )


@dataset_registry.register_train('Pediatric')
def create_pediatric_train(config, imaging_system):
    """Create the released training split."""
    return _create_pediatric(config, imaging_system, 'train')


@dataset_registry.register_val('Pediatric')
def create_pediatric_val(config, imaging_system):
    """Create the released validation split."""
    return _create_pediatric(config, imaging_system, 'val')
