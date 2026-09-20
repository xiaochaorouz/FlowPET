"""PET image datasets backed by LMDB or NPZ files."""

import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import lmdb
import pickle
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import threading

_lmdb_envs = {}
_lmdb_envs_lock = threading.Lock()


class CompatibleUnpickler(pickle.Unpickler):
    """Load pickles created by different NumPy versions."""
    def find_class(self, module, name):
        if module.startswith('numpy._core'):
            new_module = module.replace('numpy._core', 'numpy.core')
            try:
                return super().find_class(new_module, name)
            except (AttributeError, ModuleNotFoundError):
                try:
                    import importlib
                    mod = importlib.import_module(new_module)
                    return getattr(mod, name)
                except (ImportError, AttributeError):
                    try:
                        return super().find_class(module, name)
                    except (AttributeError, ModuleNotFoundError):
                        import importlib
                        mod = importlib.import_module(module)
                        return getattr(mod, name)
        return super().find_class(module, name)


def safe_pickle_loads(data):
    """Deserialize pickle data across NumPy versions."""
    try:
        return pickle.loads(data)
    except (ModuleNotFoundError, AttributeError) as e:
        if 'numpy._core' in str(e) or 'numpy.core' in str(e):
            import io
            return CompatibleUnpickler(io.BytesIO(data)).load()
        else:
            raise


class BaseLMDB(Dataset):
    """Read images from an LMDB database."""
    def __init__(self, path, original_resolution, zfill: int = 5):
        self.original_resolution = original_resolution
        self.zfill = zfill
        self.lmdb_path = os.path.abspath(path)
        self.length = None

    def _get_env(self):
        """Return the process-local LMDB environment."""
        import os as os_module
        process_id = os_module.getpid()
        env_key = (process_id, self.lmdb_path)
        with _lmdb_envs_lock:
            if env_key not in _lmdb_envs:
                env = lmdb.open(
                    self.lmdb_path,
                    max_readers=32,
                    readonly=True,
                    lock=False,
                    readahead=False,
                    meminit=False,
                )
                if not env:
                    raise IOError('Cannot open lmdb dataset', self.lmdb_path)
                _lmdb_envs[env_key] = env
                if self.length is None:
                    with env.begin(write=False) as txn:
                        length_str = txn.get('length'.encode('utf-8'))
                        if length_str:
                            self.length = int(length_str.decode('utf-8'))
                            print(f"Read length from lmdb: {self.length}")
                        else:
                            self.length = env.stat()['entries']
                            print(f"Read length from lmdb stat: {self.length}")
            return _lmdb_envs[env_key]

    def __getstate__(self):
        """Return serializable dataset state."""
        return {
            'original_resolution': self.original_resolution,
            'zfill': self.zfill,
            'lmdb_path': self.lmdb_path,
            'length': self.length,
        }

    def __setstate__(self, state):
        """Restore serialized dataset state."""
        self.__dict__.update(state)

    def __len__(self):
        """Return the number of samples."""
        if self.length is None:
            try:
                with lmdb.open(self.lmdb_path, readonly=True, lock=False) as env:
                    with env.begin(write=False) as txn:
                        length_str = txn.get('length'.encode('utf-8'))
                        if length_str:
                            self.length = int(length_str.decode('utf-8'))
                        else:
                            self.length = env.stat()['entries']
            except Exception as e:
                print(f"Warning: Failed to read length from lmdb: {e}")
                self.length = 0
        return self.length if self.length is not None else 0

    def __getitem__(self, index):
        """Return one sample."""
        env = self._get_env()
        with env.begin(write=False) as txn:
            key = f'{self.original_resolution}-{str(index).zfill(self.zfill)}'.encode('utf-8')
            serialized_content = txn.get(key)
            if serialized_content is None:
                print(f"Key {key} not found in lmdb.")
                return None
            content = safe_pickle_loads(serialized_content)
            img = Image.fromarray(content)
            return img


class SUVlmdb(Dataset):
    """Load paired low- and full-dose PET images."""
    def __init__(self,
                 low_path=os.path.expanduser('datasets/suv_images.lmdb'),
                 full_path=os.path.expanduser('datasets/suv_images.lmdb'),
                 projection=None,
                 image_size=256,
                 original_resolution=256,
                 do_augment: bool = False,
                 do_normalize: bool = False,
                 Anscobe_normalize: bool = False,
                 minmax_normalize: bool = False,
                 lmdb_zfill: int = 6,
                 SUV_window_threshold: float =  4.0,
                 load_keys: list = None,
                 **kwargs):
        self.original_resolution = original_resolution
        if load_keys is None:
            load_keys = []
            if full_path is not None:
                load_keys.append('full')
            if low_path is not None:
                load_keys.append('low')
        if 'full' not in load_keys:
            if full_path is None:
                raise ValueError("load_keys must include 'full', or full_path must be provided")
            load_keys.insert(0, 'full')
        self.data_L = None
        self.data_F = None
        if 'full' in load_keys:
            if full_path is None:
                raise ValueError("full_path is required when loading 'full' key")
            self.data_F = BaseLMDB(full_path, original_resolution, zfill=lmdb_zfill)
            self.length = len(self.data_F)
        else:
            raise ValueError("'full' key must be loaded (required for reference length)")
        if 'low' in load_keys:
            if low_path is None:
                raise ValueError("low_path is required when loading 'low' key")
            self.data_L = BaseLMDB(low_path, original_resolution, zfill=lmdb_zfill)
            assert len(self.data_L) == len(self.data_F), "low dose and full dose data length not equal"
        self.image_size = image_size
        self.projection = projection
        self.SUV_window_threshold = SUV_window_threshold
        transform = [
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ]
        if do_augment:
            transform.append(transforms.RandomRotation(3))
        if do_normalize:
            threshold_normalize = transforms.Lambda(
                lambda x: torch.clamp(x, max=self.SUV_window_threshold) / self.SUV_window_threshold
            )
            transform.append(threshold_normalize)
        if Anscobe_normalize:
            Anscobe_transform = transforms.Lambda(
                lambda x: 2*torch.sqrt(x+3/8)
            )
            transform.append(Anscobe_transform)
        if minmax_normalize:
            minmax_normalize_transform = transforms.Lambda(
                lambda x: (x-x.min())/(x.max()-x.min())
            )
            transform.append(minmax_normalize_transform)
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        result = {}
        if self.data_F is not None:
            F = self.data_F[index]
            if self.transform is not None:
                F = self.transform(F)
            result['full'] = F
        if self.data_L is not None:
            L = self.data_L[index]
            if self.transform is not None:
                L = self.transform(L)
            result['low'] = L
        return result


class multidose_SUVlmdb(Dataset):
    """Load fixed PET dose levels from separate LMDB databases."""
    def __init__(self,
                 ultra_ultra_low_path=None,
                 ultra_low_path=None,
                 low_path=None,
                 full_path=None,
                 prior_path=None,
                 projection=None,
                 image_size=256,
                 original_resolution=256,
                 do_augment: bool = False,
                 do_normalize: bool = False,
                 Anscobe_normalize: bool = False,
                 minmax_normalize: bool = False,
                 lmdb_zfill: int = 6,
                 SUV_window_threshold: float = 4.0,
                 load_keys: list = None,
                 **kwargs):
        self.original_resolution = original_resolution
        self.image_size = image_size
        self.projection = projection
        if load_keys is None:
            load_keys = []
            if full_path is not None:
                load_keys.append('full')
            if low_path is not None:
                load_keys.append('low')
            if ultra_low_path is not None:
                load_keys.append('ultra_low')
            if ultra_ultra_low_path is not None:
                load_keys.append('ultra_ultra_low')
            if prior_path is not None:
                load_keys.append('prior')
        if 'full' not in load_keys:
            if full_path is None:
                raise ValueError("load_keys must include 'full', or full_path must be provided")
            load_keys.insert(0, 'full')
        self.data = {}
        self.keys = []
        if 'full' in load_keys:
            if full_path is None:
                raise ValueError("full_path is required when loading 'full' key")
            self.data['full'] = BaseLMDB(full_path, original_resolution, zfill=lmdb_zfill)
            self.keys.append('full')
            self.length = len(self.data['full'])
        else:
            raise ValueError("'full' key must be loaded (required for reference length)")
        if 'low' in load_keys:
            if low_path is None:
                raise ValueError("low_path is required when loading 'low' key")
            self.data['low'] = BaseLMDB(low_path, original_resolution, zfill=lmdb_zfill)
            self.keys.append('low')
            assert len(self.data['low']) == self.length, "low dose and full dose data length not equal"
        if 'ultra_low' in load_keys:
            if ultra_low_path is None:
                raise ValueError("ultra_low_path is required when loading 'ultra_low' key")
            self.data['ultra_low'] = BaseLMDB(ultra_low_path, original_resolution, zfill=lmdb_zfill)
            self.keys.append('ultra_low')
            assert len(self.data['ultra_low']) == self.length, "ultra_low dose and full dose data length not equal"
        if 'ultra_ultra_low' in load_keys:
            if ultra_ultra_low_path is None:
                raise ValueError("ultra_ultra_low_path is required when loading 'ultra_ultra_low' key")
            self.data['ultra_ultra_low'] = BaseLMDB(ultra_ultra_low_path, original_resolution, zfill=lmdb_zfill)
            self.keys.append('ultra_ultra_low')
            assert len(self.data['ultra_ultra_low']) == self.length, "ultra_ultra_low dose and full dose data length not equal"
        if 'prior' in load_keys:
            if prior_path is None:
                raise ValueError("prior_path is required when loading 'prior' key")
            self.data['prior'] = BaseLMDB(prior_path, original_resolution, zfill=lmdb_zfill)
            self.keys.append('prior')
            assert len(self.data['prior']) == self.length, "prior dose and full dose data length not equal"
        self.SUV_window_threshold = SUV_window_threshold
        transform = [
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ]
        if do_augment:
            transform.append(transforms.RandomRotation(15))
        if do_normalize:
            threshold_normalize = transforms.Lambda(
                lambda x: torch.clamp(x, max=self.SUV_window_threshold) / self.SUV_window_threshold
            )
            transform.append(threshold_normalize)
        if Anscobe_normalize:
            Anscobe_transform = transforms.Lambda(
                lambda x: 2*torch.sqrt(x+3/8)
            )
            transform.append(Anscobe_transform)
        if minmax_normalize:
            minmax_normalize_transform = transforms.Lambda(
                lambda x: (x-x.min())/(x.max()-x.min())
            )
            transform.append(minmax_normalize_transform)
        prior_transform = [
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ]
        if prior_path is not None and "CT" in prior_path.upper():
            # lung window
            WL, WW = -600, 1500
            L, U = WL - WW/2, WL + WW/2
            prior_transform.extend([
                transforms.Lambda(lambda x: torch.clamp(x, L, U)),
                transforms.Lambda(lambda x: (x - L) / (U - L)),
            ])
        elif prior_path is not None and any(tag in prior_path.upper() for tag in ("MR", "T1", "T2")):
            prior_transform.extend([
                transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
            ])
        self.prior_transform = transforms.Compose(prior_transform)
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        result = {}
        for key in self.keys:
            img = self.data[key][index]
            if key == 'prior' and self.prior_transform is not None:
                img = self.prior_transform(img)
            elif self.transform is not None:
                img = self.transform(img)
            result[key] = img
        return result


class multidose_flexible_SUVlmdb(Dataset):
    """Load named PET dose levels from LMDB databases."""
    def __init__(self,
                 count_paths: dict,
                 projection=None,
                 image_size=256,
                 original_resolution=256,
                 do_augment: bool = False,
                 do_normalize: bool = False,
                 Anscobe_normalize: bool = False,
                 minmax_normalize: bool = False,
                 lmdb_zfill: int = 6,
                 SUV_window_threshold: float = 4.0,
                 load_keys: list = None,
                 **kwargs):
        self.original_resolution = original_resolution
        self.image_size = image_size
        self.projection = projection
        self.data = {}
        self.keys = []
        self.key_mapping = {}
        self.reverse_mapping = {}
        reference_key = None
        reference_path = None
        for key in ['Full', 'full']:
            if key in count_paths:
                reference_key = key
                reference_path = count_paths[key]
                break
        if reference_key is None:
            for key, path in count_paths.items():
                if path is not None:
                    reference_key = key
                    reference_path = path
                    break
        if reference_path is None:
            raise ValueError("count_paths must contain at least one valid path")
        if load_keys is None:
            load_keys = [k for k, v in count_paths.items() if v is not None]
        else:
            for key in load_keys:
                if key not in count_paths:
                    raise ValueError(f"Key '{key}' from load_keys is not present in count_paths")
                if count_paths[key] is None:
                    raise ValueError(f"Path for load_keys entry '{key}' is None")
        if reference_key not in load_keys:
            load_keys.insert(0, reference_key)
        for key in load_keys:
            if key not in count_paths or count_paths[key] is None:
                continue
            path = count_paths[key]
            normalized_key = key.lower()
            self.key_mapping[key] = normalized_key
            self.reverse_mapping[normalized_key] = key
            self.data[normalized_key] = BaseLMDB(path, original_resolution, zfill=lmdb_zfill)
            self.keys.append(normalized_key)
            if normalized_key == reference_key.lower():
                self.length = len(self.data[normalized_key])
            else:
                assert len(self.data[normalized_key]) == self.length, (
                    f"Dataset lengths differ for {key} and {reference_key}: "
                    f"{len(self.data[normalized_key])} vs {self.length}"
                )
        self.SUV_window_threshold = SUV_window_threshold
        transform = [
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: torch.clamp(x, max=self.SUV_window_threshold) / self.SUV_window_threshold
            )
        ]
        if do_augment:
            transform.insert(-1, transforms.RandomRotation(15))
        if Anscobe_normalize:
            Anscobe_transform = transforms.Lambda(
                lambda x: 2*torch.sqrt(x+3/8)
            )
            transform.insert(-1, Anscobe_transform)
        if minmax_normalize:
            minmax_normalize_transform = transforms.Lambda(
                lambda x: (x-x.min())/(x.max()-x.min())
            )
            transform.insert(-1, minmax_normalize_transform)
        self.transform = transforms.Compose(transform)
        ct_transform = [
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ]
        has_ct = any('ct' in k.lower() for k in count_paths.keys())
        if has_ct:
            WL, WW = -600, 1500
            L, U = WL - WW/2, WL + WW/2
            ct_transform.extend([
                transforms.Lambda(lambda x: torch.clamp(x, L, U)),
                transforms.Lambda(lambda x: (x - L) / (U - L)),
            ])
        self.ct_transform = transforms.Compose(ct_transform) if has_ct else None
        self.count_key = None
        for key in ['Full', 'full']:
            normalized_key = key.lower()
            if normalized_key in self.keys:
                self.count_key = normalized_key
                break
        if self.count_key is None:
            for normalized_key in self.keys:
                if normalized_key != 'ct':
                    self.count_key = normalized_key
                    break

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """Return one sample."""
        result = {}
        for normalized_key in self.keys:
            if self.data[normalized_key] is not None:
                img = self.data[normalized_key][index]
                if normalized_key == 'ct' and self.ct_transform is not None:
                    img = self.ct_transform(img)
                elif self.transform is not None:
                    img = self.transform(img)
                result[normalized_key] = img
        if self.count_key is not None and self.count_key in result:
            result['count'] = result[self.count_key]
        return result


class unpaired_multidose_flexible_SUVlmdb(multidose_flexible_SUVlmdb):
    """Mix paired and unpaired PET dose samples."""
    def __init__(self,
                 count_paths: dict,
                 paired_ratio: float = 0.8,
                 projection=None,
                 image_size=256,
                 original_resolution=256,
                 do_augment: bool = False,
                 do_normalize: bool = False,
                 Anscobe_normalize: bool = False,
                 minmax_normalize: bool = False,
                 lmdb_zfill: int = 6,
                 SUV_window_threshold: float = 4.0,
                 load_keys: list = None,
                 **kwargs):
        super().__init__(
            count_paths=count_paths,
            projection=projection,
            image_size=image_size,
            original_resolution=original_resolution,
            do_augment=do_augment,
            do_normalize=do_normalize,
            Anscobe_normalize=Anscobe_normalize,
            minmax_normalize=minmax_normalize,
            lmdb_zfill=lmdb_zfill,
            SUV_window_threshold=SUV_window_threshold,
            load_keys=load_keys,
            **kwargs
        )
        if not 0.0 <= paired_ratio <= 1.0:
            raise ValueError(f"paired_ratio must be in [0, 1], got {paired_ratio}")
        self.paired_ratio = paired_ratio
        self.paired_length = int(self.length * paired_ratio)
        self.unpaired_length = self.length - self.paired_length
        self.full_keys = []
        for key in ['Full', 'full']:
            normalized_key = key.lower()
            if normalized_key in self.keys:
                self.full_keys.append(normalized_key)
        if not self.full_keys and self.count_key is not None:
            self.full_keys.append(self.count_key)

    def is_paired(self, index):
        """Handle paired."""
        return index < self.paired_length

    def get_paired_indices(self):
        """Return paired indices."""
        return list(range(self.paired_length))

    def get_unpaired_indices(self):
        """Return unpaired indices."""
        return list(range(self.paired_length, self.length))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """Return one sample."""
        is_paired = index < self.paired_length
        if is_paired:
            result = super().__getitem__(index)
        else:
            result = {}
            unpaired_index = index
            for normalized_key in self.keys:
                if normalized_key in self.full_keys:
                    continue
                if self.data[normalized_key] is not None:
                    img = self.data[normalized_key][unpaired_index]
                    if normalized_key == 'ct' and self.ct_transform is not None:
                        img = self.ct_transform(img)
                    elif self.transform is not None:
                        img = self.transform(img)
                    result[normalized_key] = img
        return result


class Brainweb_simulated(Dataset):
    """Load simulated BrainWeb PET and anatomical priors."""
    def __init__(self,
                 pet_path=os.path.expanduser('datasets/suv_images.lmdb'),
                 prior_path=os.path.expanduser('datasets/suv_images.lmdb'),
                 projection=None,
                 image_size=256,
                 original_resolution=256,
                 do_augment: bool = False,
                 do_normalize: bool = False,
                 Anscobe_normalize: bool = False,
                 minmax_normalize: bool = False,
                 lmdb_zfill: int = 6,
                 W: float = 4.0,  # SUV window threshold
                 load_keys: list = None,
                 **kwargs):
        self.original_resolution = original_resolution
        if load_keys is None:
            load_keys = []
            if pet_path is not None:
                load_keys.append('full')
            if prior_path is not None:
                load_keys.append('prior')
        if 'full' not in load_keys:
            if pet_path is None:
                raise ValueError("load_keys must include 'full', or pet_path must be provided")
            load_keys.insert(0, 'full')
        self.data_pet = None
        self.data_prior = None
        if 'full' in load_keys:
            if pet_path is None:
                raise ValueError("pet_path is required when loading 'full' key")
            self.data_pet = BaseLMDB(pet_path, original_resolution, zfill=lmdb_zfill)
            self.length = len(self.data_pet)
        else:
            raise ValueError("'full' key must be loaded (required for reference length)")
        if 'prior' in load_keys:
            if prior_path is None:
                raise ValueError("prior_path is required when loading 'prior' key")
            self.data_prior = BaseLMDB(prior_path, original_resolution, zfill=lmdb_zfill)
            assert len(self.data_prior) == self.length, "pet and prior data length not equal"
        self.image_size = image_size
        self.SUV_window_threshold = W
        transform = [
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ]
        if do_augment:
            transform.append(transforms.RandomRotation(15))
        if do_normalize:
            threshold_normalize = transforms.Lambda(
                lambda x: torch.clamp(x, max=self.SUV_window_threshold) / self.SUV_window_threshold
            )
            transform.append(threshold_normalize)
        if Anscobe_normalize:
            Anscobe_transform = transforms.Lambda(
                lambda x: 2*torch.sqrt(x+3/8)
            )
            transform.append(Anscobe_transform)
        if minmax_normalize:
            minmax_normalize_transform = transforms.Lambda(
                lambda x: (x-x.min())/(x.max()-x.min())
            )
            transform.append(minmax_normalize_transform)
        self.transform = transforms.Compose(transform)
        prior_transform = [
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ]
        if prior_path is not None and "CT" in prior_path.upper():
            # lung window
            WL, WW = -600, 1500
            L, U = WL - WW/2, WL + WW/2
            prior_transform.extend([
                transforms.Lambda(lambda x: torch.clamp(x, L, U)),
                transforms.Lambda(lambda x: (x - L) / (U - L)),
            ])
        elif prior_path is not None and any(tag in prior_path.upper() for tag in ("MR", "T1", "T2")):
            prior_transform.extend([
                transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
            ])
        self.prior_transform = transforms.Compose(prior_transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        result = {}
        if self.data_pet is not None:
            pet = self.data_pet[index]
            if self.transform is not None:
                pet = self.transform(pet)
            result['full'] = pet
        if self.data_prior is not None:
            prior = self.data_prior[index]
            if self.prior_transform is not None:
                prior = self.prior_transform(prior)
            result['prior'] = prior
        return result


class Brainweb_simulated_offline_npz(Dataset):
    """Load offline BrainWeb simulations from NPZ files."""
    def __init__(self,
                 npz_dir=None,
                 projection=None,
                 image_size=256,
                 original_resolution=256,
                 do_augment: bool = False,
                 do_normalize: bool = False,
                 Anscobe_normalize: bool = False,
                 minmax_normalize: bool = False,
                 W: float = 4.0,  # SUV window threshold
                 test_index: int = None,
                 patient_idx: int = None,
                 slices_per_patient: int = None,
                 mode: str = 'train',
                 load_keys: list = None,
                 **kwargs):
        import glob
        self.original_resolution = original_resolution
        self.npz_dir = npz_dir
        if npz_dir is None:
            raise ValueError("npz_dir is required")
        if not os.path.exists(npz_dir):
            raise ValueError(f"NPZ directory does not exist: {npz_dir}")
        npz_files = sorted(glob.glob(os.path.join(npz_dir, '*.npz')))
        if len(npz_files) == 0:
            raise ValueError(f"No NPZ files found in {npz_dir}")
        self.npz_files = npz_files
        self.total_length = len(npz_files)
        if load_keys is None:
            load_keys = ['full', 'ultra_ultra_low']
        if 'full' not in load_keys:
            raise ValueError("load_keys must include 'full'")
        self.load_keys = load_keys
        if test_index is not None and patient_idx is not None:
            raise ValueError("test_index and patient_idx are mutually exclusive")
        if patient_idx is not None:
            if slices_per_patient is None:
                raise ValueError("slices_per_patient is required for patient-level LOOCV")
            num_patients = self.total_length // slices_per_patient
            if patient_idx < 0 or patient_idx >= num_patients:
                raise ValueError(f"patient_idx {patient_idx} is outside [0, {num_patients - 1}]")
            test_start_idx = patient_idx * slices_per_patient
            test_end_idx = test_start_idx + slices_per_patient
            train_indices = list(range(0, test_start_idx)) + list(range(test_end_idx, self.total_length))
            test_indices = list(range(test_start_idx, test_end_idx))
            if mode == 'train':
                self.indices = train_indices
            elif mode == 'test':
                self.indices = test_indices
            else:
                raise ValueError(f"mode must be 'train' or 'test', got {mode!r}")
            self.test_index = None
            self.patient_idx = patient_idx
            self.slices_per_patient = slices_per_patient
            self.mode = mode
        elif test_index is not None:
            if test_index < 0 or test_index >= self.total_length:
                raise ValueError(f"test_index {test_index} is outside [0, {self.total_length - 1}]")
            train_indices = [i for i in range(self.total_length) if i != test_index]
            test_indices = [test_index]
            if mode == 'train':
                self.indices = train_indices
            elif mode == 'test':
                self.indices = test_indices
            else:
                raise ValueError(f"mode must be 'train' or 'test', got {mode!r}")
            self.test_index = test_index
            self.patient_idx = None
            self.slices_per_patient = None
            self.mode = mode
        else:
            self.indices = list(range(self.total_length))
            self.test_index = None
            self.patient_idx = None
            self.slices_per_patient = None
            self.mode = None
        self.length = len(self.indices)
        self.image_size = image_size
        self.SUV_window_threshold = W
        transform = [
            transforms.Resize((self.image_size, self.image_size),
                            interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ]
        if do_augment:
            transform.append(transforms.RandomRotation(15))
        if do_normalize:
            threshold_normalize = transforms.Lambda(
                lambda x: torch.clamp(x, max=self.SUV_window_threshold) / self.SUV_window_threshold
            )
            transform.append(threshold_normalize)
        if Anscobe_normalize:
            Anscobe_transform = transforms.Lambda(
                lambda x: 2*torch.sqrt(x+3/8)
            )
            transform.append(Anscobe_transform)
        if minmax_normalize:
            minmax_normalize_transform = transforms.Lambda(
                lambda x: (x-x.min())/(x.max()-x.min())
            )
            transform.append(minmax_normalize_transform)
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """Return one sample."""
        actual_idx = self.indices[idx]
        npz_path = self.npz_files[actual_idx]
        data = np.load(npz_path)
        result = {}
        if 'full' in self.load_keys:
            if 'full' not in data:
                raise KeyError(f"NPZ file {npz_path} does not contain key 'full'")
            full_image = data['full']
            if isinstance(full_image, np.ndarray):
                if len(full_image.shape) == 3:
                    full_image = full_image.squeeze(0)
                elif len(full_image.shape) == 4:
                    full_image = full_image.squeeze(0).squeeze(0)
                full_pil = Image.fromarray(full_image.astype(np.float32), mode='F')
            else:
                full_pil = Image.fromarray(full_image)
            if self.transform is not None:
                full_tensor = self.transform(full_pil)
            else:
                full_tensor = torch.from_numpy(np.array(full_pil)).unsqueeze(0)
            result['full'] = full_tensor
        if 'ultra_ultra_low' in self.load_keys:
            if 'ultra_ultra_low' not in data:
                raise KeyError(f"NPZ file {npz_path} does not contain key 'ultra_ultra_low'")
            ultra_ultra_low_image = data['ultra_ultra_low']
            if isinstance(ultra_ultra_low_image, np.ndarray):
                if len(ultra_ultra_low_image.shape) == 3:
                    ultra_ultra_low_image = ultra_ultra_low_image.squeeze(0)
                elif len(ultra_ultra_low_image.shape) == 4:
                    ultra_ultra_low_image = ultra_ultra_low_image.squeeze(0).squeeze(0)
                ultra_ultra_low_pil = Image.fromarray(ultra_ultra_low_image.astype(np.float32), mode='F')
            else:
                ultra_ultra_low_pil = Image.fromarray(ultra_ultra_low_image)
            if self.transform is not None:
                ultra_ultra_low_tensor = self.transform(ultra_ultra_low_pil)
            else:
                ultra_ultra_low_tensor = torch.from_numpy(np.array(ultra_ultra_low_pil)).unsqueeze(0)
            result['ultra_ultra_low'] = ultra_ultra_low_tensor
        return result

    def get_test_index(self):
        """Return test index."""
        return self.test_index

    def get_patient_idx(self):
        """Return patient idx."""
        return self.patient_idx

    def get_train_size(self):
        """Return train size."""
        if self.patient_idx is not None:
            return self.total_length - self.slices_per_patient
        elif self.test_index is not None:
            return self.total_length - 1
        return self.total_length

    def get_test_size(self):
        """Return test size."""
        if self.patient_idx is not None:
            return self.slices_per_patient
        elif self.test_index is not None:
            return 1
        return 0
