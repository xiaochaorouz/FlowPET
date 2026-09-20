#!/usr/bin/env python3
"""FlowPET reconstruction and metric helpers retained from the ICML code."""

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
import json

from utils.common_config import (
    get_imaging_system,
    get_val_dataloader_LOOCV,
    get_model,
    get_val_dataset,
    get_val_dataloader
)
from trains.flowpet_trainer import (
    reconstruct_from_sinogram,
)
from torch import nn

VMAX = 100
VMIN = 0
COLORMAP = 'gray'
ERROR_CMAP = 'jet'
ERROR_VMAX = None


def mlem_reconstruction(sino, imaging_system, num_iters=20, eps=1e-3, ratio_max=2.0, alpha=1.0, early_stop_tol=1e-4):
    """Reconstruct an image with maximum-likelihood expectation maximization."""
    device = sino.device
    B, C, N_ang, N_bins = sino.shape
    H, W = imaging_system.volume.height, imaging_system.volume.width
    x = torch.ones((B, 1, H, W), device=device)
    ones_sino = torch.ones_like(sino)
    S = torch.clamp(imaging_system.AT(ones_sino), min=eps)
    prev_x = x.clone()
    for _ in range(num_iters):
        proj_est = torch.clamp(imaging_system.A(x), min=1e-6)
        ratio = torch.clamp(sino / proj_est, min=1e-6, max=ratio_max)
        ratio[ratio != ratio] = 0
        back = imaging_system.AT(ratio)
        x_update = x * back / S
        x = x.pow(1 - alpha) * x_update.pow(alpha) if alpha != 1.0 else x_update
        x = torch.clamp(x, min=0.0)
        diff = torch.norm(x - prev_x) / (torch.norm(prev_x) + 1e-9)
        if diff < early_stop_tol:
            break
        prev_x = x.clone()
    return x


def lowdose_simulate(count=2e5, imaging_system=None, full_dose=None, thresh=None):
    """Simulate a low-count sinogram from a full-dose image."""
    if thresh is not None:
        full_dose = torch.clip(full_dose, min=0, max=thresh) / thresh
    proj = imaging_system.A(full_dose)
    mul_factor = torch.ones_like(proj)
    mul_factor = mul_factor + (torch.rand_like(mul_factor) * 0.2 - 0.1)
    noise = torch.ones_like(proj) * torch.mean(mul_factor * proj, dim=(-1, -2), keepdims=True) * 0.2
    sinogram = mul_factor * proj + noise
    cs = count / (1e-9 + torch.sum(sinogram, dim=(-1, -2), keepdim=True))
    sinogram = sinogram * cs
    mul_factor = mul_factor * cs
    noise = noise * cs
    x = torch.poisson(sinogram)
    sino = nn.ReLU()((x - noise) / mul_factor)
    return sino


def load_model_config(model_dir):
    """Load model config."""
    config_path = None
    for file in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file)
        if file.endswith('.yml') and 'env' not in file:
            config_path = file_path
            break
    if config_path is None:
        raise FileNotFoundError(f"No model configuration found in {model_dir}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model configuration does not exist: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(model_dir, device='cuda'):
    """Load model."""
    config = load_model_config(model_dir)
    model = get_model(config)
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pth.tar')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    return model, config


def load_loocv_model(model_dir, fold_num, device='cuda'):
    """Load a model checkpoint for one LOOCV fold."""
    config = load_model_config(model_dir)
    model = get_model(config)
    checkpoint_path = os.path.join(model_dir, f'LOOCV_{fold_num}', 'checkpoints', f'best_checkpoint_fold{fold_num}.pth')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"LOOCV checkpoint does not exist: {checkpoint_path}")
    print(f"Loading LOOCV checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model, config


def reconstruct_image(config, model, measurement_y, device='cuda', imaging_system=None):
    """Reconstruct an image from a measured sinogram."""
    with torch.no_grad():
        if isinstance(measurement_y, np.ndarray):
            measurement_y = torch.from_numpy(measurement_y).float()
        measurement_y = measurement_y.to(device)
        if config.get('data_range_mode', '[0,1]') != '[0,1]':
            raise ValueError('Released FlowPET expects measurements projected from [0,1] images')
        reconstruction = reconstruct_from_sinogram(
            measurement_y, model, imaging_system, config,
            num_steps=config.get('val_sampling_steps', config.get('sampling_steps', 4)),
        )
        return reconstruction.cpu().numpy()


def calculate_metrics(gt, pred):
    """Compute the released [0, 1] PET metrics."""
    from utils.utils import calculate_pet_metrics
    return calculate_pet_metrics(gt, pred)


def get_sample_data(dataset, sample_idx, config):
    """Return sample data."""
    dataset_size = len(dataset)
    assert 0 <= sample_idx < dataset_size, (
        f"Sample index {sample_idx} is outside [0, {dataset_size - 1}]"
    )
    data = dataset[sample_idx]
    batch_data = {}
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            batch_data[key] = value.unsqueeze(0)
        else:
            batch_data[key] = value
    return batch_data


def reconstruct_single_sample(sample_idx, models, imaging_system, config, device, dataset, data_selection='ultra_ultra_low'):
    """Reconstruct single sample."""
    print(f"Processing sample {sample_idx}")
    data = get_sample_data(dataset, sample_idx, config)
    keys = data.keys()
    print(f"Sample {sample_idx} data keys: {list(keys)}")
    ultra_ultra_low = data['ultra_ultra_low'].to(device).float() if 'ultra_ultra_low' in keys else None
    ultra_low = data['ultra_low'].to(device).float() if 'ultra_low' in keys else None
    low = data['low'].to(device).float() if 'low' in keys else None
    full = data['full'].to(device).float() if 'full' in keys else None
    print(
        f"Sample {sample_idx} availability: "
        f"ultra_ultra_low={ultra_ultra_low is not None}, "
        f"ultra_low={ultra_low is not None}, low={low is not None}, "
        f"full={full is not None}"
    )
    results = {}
    if ultra_ultra_low is not None:
        results['ultra_ultra_low'] = ultra_ultra_low[0, 0].detach().cpu().numpy()
    if full is not None:
        results['full'] = full[0, 0].detach().cpu().numpy()
    for method_name, model_info in models.items():
        try:
            model = model_info['model']
            model_config = model_info['config']
            if config.get('simulate_data', False):
                if full is None:
                    print(f"Cannot reconstruct sample {sample_idx} with {method_name}: "
                          "full-dose data are required for simulation")
                    continue
                sino = lowdose_simulate(
                    count=float(config.get('count', 2e5)),
                    imaging_system=imaging_system,
                    full_dose=full
                )
                measurement_y = sino
            else:
                input_data = None
                if data_selection == 'ultra_low':
                    input_data = ultra_low
                elif data_selection == 'low':
                    input_data = low
                elif data_selection == 'ultra_ultra_low':
                    input_data = ultra_ultra_low
                elif data_selection == 'full':
                    input_data = full
                if input_data is None:
                    print(f"Cannot reconstruct sample {sample_idx} with {method_name}: "
                          f"{data_selection} data are unavailable")
                    continue
                measurement_y = imaging_system.A(input_data)
            reconstruction = reconstruct_image(
                model_config, model, measurement_y, device=device,
                imaging_system=imaging_system,
            )
            reconstruction = np.clip(reconstruction, 0, None)
            results[method_name] = reconstruction[0, 0]
        except Exception as e:
            print(f"Failed to reconstruct sample {sample_idx} with {method_name}: {e}")
            continue
    return results


def reconstruct_3d_volume(start_slice, end_slice, models, imaging_system, config, device, dataset, data_selection='ultra_ultra_low'):
    """Reconstruct a 3D volume over a half-open slice range."""
    print(f"Reconstructing 3D volume over slices [{start_slice}, {end_slice})")
    volume_results = {}
    for method_name in models.keys():
        volume_results[method_name] = []
    volume_results['ultra_ultra_low'] = []
    volume_results['full'] = []
    if config.get('simulate_data', False):
        volume_results['ultra_ultra_low_mlem'] = []
    for slice_idx in range(start_slice, end_slice):
        print(f"Reconstructing slice {slice_idx} ({slice_idx - start_slice + 1}/{end_slice - start_slice})")
        try:
            results = reconstruct_single_sample(slice_idx, models, imaging_system, config, device, dataset, data_selection)
            for method_name, recon in results.items():
                if method_name in ['ultra_ultra_low', 'full']:
                    volume_results[method_name].append(recon)
                elif method_name in volume_results:
                    volume_results[method_name].append(recon)
            if config.get('simulate_data', False) and 'full' in results:
                try:
                    full_tensor = torch.from_numpy(results['full']).unsqueeze(0).unsqueeze(0).to(device)
                    sino = imaging_system.A(full_tensor)
                    count = config.get('count', 2e5)
                    low_dose_sino = lowdose_simulate(count=count, imaging_system=imaging_system, full_dose=full_tensor)
                    mlem_recon = mlem_reconstruction(low_dose_sino, imaging_system)
                    mlem_recon = mlem_recon[0, 0].detach().cpu().numpy()
                    volume_results['ultra_ultra_low_mlem'].append(mlem_recon)
                    print(f"MLEM reconstruction complete; target count: {count}")
                except Exception as e:
                    print(f"MLEM reconstruction failed: {e}")
                    if volume_results['ultra_ultra_low_mlem']:
                        zero_slice = np.zeros_like(volume_results['ultra_ultra_low_mlem'][0])
                    else:
                        zero_slice = np.zeros((128, 128))
                    volume_results['ultra_ultra_low_mlem'].append(zero_slice)
        except Exception as e:
            print(f"Failed to reconstruct slice {slice_idx}: {e}")
            for method_name in volume_results.keys():
                if volume_results[method_name]:
                    zero_slice = np.zeros_like(volume_results[method_name][0])
                else:
                    zero_slice = np.zeros((128, 128))
                volume_results[method_name].append(zero_slice)
    for method_name in volume_results.keys():
        if volume_results[method_name]:
            try:
                volume_results[method_name] = np.stack(volume_results[method_name], axis=2)
                print(f"{method_name}: volume shape {volume_results[method_name].shape}")
            except Exception as e:
                print(f"Failed to stack {method_name} slices into a 3D array: {e}")
                if volume_results[method_name]:
                    first_slice = volume_results[method_name][0]
                    if hasattr(first_slice, 'shape'):
                        shape = first_slice.shape
                    else:
                        shape = (128, 128)
                    volume_results[method_name] = np.zeros((shape[0], shape[1], len(volume_results[method_name])))
                else:
                    volume_results[method_name] = np.zeros((128, 128, end_slice - start_slice))
                print(f"{method_name}: zero-filled volume shape {volume_results[method_name].shape}")
        else:
            print(f"{method_name}: no reconstructions available; using a zero-filled volume")
            volume_results[method_name] = np.zeros((128, 128, end_slice - start_slice))
    return volume_results


def save_3d_volumes_as_nii(volume_results, save_root, start_slice, end_slice, dataset_name=None, simulate_data=False, count=None, solver=None, sampling_steps=None):
    """Save 3D volumes in NIfTI format."""
    import nibabel as nib
    folder_name = "3d_reconstructions"
    if dataset_name:
        folder_name += f"_{dataset_name}"
    if simulate_data and count is not None:
        folder_name += f"_simulate_{count}"
    if solver and sampling_steps is not None:
        folder_name += f"_{solver}_{sampling_steps}"
    nii_dir = os.path.join(save_root, folder_name)
    os.makedirs(nii_dir, exist_ok=True)
    print(f"Saving 3D reconstructions to {nii_dir}")
    for method_name, volume in volume_results.items():
        if not isinstance(volume, np.ndarray):
            print(f"Skipping {method_name}: volume is not a NumPy array")
            continue
        if len(volume.shape) != 3:
            print(f"Skipping {method_name}: expected a 3D volume, got shape {volume.shape}")
            continue
        try:
            nii_img = nib.Nifti1Image(volume, np.eye(4))
            if solver and sampling_steps is not None:
                file_name = f"{method_name}_{solver}_{sampling_steps}.nii.gz"
            else:
                file_name = f"{method_name}.nii.gz"
            nii_path = os.path.join(nii_dir, file_name)
            nib.save(nii_img, nii_path)
            print(f"Saved {method_name}: {nii_path} (shape: {volume.shape})")
        except Exception as e:
            print(f"Failed to save {method_name}: {e}")
            continue
    first_volume_shape = None
    for method_name, volume in volume_results.items():
        if isinstance(volume, np.ndarray) and len(volume.shape) == 3:
            first_volume_shape = list(volume.shape)
            break
    info_data = {
        'reconstruction_info': {
            'start_slice': start_slice,
            'end_slice': end_slice,
            'total_slices': end_slice - start_slice,
            'volume_shape': first_volume_shape,
            'methods': list(volume_results.keys()),
            'dataset_name': dataset_name,
            'simulate_data': simulate_data,
            'count': count,
            'solver': solver,
            'sampling_steps': sampling_steps
        }
    }
    info_path = os.path.join(nii_dir, "reconstruction_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info_data, f, indent=2, ensure_ascii=False)
    print(f"Saved reconstruction metadata: {info_path}")


def save_sample_results(sample_idx, results, save_root, solver=None, sampling_steps=None):
    """Save sample results."""
    sample_dir = os.path.join(save_root, f"sample_{sample_idx:03d}")
    os.makedirs(sample_dir, exist_ok=True)
    global_error_max = None
    if ERROR_VMAX is not None:
        global_error_max = ERROR_VMAX
    else:
        error_maps = []
        for model_name, recon in results.items():
            if model_name not in ['ultra_ultra_low', 'full'] and 'full' in results:
                error_map = np.abs(recon - results['full'])
                error_maps.append(error_map)
        if error_maps:
            global_error_max = np.max([np.max(err) for err in error_maps])
    if solver and sampling_steps is not None:
        suffix = f"_{solver}_{sampling_steps}"
    else:
        suffix = ""
    for model_name, recon in results.items():
        if model_name in ['ultra_ultra_low', 'full']:
            continue
        save_path = os.path.join(sample_dir, f"{model_name}{suffix}_recon.png")
        plt.figure(figsize=(4, 4))
        plt.imshow(recon, cmap=COLORMAP, vmin=VMIN, vmax=VMAX)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        if 'full' in results:
            error_map = np.abs(recon - results['full'])
            save_path_err = os.path.join(sample_dir, f"{model_name}{suffix}_error_map.png")
            plt.figure(figsize=(4, 4))
            error_vmax = global_error_max if global_error_max is not None else np.max(error_map)
            plt.imshow(error_map, cmap=ERROR_CMAP, vmin=0, vmax=error_vmax)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(save_path_err, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()
    if 'full' in results:
        save_path_full = os.path.join(sample_dir, "full.png")
        plt.figure(figsize=(4, 4))
        plt.imshow(results['full'], cmap=COLORMAP, vmin=VMIN, vmax=VMAX)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(save_path_full, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
    if 'ultra_ultra_low' in results:
        save_path_uul = os.path.join(sample_dir, "ultra_ultra_low.png")
        plt.figure(figsize=(4, 4))
        plt.imshow(results['ultra_ultra_low'], cmap=COLORMAP, vmin=VMIN, vmax=VMAX)
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(save_path_uul, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()


def main():
    """The configurable runner replaces this hook at runtime."""
    raise RuntimeError("Use evaluate.py and provide --base_dir.")

if __name__ == "__main__":
    raise SystemExit("Use evaluate.py and provide --base_dir.")
