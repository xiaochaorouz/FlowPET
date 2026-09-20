import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils.criterion_registry import deep_mse_loss, weighted_mse_loss, wstl_loss, get_criterion
import warnings
import os
from tqdm import tqdm
from pytorch_msssim import SSIM
from torch import nn
import torch.distributed as dist
from utils.utils import Fourier_Utils
import math

V_MAX = 1
V_MIN = 0


class DegradationSchedule:
    """
    Linear Optimal Transport Schedule (Free Particle Model).

    Logic:
    - Position x: Linear interpolation from x0 to x1.
    - Momentum p: Linear interpolation between physical p0 and null-space p1.

    Formulation:
    - alpha_t = 1 - t
    - sigma_t = t
    """
    def __init__(self):
        # No hyperparameters needed for linear schedule
        pass

    def __call__(self, t):
        """Handle call."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        # Use the same linear schedule for position and momentum.
        alpha = 1.0 - t
        sigma = t
        # alpha_x, sigma_x, alpha_p, sigma_p
        return alpha, sigma, alpha, sigma

    def compute_derivatives(self, t):
        """Compute derivatives."""
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        neg_one = -torch.ones_like(t)
        one = torch.ones_like(t)
        # dot_alpha_x, dot_sigma_x, dot_alpha_p, dot_sigma_p
        return neg_one, one, neg_one, one

def unsqueeze_as(t, target):
    """Reshape a batch vector for broadcasting."""
    return t.view(-1, *([1] * (target.ndim - 1)))


def measurement_to_image(measurement_y, imaging_system, p):
    """A-dagger y for the network state; A-transpose y remains a separate condition."""
    if p.get('AT_type', 'FBP') != 'FBP':
        raise ValueError("The released FlowPET configuration uses AT_type=FBP")
    return imaging_system.AT_filtered(measurement_y), imaging_system.AT(measurement_y)


def sample_terminal_momentum(x_1, imaging_system, p):
    """Main-text Eq. (8): gamma (I - A-dagger A) xi, without batch rescaling."""
    momentum_type = p.get('momentum_type', 'range_null')
    if momentum_type != 'range_null':
        raise ValueError('Only the full FlowPET momentum design is included in the release')
    noise = torch.randn_like(x_1)
    projected, _ = measurement_to_image(imaging_system.A(noise), imaging_system, p)
    return float(p.get('gamma', 0.01)) * (noise - projected)


def reconstruct_from_sinogram(measurement_y, velocity_network, imaging_system, p,
                              num_steps=None, return_intermediates=False):
    """Public reconstruction interface. Requires y only; never a full-count reference."""
    x_1, condition = measurement_to_image(measurement_y, imaging_system, p)
    return sample_flowpet(
        x_1, velocity_network, DegradationSchedule(),
        num_steps=num_steps if num_steps is not None else p.get('sampling_steps', 4),
        solver=p.get('solver', 'leapfrog'),
        return_intermediates=return_intermediates, imaging_system=imaging_system,
        p=p, condition=condition,
    )


def compute_flow_matching_loss(
    x_0, x_1, velocity_network, schedule, device,
    criterion=None,
    loss_weight=1.0,
    use_constant_auxiliary=False,
    momentum_alpha=0.0, condition=None, imaging_system=None,
    loss_vx_weight=1.0,
    loss_vp_weight=1.0,
    p=None,
    physical_gradient=None,
    measurement_y=None,
    **kwargs  # Historical keyword compatibility; paper gamma is read from p["gamma"].
):
    """FlowPET main-text Eqs. (7)--(10), retaining the original vector networks.

    x_0 is the full-count target; x_1 is A-dagger measurement_y.
    p['gamma'] scales both boundaries. condition defaults to A-transpose y.
    The classic config uses equally weighted MSE for the two vector components.
    Legacy auxiliary/gradient arguments remain for call compatibility only.
    """
    p = {} if p is None else p
    if use_constant_auxiliary or momentum_alpha or physical_gradient is not None or 'gamma' in kwargs:
        raise ValueError('Legacy momentum overrides are unsupported; use p[gamma] and measurement_y')
    if imaging_system is None:
        raise ValueError("FlowPET requires the PET forward/backprojection operator")
    B, C, H, W = x_0.shape
    t = torch.rand(B, device=device)
    gamma = float(p.get('gamma', 0.01))
    # measurement_y is the projection of the degraded dataset image, NOT A(x_1).
    if measurement_y is None:
        raise ValueError("Pass measurement_y; x_1 must be reconstructed from that same y")
    p_0 = gamma * imaging_system.AT(measurement_y - imaging_system.A(x_0))
    p_1 = sample_terminal_momentum(x_1, imaging_system, p)
    condition = imaging_system.AT(measurement_y) if condition is None else condition
    alpha_x, sigma_x, alpha_p, sigma_p = schedule(t)
    x_t = unsqueeze_as(alpha_x, x_0) * x_0 + unsqueeze_as(sigma_x, x_0) * x_1
    p_t = unsqueeze_as(alpha_p, p_0) * p_0 + unsqueeze_as(sigma_p, p_0) * p_1
    target_v_x = x_1 - x_0
    target_v_p = p_1 - p_0
    target_v = torch.cat([target_v_x, target_v_p], dim=1)
    v_pred = velocity_network(x_t, p_t, t, condition=condition)
    v_pred_x = v_pred[:, :C]
    v_pred_p = v_pred[:, C:]
    target_v_x = target_v[:, :C]
    target_v_p = target_v[:, C:]
    if criterion is not None:
        if isinstance(criterion, torch.nn.Module):
            loss_vx = criterion(v_pred_x, target_v_x)
            loss_vp = criterion(v_pred_p, target_v_p)
        else:
            try:
                loss_vx = criterion(v_pred_x, target_v_x)
                loss_vp = criterion(v_pred_p, target_v_p)
            except (TypeError, ValueError):
                loss_vx = criterion(target=target_v_x, input=v_pred_x)
                loss_vp = criterion(target=target_v_p, input=v_pred_p)
    else:
        loss_vx = F.mse_loss(v_pred_x, target_v_x)
        loss_vp = F.mse_loss(v_pred_p, target_v_p)
    weighted_loss_vx = loss_vx_weight * loss_vx
    weighted_loss_vp = loss_vp_weight * loss_vp
    total_loss = weighted_loss_vx + weighted_loss_vp
    weighted_loss = loss_weight * total_loss
    return weighted_loss, {
        'flow_matching_loss': weighted_loss,
        'loss_vx': weighted_loss_vx.detach(),
        'loss_vp': weighted_loss_vp.detach(),
    }


def sample_flowpet(
    x_1,
    velocity_network,
    schedule,
    num_steps=4,
    solver='leapfrog',
    return_intermediates=False,
    device=None,
    imaging_system=None,
    p=None,
    condition=None
):
    """Original image-domain integrator, from (x_1,p_1) at t=1 to t=0.

    Public callers should use reconstruct_from_sinogram to obtain x_1=A-dagger y
    and the distinct static condition=A-transpose y. Both states are integrated
    for Euler, RK4 and Leapfrog. No reference-dependent momentum is accepted.
    Returns x_0, optionally together with intermediate images.
    """
    if device is None:
        device = x_1.device
    B, C, H, W = x_1.shape
    p = {} if p is None else p
    if imaging_system is None:
        raise ValueError("FlowPET sampling requires imaging_system")
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if condition is None:
        raise ValueError("FlowPET sampling requires the static condition A-transpose y")
    p_1 = sample_terminal_momentum(x_1, imaging_system, p)
    t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
    x_curr = x_1.clone()
    p_curr = p_1.clone()
    intermediates = [] if return_intermediates else None
    if return_intermediates:
        intermediates.append(x_curr.clone())
    velocity_network.eval()
    with torch.no_grad():
        for i in range(num_steps):
            t = t_steps[i]
            t_next = t_steps[i + 1]
            dt = t_next - t
            t_batch = t.expand(B) if t.dim() == 0 else t
            t_next_batch = t_next.expand(B) if t_next.dim() == 0 else t_next
            if solver == 'euler':
                v = velocity_network(x_curr, p_curr, t_batch, condition=condition)  # [B, 2*C, H, W]
                v_x = v[:, :C, :, :]  # [B, C, H, W]
                v_p = v[:, C:, :, :]  # [B, C, H, W]
                x_next = x_curr + v_x * dt
                p_next = p_curr + v_p * dt
            elif solver == 'rk4':
                t_mid = (t + dt / 2).expand(B) if (t + dt / 2).dim() == 0 else (t + dt / 2)
                # k1
                v1 = velocity_network(x_curr, p_curr, t_batch, condition=condition)
                v1_x = v1[:, :C, :, :]
                v1_p = v1[:, C:, :, :]
                k1_x = v1_x * dt
                k1_p = v1_p * dt
                # k2
                x_mid1 = x_curr + k1_x / 2
                p_mid1 = p_curr + k1_p / 2
                v2 = velocity_network(x_mid1, p_mid1, t_mid, condition=condition)
                v2_x = v2[:, :C, :, :]
                v2_p = v2[:, C:, :, :]
                k2_x = v2_x * dt
                k2_p = v2_p * dt
                # k3
                x_mid2 = x_curr + k2_x / 2
                p_mid2 = p_curr + k2_p / 2
                v3 = velocity_network(x_mid2, p_mid2, t_mid, condition=condition)
                v3_x = v3[:, :C, :, :]
                v3_p = v3[:, C:, :, :]
                k3_x = v3_x * dt
                k3_p = v3_p * dt
                # k4
                x_mid3 = x_curr + k3_x
                p_mid3 = p_curr + k3_p
                v4 = velocity_network(x_mid3, p_mid3, t_next_batch, condition=condition)
                v4_x = v4[:, :C, :, :]
                v4_p = v4[:, C:, :, :]
                k4_x = v4_x * dt
                k4_p = v4_p * dt
                x_next = x_curr + (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
                p_next = p_curr + (k1_p + 2*k2_p + 2*k3_p + k4_p) / 6
            elif solver == 'leapfrog':
                t_mid = (t + dt / 2).expand(B) if (t + dt / 2).dim() == 0 else (t + dt / 2)
                v_all_curr = velocity_network(x_curr, p_curr, t_batch, condition=condition)
                force_curr = v_all_curr[:, C:, :, :]  # v_p
                p_half = p_curr + force_curr * (dt / 2)
                v_all_half = velocity_network(x_curr, p_half, t_mid, condition=condition)
                velocity_half = v_all_half[:, :C, :, :]  # v_x
                x_next = x_curr + velocity_half * dt
                v_all_next = velocity_network(x_next, p_half, t_next_batch, condition=condition)
                force_next = v_all_next[:, C:, :, :]  # v_p
                p_next = p_half + force_next * (dt / 2)
            else:
                raise ValueError(f"Unknown solver: {solver}. Choose 'euler', 'rk4', or 'leapfrog'")
            x_curr = x_next
            p_curr = p_next
            if return_intermediates:
                intermediates.append(x_curr.clone())
    x_0 = x_curr
    if return_intermediates:
        return x_0, intermediates
    else:
        return x_0


def restore_image(
    x_1, velocity_network, schedule, device, num_steps=4,
    imaging_system=None, p=None, condition=None
):
    """Compatibility wrapper for the original image-domain sampler.

    Pass explicit condition=A-transpose y for paper-aligned inference;
    reconstruct_from_sinogram is the public measurement-domain entry point.
    """
    solver = p.get('solver', 'leapfrog') if p is not None else 'leapfrog'
    return sample_flowpet(
        x_1=x_1,
        velocity_network=velocity_network,
        schedule=schedule,
        num_steps=num_steps,
        solver=solver,
        return_intermediates=False,
        device=device,
        imaging_system=imaging_system,
        p=p,
        condition=condition
    )


def is_distributed():
    """Return whether distributed training is active."""
    return dist.is_available() and dist.is_initialized()


def get_rank():
    """Return rank."""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_world_size():
    """Return world size."""
    if is_distributed():
        return dist.get_world_size()
    return 1


def is_main_process():
    """Return whether this is the main process."""
    return get_rank() == 0


def reduce_tensor(tensor):
    """Reduce tensor."""
    if not is_distributed():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def select_input_key(input_keys, data, random_select=True):
    """Select input key."""
    candidate_keys = []
    available_keys = list(data.keys())
    excluded_keys = ['count', 'prior', 'ct', 'full']
    for key in input_keys:
        if key in data and key not in excluded_keys:
            candidate_keys.append(key)
    if len(candidate_keys) == 0:
        for key in available_keys:
            if key not in excluded_keys:
                candidate_keys.append(key)
    if len(candidate_keys) == 0:
        raise ValueError(
            f"Cannot determine input data from input_keys: {input_keys}. "
            f"Available keys: {available_keys}. "
            f"Please ensure input_keys contains at least one valid input key (excluding 'count', 'prior', 'ct')."
        )
    if len(candidate_keys) > 1 and random_select:
        input_key = np.random.choice(candidate_keys)
    else:
        input_key = candidate_keys[0]
    return input_key


def prepare_input_data(data, input_keys, p, imaging_system, device, is_training=True, tokenizer=None):
    input_mode = p.get('input_mode', 'sinogram')
    prior = None
    if p.get('use_prior', False) and 'prior' in data:
        prior = data['prior'].to(device).float()
    data_range_mode = p.get('data_range_mode', '[-1,1]')
    ref = data['full'].to(device).float()
    if data_range_mode == '[-1,1]':
        ref = ref * 2.0 - 1.0
    random_select = p.get('random_input_key', True)
    input_key = select_input_key(input_keys, data, random_select=random_select)
    input_tensor = data[input_key].to(device).float()
    if data_range_mode == '[-1,1]':
        input_tensor = input_tensor * 2.0 - 1.0
    if input_mode == 'sinogram':
        if p.get('simulate_data', False):
            sino = imaging_system.A_degrade(ref, count=float(p['count']), thresh=None)
            input = sino
        else:
            sino = imaging_system.A(input_tensor)
            input = sino
    elif input_mode == 'image':
        if tokenizer is not None:
            if len(input_tensor.shape) == 4 and input_tensor.shape[1] == 1:
                assert input_tensor.min() < 0, "input should be in [-1, 1] range"
                input = tokenizer.encode(input_tensor, deterministic=True)  # [B, 4, H/8, W/8]
            if ref is not None:
                assert ref.min() < 0, "full should be in [-1, 1] range"
                ref = tokenizer.encode(ref, deterministic=True)  # [B, 4, H/8, W/8]
        else:
            input = input_tensor
    elif input_mode == 'self_supervised':
        prob_min = p.get('prob_min', 0.1)
        prob_max = p.get('prob_max', 0.9)
        prob = torch.rand(1).item() * (prob_max - prob_min) + prob_min
        if data_range_mode == '[-1,1]':
            ref = (ref + 1.0) / 2.0
        sino1, sino2 = imaging_system.A_split(ref, prob)
        input = sino1
        ref = sino2
    else:
        raise ValueError(f"Invalid input mode: {input_mode}")
    return input, ref, prior


def forward_pass(model, input, prior, use_prior):
    """Handle pass."""
    if use_prior:
        if prior is None:
            prior = input
        outputs, diff = model(input, prior)
    else:
        outputs, diff = model(input)
    if isinstance(outputs, list):
        predx = outputs[-1]
    else:
        predx = outputs
    return outputs, diff, predx


def parse_loss_config(loss_config):
    """Handle loss config."""
    if isinstance(loss_config, dict):
        weight = float(loss_config.get('weight', loss_config.get('w', 0.0)))
        domain = loss_config.get('domain', 'image').lower()
        if domain not in ['latent', 'image']:
            raise ValueError(f"Invalid domain '{domain}'. Must be 'latent' or 'image'")
        return weight, domain
    else:
        weight = float(loss_config)
        return weight, 'image'


def compute_loss(predx, outputs, diff, full, criterion, p,
                 predx_latent=None, outputs_latent=None, full_latent=None, tokenizer=None):
    full_f32 = full.float()
    predx_f32 = predx.float()
    if torch.isnan(predx_f32).any() or torch.isinf(predx_f32).any():
        predx_f32 = torch.nan_to_num(predx_f32, nan=0.0, posinf=1.0, neginf=-1.0)
    if torch.isnan(full_f32).any() or torch.isinf(full_f32).any():
        full_f32 = torch.nan_to_num(full_f32, nan=0.0, posinf=1.0, neginf=-1.0)
    data_range_mode = p.get('data_range_mode', '[-1,1]')
    if data_range_mode == '[-1,1]':
        data_range = full_f32.max() - full_f32.min()
        if data_range <= 0:
            data_range = 2.0
    else:
        data_range = full_f32.max() - full_f32.min()
        if data_range <= 0:
            data_range = 1.0
    ssim_module = SSIM(data_range=data_range, size_average=True, channel=1)
    device = predx.device
    loss_dict = {
        'loss_pred': torch.tensor(0.0, device=device),
        'amp_loss': torch.tensor(0.0, device=device),
        'phase_loss': torch.tensor(0.0, device=device),
        'f_loss': torch.tensor(0.0, device=device),
        'loss_sim': torch.tensor(0.0, device=device),
        'loss_sinogram': torch.tensor(0.0, device=device),
        'inner_loss_pred': torch.tensor(0.0, device=device),
        'mse_loss': torch.tensor(0.0, device=device),
        'wstl_loss': torch.tensor(0.0, device=device),
        'weighted_deep_mse_loss': torch.tensor(0.0, device=device),
    }
    for key, value in p['loss_type'].items():
        weight, domain = parse_loss_config(value)
        if weight == 0.0:
            continue
        if domain == 'latent':
            if tokenizer is None:
                import warnings
                warnings.warn(f"Loss '{key}' is configured to compute in latent domain, but tokenizer is not available. "
                            f"Falling back to image domain computation.")
                domain = 'image'
            if domain == 'latent':
                if predx_latent is None or full_latent is None:
                    raise ValueError(f"Loss '{key}' requires latent domain but latent data is not provided. "
                                   f"Make sure tokenizer is properly initialized and data is encoded to latent space.")
                predx_data = predx_latent.float()
                full_data = full_latent.float()
                outputs_data = outputs_latent if outputs_latent is not None else outputs
            else:
                predx_data = predx_f32
                full_data = full_f32
                outputs_data = outputs
        else:
            predx_data = predx_f32
            full_data = full_f32
            outputs_data = outputs
        if key == 'loss_pred':
            loss_dict['loss_pred'] = weight * criterion(target=full_data, input=predx_data)
        elif key == 'loss_phase_amp':
            if domain == 'latent':
                import warnings
                warnings.warn(f"Loss '{key}' is typically computed in image domain, but 'latent' is specified. "
                            f"Proceeding with latent domain computation.")
            predx_phase = predx_data
            full_phase = full_data
            p_loss, a_loss = Fourier_Utils.get_phase_amp_loss(predx_phase, full_phase)
            loss_dict['phase_loss'] += weight * p_loss
            loss_dict['amp_loss'] += weight * a_loss
        elif key == 'loss_fourier':
            if domain == 'latent':
                import warnings
                warnings.warn(f"Loss '{key}' is typically computed in image domain, but 'latent' is specified. "
                            f"Proceeding with latent domain computation.")
            predx_fourier = predx_data
            full_fourier = full_data
            f_loss_ = Fourier_Utils.get_fourier_loss(predx_fourier, full_fourier)
            loss_dict['f_loss'] += weight * f_loss_
        elif key == 'loss_ssim':
            if domain == 'latent':
                data_range_latent = full_data.max() - full_data.min()
                if data_range_latent <= 0:
                    data_range_latent = full_data.std() * 6.0
                ssim_module_latent = SSIM(data_range=data_range_latent, size_average=True, channel=full_data.shape[1])
                loss_sim_ = 1 - ssim_module_latent(full_data, predx_data)
            else:
                loss_sim_ = 1 - ssim_module(full_data, predx_data)
            if torch.isnan(loss_sim_) or torch.isinf(loss_sim_):
                loss_sim_ = torch.tensor(0.0, device=device)
            loss_dict['loss_sim'] += weight * loss_sim_
        elif key == 'loss_sino':
            loss_dict['loss_sinogram'] += weight * criterion(target=diff, input=torch.zeros_like(diff))
        elif key == 'inner_loss_pred':
            losses = []
            for output in outputs_data:
                output_f32 = output.float()
                losses.append(criterion(target=full_data, input=output_f32))
            loss_dict['inner_loss_pred'] += weight * torch.mean(torch.stack(losses))
        elif key == 'loss_deep_mse':
            assert isinstance(outputs_data, list), "outputs must be a list"
            outputs_f32 = [out.float() for out in outputs_data]
            loss_dict['deep_mse_loss'] += weight * deep_mse_loss(outputs_f32, full_data)
        elif key == 'loss_weighted_mse':
            assert isinstance(outputs_data, list), "outputs must be a list"
            outputs_f32 = [out.float() for out in outputs_data]
            loss_dict['weighted_deep_mse_loss'] += weight * weighted_mse_loss(outputs_f32, full_data)
        elif key == 'loss_wstl':
            assert isinstance(outputs_data, list), "outputs must be a list"
            outputs_f32 = [out.float() for out in outputs_data]
            loss_dict['wstl_loss'] += weight * wstl_loss(outputs_f32, alpha=0.5, beta=0.5, epsilon=1e-6)
    loss = sum(loss_dict.values())
    return loss, loss_dict


def compute_metrics(predx, full, p, data_range=None, modelity='PET', full_mean=None, full_std=None):
    """Compute metrics."""
    if modelity == 'PET':
        from utils.utils import calculate_pet_metrics
        np.seterr(divide='ignore', invalid='ignore')
        pred_img = np.squeeze(predx.cpu().detach().numpy() if torch.is_tensor(predx) else predx, axis=1)
        ref = np.squeeze(full.cpu().detach().numpy() if torch.is_tensor(full) else full, axis=1)
        metrics = calculate_pet_metrics(ref, pred_img)
        return metrics['ssim'], metrics['psnr'], metrics['rmse'], pred_img, ref
    elif modelity == 'CT':
        from utils.utils import get_mean_ssim_ct, get_mean_psnr_ct, get_mean_rmse_ct
        from data.ct_data import CTTools
        cttool = CTTools()
        window_width = p.get('window_width', 3000)
        window_center = p.get('window_center', 500)
        if torch.is_tensor(predx):
            pred_hu = cttool.mu2HU(predx)
            ref_hu = cttool.mu2HU(full)
            pred_hu_np = pred_hu.cpu().detach().numpy() if torch.is_tensor(pred_hu) else pred_hu
            ref_hu_np = ref_hu.cpu().detach().numpy() if torch.is_tensor(ref_hu) else ref_hu
            pred_img = pred_hu_np
            ref = ref_hu_np
        else:
            pred_hu = cttool.mu2HU(predx)
            ref_hu = cttool.mu2HU(full)
        data_range = ref.max() - ref.min()
        ssim = get_mean_ssim_ct(ref, pred_img, data_range=data_range)
        psnr = get_mean_psnr_ct(ref, pred_img, data_range=data_range)
        rmse = get_mean_rmse_ct(ref, pred_img)
        return ssim, psnr, rmse, pred_img, ref
    elif modelity == 'MRI':
        from utils.utils import get_mean_ssim_mri, get_mean_psnr_mri, get_mean_nmse_mri
        if torch.is_tensor(predx):
            predx_np = predx.cpu().detach().numpy()
        else:
            predx_np = predx
        if torch.is_tensor(full):
            full_np = full.cpu().detach().numpy()
        else:
            full_np = full
        if len(predx_np.shape) == 4:
            predx_np = predx_np[:, 0]
        if len(full_np.shape) == 4:
            full_np = full_np[:, 0]
        predx_norm = predx_np.copy()
        full_norm = full_np.copy()
        if full_mean is not None and full_std is not None and p.get('input_normalize') == 'mean_std':
            if torch.is_tensor(full_mean):
                full_mean_np = full_mean.cpu().numpy()
            else:
                full_mean_np = full_mean
            if torch.is_tensor(full_std):
                full_std_np = full_std.cpu().numpy()
            else:
                full_std_np = full_std
            batch_size = predx_np.shape[0]
            if len(full_mean_np.shape) == 1:
                full_mean_np = full_mean_np[:, np.newaxis, np.newaxis]
                full_std_np = full_std_np[:, np.newaxis, np.newaxis]
            full_np_denorm = full_np * full_std_np + full_mean_np
            predx_np_denorm = predx_np * full_std_np + full_mean_np
            full_min = full_np_denorm.min()
            full_max = full_np_denorm.max()
            predx_min = predx_np_denorm.min()
            predx_max = predx_np_denorm.max()
            combined_min = min(full_min, predx_min)
            combined_max = max(full_max, predx_max)
            if combined_max > combined_min:
                full_np_normalized = (full_np_denorm - combined_min) / (combined_max - combined_min)
                predx_np_normalized = (predx_np_denorm - combined_min) / (combined_max - combined_min)
            else:
                full_np_normalized = np.full_like(full_np_denorm, 0.5)
                predx_np_normalized = np.full_like(predx_np_denorm, 0.5)
            full_np_uint8 = (np.clip(full_np_normalized, 0, 1) * 255).astype(np.uint8)
            predx_np_uint8 = (np.clip(predx_np_normalized, 0, 1) * 255).astype(np.uint8)
        else:
            full_np_uint8 = (np.clip(full_np, 0, 1) * 255).astype(np.uint8)
            predx_np_uint8 = (np.clip(predx_np, 0, 1) * 255).astype(np.uint8)
        nmse = get_mean_nmse_mri(full_norm, predx_norm)
        psnr = get_mean_psnr_mri(full_np_uint8, predx_np_uint8)
        ssim = get_mean_ssim_mri(full_np_uint8, predx_np_uint8)
        pred_img = predx_np_uint8
        ref = full_np_uint8
        return ssim, psnr, nmse, pred_img, ref


def visualize_results(ref, pred_img, epoch, p, prefix='', rand_idx=None, imaging_system=None):
    """Visualize results."""
    if not is_main_process():
        return
    if ref is None or pred_img is None:
        return
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    is_ct = False
    if imaging_system is not None:
        imaging_system_type = type(imaging_system).__name__
        if 'CT' in imaging_system_type:
            is_ct = True
    ref_min, ref_max = ref.min(), ref.max()
    pred_min, pred_max = pred_img.min(), pred_img.max()
    data_min = min(ref_min, pred_min)
    data_max = max(ref_max, pred_max)
    is_uint8_range = (ref.dtype == np.uint8 or pred_img.dtype == np.uint8) or data_max > 1.5
    if (data_min < -0.1 or data_max > 1.1) and is_ct:
        from data.ct_data import CTTools
        cttool = CTTools()
        window_width = p.get('window_width', 3000)
        window_center = p.get('window_center', 500)
        ref = cttool.window_transform(ref, width=window_width, center=window_center)
        pred_img = cttool.window_transform(pred_img, width=window_width, center=window_center)
        is_uint8_range = False
    if len(ref.shape) == 3:
        batch_size = ref.shape[0]
    elif len(ref.shape) == 2:
        batch_size = 1
        ref = ref[np.newaxis, :, :]
        pred_img = pred_img[np.newaxis, :, :]
    else:
        return
    num_to_show = min(batch_size, 4)
    indices = list(range(num_to_show))
    ref_images = [np.squeeze(ref[i]) if len(ref.shape) > 2 else ref[i] for i in indices]
    pred_images = [np.squeeze(pred_img[i]) if len(pred_img.shape) > 2 else pred_img[i] for i in indices]
    error_images = [np.abs(pred_images[i] - ref_images[i]) for i in range(len(indices))]
    fig, axes = plt.subplots(3, num_to_show, figsize=(2.5 * num_to_show, 7.5), dpi=200)
    if num_to_show == 1:
        axes = axes.reshape(3, 1)
    row_titles = ['Reference', 'Prediction', 'Error']
    colormap = 'jet'
    if imaging_system is not None:
        imaging_system_type = type(imaging_system).__name__
        if 'CT' in imaging_system_type:
            colormap = 'gray'
        elif 'PET' in imaging_system_type:
            colormap = 'jet'
        elif 'MRI' in imaging_system_type:
            colormap = 'gray'
    if is_uint8_range:
        vmin, vmax = 0, 255
    elif not p.get('do_normalize', False):
        vmin, vmax = ref.min(), ref.max()
    else:
        vmin, vmax = V_MIN, V_MAX
    for row_idx, (row_images, row_title) in enumerate(zip(
        [ref_images, pred_images, error_images],
        row_titles
    )):
        ims = []
        for col_idx in range(num_to_show):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(row_images[col_idx], cmap=colormap, vmin=vmin, vmax=vmax)
            ims.append(im)
            ax.axis('off')
            if col_idx == 0:
                ax.text(-0.1, 0.5, row_title, transform=ax.transAxes,
                       rotation=90, va='center', ha='right', fontsize=12, fontweight='bold')
        if num_to_show > 0:
            cbar = fig.colorbar(ims[-1], ax=axes[row_idx, -1], orientation='vertical',
                              fraction=0.046, pad=0.04)
            if is_uint8_range:
                cbar.set_ticks([vmin, vmax // 2, vmax])
            else:
                cbar.set_ticks([vmin, 0.5, vmax])
            cbar.set_label('Intensity', fontsize=9)
    fig.suptitle(f'Epoch = {epoch}', fontsize=14, y=0.995)
    plt.tight_layout(rect=[0.02, 0, 0.98, 0.98])
    plot_figure = f'{prefix}img_plot_epoch_{epoch}_{int(time.time())}.png'
    plot_path = os.path.join(p['figures_base'], plot_figure)
    fig.savefig(plot_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print("Visualization saved at: " + plot_path)


def process_flowpet_batch(data, input_keys, model, criterion, optimizer, p, imaging_system, device,
                          is_training=True, schedule=None):
    """Process one FlowPET batch."""
    if schedule is None:
        raise ValueError("schedule is required for FlowPET")
    random_select = p.get('random_input_key', True)
    input_key = select_input_key(input_keys, data, random_select=random_select)
    target_key = 'full'
    if input_key not in data or target_key not in data:
        raise ValueError(
            f'Task {input_key}->{target_key} is unavailable; batch keys={list(data.keys())}'
        )
    ref = data[target_key].to(device).float()
    if ref.dim() == 3:
        ref = ref.unsqueeze(1)
    x_1 = data[input_key].to(device).float()
    if x_1.dim() == 3:
        x_1 = x_1.unsqueeze(1)
    data_range_mode = p.get('data_range_mode', '[-1,1]')
    if data_range_mode == '[-1,1]':
        ref = ref * 2.0 - 1.0
        x_1 = x_1 * 2.0 - 1.0
    if data_range_mode != '[0,1]':
        raise ValueError("Released FlowPET uses [0,1] images before forward projection")
    source_measurement_key = f'{input_key}_sinogram'
    measurement_key = p.get('measurement_key', 'sinogram')
    if source_measurement_key in data:
        measurement_y = data[source_measurement_key].to(device).float()
    elif measurement_key in data:
        measurement_y = data[measurement_key].to(device).float()
    else:
        measurement_y = imaging_system.A(x_1)
    x_1, static_condition = measurement_to_image(measurement_y, imaging_system, p)
    condition = static_condition
    loss = None
    loss_dict = None
    if is_training:
        x_0 = ref
        momentum_alpha = p.get('momentum_alpha', 0.0)
        loss_fm_config = p.get('loss_type', {}).get('loss_fm', {})
        if isinstance(loss_fm_config, dict):
            loss_weight = float(loss_fm_config.get('weight', loss_fm_config.get('w', 1.0)))
        else:
            loss_weight = float(loss_fm_config) if loss_fm_config else 1.0
        criterion_name = p.get('criterion', 'mse')
        if isinstance(loss_fm_config, dict) and 'criterion' in loss_fm_config:
            criterion_name = loss_fm_config['criterion']
        criterion_obj = get_criterion(criterion_name)
        loss_vx_weight = p.get('loss_vx_weight', 1.0)
        loss_vp_weight = p.get('loss_vp_weight', 1.0)
        if isinstance(loss_fm_config, dict):
            loss_vx_weight = loss_fm_config.get('loss_vx_weight', loss_vx_weight)
            loss_vp_weight = loss_fm_config.get('loss_vp_weight', loss_vp_weight)
        loss, loss_dict = compute_flow_matching_loss(
            x_0=x_0,
            x_1=x_1,
            velocity_network=model,
            schedule=schedule,
            device=device,
            criterion=criterion_obj,
            loss_weight=loss_weight,
            measurement_y=measurement_y,
            use_constant_auxiliary=p.get('use_constant_auxiliary', False),
            momentum_alpha=momentum_alpha,
            condition=condition,
            imaging_system=imaging_system,
            loss_vx_weight=loss_vx_weight,
            loss_vp_weight=loss_vp_weight,
            p=p,
        )
        gradient_accumulation_steps = p.get('gradient_accumulation_steps', 1)
        normalized_loss = loss / gradient_accumulation_steps
        normalized_loss.backward()
    num_steps = p.get('sampling_steps', 4)
    if not is_training or p.get('sample_during_training', False):
        with torch.no_grad():
            x_1_for_sampling = x_1
            x_0_pred = restore_image(
                x_1=x_1_for_sampling,
                velocity_network=model,
                schedule=schedule,
                device=device,
                num_steps=num_steps,
                condition=condition,
                imaging_system=imaging_system,
                p=p
            )
            predx_f32 = x_0_pred.float()
            full_f32 = ref.float()
            modelity = p.get('imaging_system', 'PET')
            if modelity == 'MRI':
                full_mean = data['full_mean'].to(device).float()
                full_std = data['full_std'].to(device).float()
                ssim, psnr, rmse, pred_img, ref_img = compute_metrics(predx_f32, full_f32, p, modelity=modelity, full_mean=full_mean, full_std=full_std)
            else:
                if data_range_mode == '[-1,1]' and p.get('do_normalize', True):
                    predx_for_metrics = torch.clamp((x_0_pred + 1.0) / 2.0, 0.0, 1.0)
                    full_for_metrics = torch.clamp((ref + 1.0) / 2.0, 0.0, 1.0)
                    data_range = 2.0
                elif data_range_mode == '[0,1]' and p.get('do_normalize', True):
                    predx_for_metrics = torch.clamp(x_0_pred, 0.0, 1.0)
                    full_for_metrics = torch.clamp(ref, 0.0, 1.0)
                    data_range = 1.0
                elif not p.get('do_normalize', False):
                    predx_for_metrics = x_0_pred
                    full_for_metrics = ref
                    data_range = ref.max() - ref.min()
                ssim, psnr, rmse, pred_img, ref_img = compute_metrics(predx_for_metrics, full_for_metrics, p, data_range=data_range, modelity=modelity)
    else:
        ssim, psnr, rmse = 0.0, 0.0, 0.0
        pred_img = None
        ref_img = None
    return loss, loss_dict, ssim, psnr, rmse, pred_img, ref_img


def process_batch(data, input_keys, model, criterion, optimizer, p, imaging_system, device,
                  is_training=True, tokenizer=None):
    """Process batch."""
    input, full, prior = prepare_input_data(data, input_keys, p, imaging_system, device, is_training=is_training, tokenizer=tokenizer)
    outputs, diff, predx = forward_pass(model, input, prior, p.get('use_prior', False))
    predx_latent = None
    outputs_latent = None
    full_latent = None
    if tokenizer is not None:
        predx_latent = predx.clone()  # [B, 4, H/8, W/8]
        if isinstance(outputs, list):
            outputs_latent = [out.clone() for out in outputs]
        else:
            outputs_latent = outputs.clone() if hasattr(outputs, 'clone') else outputs
        full_latent = full.clone()  # [B, 4, H/8, W/8]
        predx_decoded = tokenizer.decode(predx, to_01_range=True)
        predx = predx_decoded[:, 0:1, :, :]  # [B, 1, H, W]
        if isinstance(outputs, list):
            outputs_decoded = []
            for out in outputs:
                out_decoded = tokenizer.decode(out, to_01_range=True)  # [B, 3, H, W]
                out_decoded = out_decoded[:, 0:1, :, :]  # [B, 1, H, W]
                outputs_decoded.append(out_decoded)
            outputs = outputs_decoded
        full_decoded = tokenizer.decode(full, to_01_range=True)
        full = full_decoded[:, 0:1, :, :]  # [B, 1, H, W]
        full_for_loss = full
        full_for_metrics = full
    else:
        full_for_loss = full
        full_for_metrics = full
        if isinstance(outputs, list):
            for out in outputs:
                outputs.append(out)
        elif not isinstance(outputs, list):
            outputs = outputs
    loss = None
    loss_dict = None
    if is_training:
        loss, loss_dict = compute_loss(
            predx, outputs, diff, full_for_loss, criterion, p,
            predx_latent=predx_latent, outputs_latent=outputs_latent,
            full_latent=full_latent, tokenizer=tokenizer
        )
        gradient_accumulation_steps = p.get('gradient_accumulation_steps', 1)
        normalized_loss = loss / gradient_accumulation_steps
        normalized_loss.backward()
        total_norm = 0.0
        param_count = 0
        nan_grad_count = 0
        inf_grad_count = 0
        max_grad = -float('inf')
        min_grad = float('inf')
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_count += 1
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                grad_max = param.grad.data.max().item()
                grad_min = param.grad.data.min().item()
                max_grad = max(max_grad, grad_max)
                min_grad = min(min_grad, grad_min)
        total_norm = total_norm ** (1. / 2)
    data_range = 1.0
    predx_for_metrics = predx
    predx_f32 = predx_for_metrics.float()
    full_f32 = full_for_metrics.float()
    modelity = p.get('imaging_system', 'PET')
    ssim, psnr, rmse, pred_img, ref = compute_metrics(predx_f32, full_f32, data_range=data_range,modelity=modelity)
    return loss, loss_dict, ssim, psnr, rmse, pred_img, ref


def flowpet_optimizer_step(optimizer, p):
    """Count optimizer updates, and advance the original scheduler once per update."""
    optimizer.step()
    p['global_step'] = p.get('global_step', 0) + 1
    scheduler = getattr(optimizer, '_flowpet_scheduler', None)
    if scheduler is not None:
        scheduler.step()
    optimizer.zero_grad()


def unrolling_train(train_loader, model, criterion, optimizer, epoch, device, p, imaging_system=None, tokenizer=None):
    """Train FlowPET for one epoch."""
    start_time = time.time()
    loss_record = []
    ssim_record = []
    psnr_record = []
    rmse_record = []
    train_input_keys = p.get('train_input_keys', None)
    if train_input_keys is None:
        train_input_keys = ['prior', 'ultra_ultra_low', 'ultra_low', 'low']
    schedule = DegradationSchedule()
    gradient_accumulation_steps = p.get('gradient_accumulation_steps', 1)
    model.train()
    if is_main_process():
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    else:
        progress_bar = enumerate(train_loader)
    optimizer.zero_grad()
    for batch_idx, sample in progress_bar:
        if p.get('max_steps') and p.get('global_step', 0) >= p['max_steps']:
            break
        is_last_step = (batch_idx + 1) % gradient_accumulation_steps == 0
        loss, loss_dict, ssim, psnr, rmse, pred_img, ref = process_flowpet_batch(
            sample, train_input_keys, model, criterion, optimizer, p, imaging_system, device,
            is_training=True, schedule=schedule,
        )
        loss = loss.detach()
        if is_last_step:
            flowpet_optimizer_step(optimizer, p)
        loss_record.append(loss.item())
        ssim_record.append(ssim)
        psnr_record.append(psnr)
        rmse_record.append(rmse)
        if is_main_process() and isinstance(progress_bar, tqdm):
            postfix_dict = {}
            if loss_dict:
                for key, value in loss_dict.items():
                    if value.item() != 0.0:
                        if key == 'flow_matching_loss':
                            postfix_key = 'Loss'
                        elif key == 'mse_loss':
                            postfix_key = 'MSE'
                        else:
                            postfix_key = key.replace('loss_', '').replace('_', ' ').title()
                        postfix_dict[postfix_key] = value.item()
            progress_bar.set_postfix(**postfix_dict)
    if (gradient_accumulation_steps > 1 and len(train_loader) % gradient_accumulation_steps != 0
            and (not p.get('max_steps') or p.get('global_step', 0) < p['max_steps'])):
        # Correct the normalization of a final, shorter accumulation window.
        remainder = len(train_loader) % gradient_accumulation_steps
        for group in optimizer.param_groups:
            for parameter in group['params']:
                if parameter.grad is not None:
                    parameter.grad.mul_(gradient_accumulation_steps / remainder)
        flowpet_optimizer_step(optimizer, p)
    avg_loss = np.mean(np.array(loss_record, dtype=np.float32))
    avg_ssim = np.mean(np.array(ssim_record, dtype=np.float32))
    avg_psnr = np.mean(np.array(psnr_record, dtype=np.float32))
    avg_rmse = np.mean(np.array(rmse_record, dtype=np.float32))
    if is_distributed():
        loss_tensor = torch.tensor(avg_loss, device=device)
        ssim_tensor = torch.tensor(avg_ssim, device=device)
        psnr_tensor = torch.tensor(avg_psnr, device=device)
        rmse_tensor = torch.tensor(avg_rmse, device=device)
        loss_tensor = reduce_tensor(loss_tensor)
        ssim_tensor = reduce_tensor(ssim_tensor)
        psnr_tensor = reduce_tensor(psnr_tensor)
        rmse_tensor = reduce_tensor(rmse_tensor)
        avg_loss = loss_tensor.cpu().item()
        avg_ssim = ssim_tensor.cpu().item()
        avg_psnr = psnr_tensor.cpu().item()
        avg_rmse = rmse_tensor.cpu().item()
    return avg_loss, avg_ssim, avg_psnr, avg_rmse


def unrolling_val(val_loader, model, criterion, optimizer, epoch, device, p, imaging_system=None, tokenizer=None):
    """Validate FlowPET for one epoch."""
    model.eval()
    ssim_record = []
    psnr_record = []
    rmse_record = []
    val_input_keys = p.get('val_input_keys', None)
    if val_input_keys is None:
        val_input_keys = ['ultra_ultra_low', 'ultra_low', 'low']
    schedule = DegradationSchedule()
    val_sampling_steps = p.get('val_sampling_steps', p.get('sampling_steps', 4))
    if is_main_process():
        np.random.seed(epoch * 1000 + len(val_loader))
        random_viz_batch_idx = np.random.randint(0, len(val_loader))
    else:
        random_viz_batch_idx = -1
    with torch.no_grad():
        if is_main_process():
            progress_bar = tqdm(enumerate(val_loader), total=len(val_loader))
        else:
            progress_bar = enumerate(val_loader)
        for batch_idx, sample in progress_bar:
            p_val = dict(p)
            p_val['sampling_steps'] = val_sampling_steps
            _, _, ssim, psnr, rmse, pred_img, ref = process_flowpet_batch(
                sample, val_input_keys, model, criterion, None, p_val, imaging_system, device,
                is_training=False, schedule=schedule
            )
            ssim_record.append(ssim)
            psnr_record.append(psnr)
            rmse_record.append(rmse)
            if is_main_process() and isinstance(progress_bar, tqdm):
                progress_bar.set_postfix(Ssim=ssim, Psnr=psnr, Rmse=rmse)
            if batch_idx == random_viz_batch_idx:
                if is_main_process():
                    np.random.seed(epoch * 1000 + batch_idx * 100)
                    rand_idx = np.random.randint(0, len(ref))
                    visualize_results(ref, pred_img, epoch, p, prefix='val_', rand_idx=rand_idx, imaging_system=imaging_system)
    model.train()
    avg_ssim = np.mean(np.array(ssim_record, dtype=np.float32))
    avg_psnr = np.mean(np.array(psnr_record, dtype=np.float32))
    avg_rmse = np.mean(np.array(rmse_record, dtype=np.float32))
    if is_distributed():
        ssim_tensor = torch.tensor(avg_ssim, device=device)
        psnr_tensor = torch.tensor(avg_psnr, device=device)
        rmse_tensor = torch.tensor(avg_rmse, device=device)
        ssim_tensor = reduce_tensor(ssim_tensor)
        psnr_tensor = reduce_tensor(psnr_tensor)
        rmse_tensor = reduce_tensor(rmse_tensor)
        avg_ssim = ssim_tensor.cpu().item()
        avg_psnr = psnr_tensor.cpu().item()
        avg_rmse = rmse_tensor.cpu().item()
    return avg_ssim, avg_psnr, avg_rmse
