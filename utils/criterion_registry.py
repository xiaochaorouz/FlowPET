"""Utilities for criterion registry."""
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
from pytorch_msssim import SSIM


class ROI_mse(nn.Module):
    """Roi mse implementation."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.mse_none = nn.MSELoss(reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_map = self.mse_none(input=input, target=target)
        mask = (target != 0).float()
        masked_loss = loss_map * mask
        loss = masked_loss.sum() / (mask.sum() + self.eps)
        return loss


def deep_mse_loss(outputs,gt):
    total_mse=[]
    for output in outputs:
        mse = F.mse_loss(output, gt)
        total_mse.append(mse)
    return torch.mean(torch.stack(total_mse))


def weighted_mse_loss(outputs, gt, p=4.0):
    """Handle mse loss."""
    T = len(outputs)
    ws = torch.tensor([((t+1)/T) ** p for t in range(T)],
                      device=outputs[0].device,
                      dtype=outputs[0].dtype)
    ws = ws / ws.sum()
    loss = 0.0
    for w, out in zip(ws, outputs):
        loss = loss + w * F.mse_loss(out, gt)
    return loss


def wstl_loss(outputs, alpha=0.5, beta=0.5, epsilon=1e-6):
    """
    Compute the Wavelet-Spectral Trajectory Loss (WSTL) for a list of iterates.

    Args:
        outputs (List[torch.Tensor]): List of tensors x^k of shape (B, C, H, W).
        alpha (float): Weight for HH subband convergence term.
        beta (float): Weight for LL subband convergence term.
        epsilon (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Scalar WSTL value.
    """
    device = outputs[0].device
    K = len(outputs)
    # Initialize wavelet transform (1-level Haar)
    dwt = DWTForward(J=1, wave='haar', mode='zero').to(device)
    spec_loss = 0.0
    ll_loss = 0.0
    hh_loss = 0.0
    for k in range(1, K):
        # FFT of current and previous iterates
        out_k = outputs[k].float() if outputs[k].dtype == torch.float16 else outputs[k]
        out_k1 = outputs[k-1].float() if outputs[k-1].dtype == torch.float16 else outputs[k-1]
        Xk   = torch.fft.rfft2(out_k,   norm='ortho')
        Xk1  = torch.fft.rfft2(out_k1, norm='ortho')
        # Spectral trajectory term
        num = torch.norm(Xk.abs() - Xk1.abs(), p='fro')**2
        den = torch.norm(Xk1.abs(), p='fro')**2 + epsilon
        spec_loss += num / den
        # Wavelet decomposition of current and previous iterates
        Yl_k,  Yh_k  = dwt(outputs[k])
        Yl_k1, Yh_k1 = dwt(outputs[k-1])
        # LL subband amplitude convergence
        ll_loss += torch.norm(Yl_k - Yl_k1, p='fro')**2
        # HH subband detail convergence (using the HH channel of the first level)
        hh_curr = Yh_k[0][..., 2]   # shape (B, C, H', W')
        hh_prev = Yh_k1[0][..., 2]
        hh_loss += torch.norm(hh_curr - hh_prev, p='fro')**2
    # Normalize by number of intervals
    wstl = (spec_loss + alpha * hh_loss + beta * ll_loss) / (K - 1)
    return wstl


def compute_geometric_idempotent_loss(v_theta, z, delta_t=0.1):
    """Compute geometric idempotent loss."""
    v1 = v_theta(z)  # [B, C, H, W]
    z1 = z + delta_t * v1
    z1_detached = z1.detach()
    v2 = v_theta(z1_detached)
    loss = (v2 ** 2).mean()
    return loss


def compute_cross_dose_consistency_loss(v_theta, z_low, z_high, gamma=0.1):
    """Compute cross dose consistency loss."""
    v_low = v_theta(z_low)
    v_high = v_theta(z_high)
    consistency_term = ((v_low - v_high) ** 2).mean()
    high_zero_term = gamma * (v_high ** 2).mean()
    loss = consistency_term + high_zero_term
    return loss


def compute_hk_kinetic_energy_loss(v_list, r_list, x_list, eta=1.0, delta_t=0.1, weight_mode='identity', epsilon=1e-6):
    """Compute hk kinetic energy loss."""
    K = len(v_list)
    if K == 0:
        return torch.tensor(0.0, device=v_list[0].device if len(v_list) > 0 else None)
    device = v_list[0].device
    total_energy = torch.tensor(0.0, device=device)
    for k in range(K):
        v_k = v_list[k]  # [B, C, H, W]
        r_k = r_list[k]  # [B, C, H, W]
        v_norm_sq = torch.sum(v_k ** 2, dim=1, keepdim=True)  # [B, 1, H, W]
        r_norm_sq = torch.sum(r_k ** 2, dim=1, keepdim=True)  # [B, 1, H, W]
        kinetic = v_norm_sq + eta * r_norm_sq  # [B, 1, H, W]
        E_k = torch.mean(kinetic)
        total_energy = total_energy + E_k
    loss = (delta_t / K) * total_energy
    return loss


def compute_physical_consistency_loss(imaging_system, pred, y_measurement):
    """Compute physical consistency loss."""
    assert pred.shape[1] == 1, "x_pred should be in image domain"
    y_pred = imaging_system.A(pred)  # [B, C, N_ang, N_bins]
    loss = ((y_pred - y_measurement) ** 2).mean()
    return loss


def compute_ssim_loss(pred, ref):
    """Compute ssim loss."""
    assert pred.shape[1] == 1, "x_pred should be in image domain"
    assert ref.shape[1] == 1, "x_high should be in image domain"
    data_range = ref.max() - ref.min()
    if data_range <= 0:
        data_range = 1.0
    ssim_module = SSIM(
        data_range=float(data_range),
        size_average=True,
        channel=ref.shape[1]
    ).to(ref.device)
    ssim_val = ssim_module(ref, pred)
    ssim_loss = 1.0 - ssim_val
    if torch.isnan(ssim_loss) or torch.isinf(ssim_loss):
        ssim_loss = torch.tensor(0.0, device=pred.device)
    return ssim_loss


def compute_gradient_penalty(critic, x_real, x_fake):
    """Compute gradient penalty."""
    batch_size = x_real.size(0)
    device = x_real.device
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    alpha = alpha.expand_as(x_real)
    x_interpolated = alpha * x_real + (1 - alpha) * x_fake
    x_interpolated.requires_grad_(True)
    d_interpolated = critic(x_interpolated)
    grad = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=x_interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    grad_norm = torch.sqrt(torch.sum(grad ** 2, dim=(1, 2, 3)) + 1e-10)
    penalty = torch.mean((grad_norm - 1.0) ** 2)
    return penalty

CRITERION_MAP = {
    'mse_loss': nn.MSELoss,
    'sl1_loss': nn.SmoothL1Loss,
    'ssim_loss': compute_ssim_loss,
    'idempotent_loss': compute_geometric_idempotent_loss,
    'cross_dose_consistency_loss': compute_cross_dose_consistency_loss,
    'gradient_penalty': compute_gradient_penalty,
    'physical_consistency_loss': compute_physical_consistency_loss,
    'hk_kinetic_energy_loss': compute_hk_kinetic_energy_loss,
    'roi_mse_loss': ROI_mse,
    'deep_mse_loss': deep_mse_loss,
    'weighted_mse_loss': weighted_mse_loss,
    'wstl_loss': wstl_loss,
}


def get_criterion(criterion_name):
    """Return criterion."""
    alias_map = {
        'sl1': 'sl1_loss',
        'mse': 'mse_loss',
    }
    if criterion_name in alias_map:
        criterion_name = alias_map[criterion_name]
    if criterion_name not in CRITERION_MAP:
        raise ValueError(f'Invalid criterion: {criterion_name}. Available criteria: {list(CRITERION_MAP.keys())}')
    obj = CRITERION_MAP[criterion_name]
    if isinstance(obj, type) and issubclass(obj, nn.Module):
        return obj()
    return obj
