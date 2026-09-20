"""
Authors: Xingyu Xie
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import random
import os
import errno

import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
from math import exp
import torch.nn.functional as F
def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def partition_list(case_num, train_num, sample_per_case, random_seed, train=True, def_train_list=False):
    """
    This function returns the train/test list by provided random number
    :param case_num: The total case number
    :param train_num: The number of case in train_set
    :param sample_per_case: Number of cases per case
    :param random_seed: a fixed number to control the random process
    :param def_train_list: a pre-set train_list
    :return: Partitioned slice list
    """
    sample_list = list(range(0, case_num))
    if not def_train_list:
        # Shuffle the list
        random.seed(random_seed)
        random.shuffle(sample_list)
        if train:
            data_case_list = sample_list[:train_num]
            data_list = [case * sample_per_case + i for case in data_case_list for i in range(sample_per_case)]
        else:
            data_case_list = sample_list[train_num:]
            data_list = [case * sample_per_case + i for case in data_case_list for i in range(sample_per_case)]
    else:
        raise NotImplementedError("Pre-set train_list is not implemented yet")
    return data_list


def circular_mask(input, sz, r):
    # Get the batch size
    N = input.shape[0]
    # Create a grid of coordinates
    x, y = np.meshgrid(np.arange(sz), np.arange(sz))
    # Calculate the distance from the center
    d = np.sqrt((x - sz // 2) ** 2 + (y - sz // 2) ** 2)
    # Create a boolean mask where True is inside the circle and False is outside
    m = d < r
    # Expand the mask to match the batch dimension
    m = np.expand_dims(m, axis=0)
    m = np.repeat(m, N, axis=0)
    # Multiply the input array by the mask
    return input * m


def calculate_pet_metrics(ref_stack, pred_stack):
    """Compute slice-mean PET metrics on the released [0, 1] scale."""
    ref_stack = np.asarray(ref_stack, dtype=np.float64)
    pred_stack = np.asarray(pred_stack, dtype=np.float64)
    if ref_stack.ndim == 2:
        ref_stack = ref_stack[None]
    if pred_stack.ndim == 2:
        pred_stack = pred_stack[None]
    if ref_stack.shape != pred_stack.shape or ref_stack.ndim != 3:
        raise ValueError(
            f'Expected matching [N,H,W] PET arrays, got {ref_stack.shape} and {pred_stack.shape}'
        )
    ref_stack = np.clip(ref_stack, 0.0, 1.0)
    pred_stack = np.clip(pred_stack, 0.0, 1.0)
    ssim_values = []
    psnr_values = []
    rmse_values = []
    for ref, pred in zip(ref_stack, pred_stack):
        ssim_values.append(ssim(ref, pred, data_range=1.0))
        psnr_values.append(peak_signal_noise_ratio(ref, pred, data_range=1.0))
        rmse_values.append(np.sqrt(np.mean((ref - pred) ** 2)))
    return {
        'ssim': float(np.mean(ssim_values)),
        'psnr': float(np.mean(psnr_values)),
        'rmse': float(np.mean(rmse_values)),
    }


def get_mean_ssim(ref_stack, pred_stack, normalize_method=None, data_range=1.0):
    """Return slice-mean SSIM for [0, 1] PET images."""
    if normalize_method is not None:
        raise ValueError('The released metric protocol does not shift images')
    return calculate_pet_metrics(ref_stack, pred_stack)['ssim']


def get_mean_psnr(ref_stack, pred_stack, data_range=1.0):
    """Return slice-mean PSNR for [0, 1] PET images."""
    return calculate_pet_metrics(ref_stack, pred_stack)['psnr']


def get_mean_rmse(ref_stack, pred_stack):
    """Return slice-mean RMSE for [0, 1] PET images."""
    return calculate_pet_metrics(ref_stack, pred_stack)['rmse']


def _compute_MSE_cvg(img1, img2):
    """Compute mean squared error."""
    return ((img1 - img2) ** 2).mean()


def _gaussian_cvg(window_size, sigma):
    """Create a normalized Gaussian kernel."""
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window_cvg(window_size, channel, spatial_dims=2):
    """Create an SSIM window."""
    _1D_window = _gaussian_cvg(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    if spatial_dims == 2:
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    else:
        window = _2D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window


def _compute_SSIM_single_cvg(img1, img2, data_range, window_size=11, channel=1, size_average=True, spatial_dims=2):
    """Compute SSIM for one image pair."""
    if not torch.is_tensor(img1):
        img1 = torch.from_numpy(img1).float()
    if not torch.is_tensor(img2):
        img2 = torch.from_numpy(img2).float()
    device = img1.device if torch.is_tensor(img1) else 'cpu'
    if torch.is_tensor(img2):
        img2 = img2.to(device)
    else:
        img2 = torch.from_numpy(img2).float().to(device)
    if len(img1.size()) == 2:
        shape_ = img1.shape
        img1 = img1.view(1, 1, *shape_)
        img2 = img2.view(1, 1, *shape_)
    window = _create_window_cvg(window_size, channel, spatial_dims=spatial_dims)
    window = window.type_as(img1).to(device)
    conv_op = F.conv2d if spatial_dims == 2 else F.conv3d
    mu1 = conv_op(img1, window, padding=window_size//2)
    mu2 = conv_op(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2
    sigma1_sq = conv_op(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = conv_op(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12 = conv_op(img1*img2, window, padding=window_size//2) - mu1_mu2
    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


def _compute_PSNR_single_cvg(img1, img2, data_range):
    """Compute PSNR for one image pair."""
    eps = 1e-10
    mse_ = _compute_MSE_cvg(img1, img2)
    if mse_ == 0:
        mse_ += eps
    if torch.is_tensor(img1):
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        return 10 * np.log10((data_range ** 2) / mse_)


def _compute_RMSE_single_cvg(img1, img2):
    """Compute RMSE for one image pair."""
    if type(img1) == torch.Tensor:
        return torch.sqrt(_compute_MSE_cvg(img1, img2)).item()
    else:
        return np.sqrt(_compute_MSE_cvg(img1, img2))


def _compute_SSIM_batch_cvg(img1, img2, data_range, window_size=11, channel=1, size_average=True, spatial_dims=2):
    """Compute SSIM for a batch."""
    if not torch.is_tensor(img1):
        img1 = torch.from_numpy(img1).float()
    if not torch.is_tensor(img2):
        img2 = torch.from_numpy(img2).float()
    device = img1.device if torch.is_tensor(img1) else 'cpu'
    if torch.is_tensor(img2):
        img2 = img2.to(device)
    else:
        img2 = torch.from_numpy(img2).float().to(device)
    if len(img1.size()) == 2:
        shape_ = img1.shape
        img1 = img1.view(1, 1, *shape_)
        img2 = img2.view(1, 1, *shape_)
    elif len(img1.size()) == 3:
        img1 = img1.unsqueeze(1)
        img2 = img2.unsqueeze(1)
    window = _create_window_cvg(window_size, channel, spatial_dims=spatial_dims)
    window = window.type_as(img1)
    conv_op = F.conv2d if spatial_dims == 2 else F.conv3d
    mu1 = conv_op(img1, window, padding=window_size//2)
    mu2 = conv_op(img2, window, padding=window_size//2)
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1*mu2
    sigma1_sq = conv_op(img1*img1, window, padding=window_size//2) - mu1_sq
    sigma2_sq = conv_op(img2*img2, window, padding=window_size//2) - mu2_sq
    sigma12 = conv_op(img1*img2, window, padding=window_size//2) - mu1_mu2
    C1, C2 = (0.01*data_range)**2, (0.03*data_range)**2
    ssim_map = ((2*mu1_mu2+C1)*(2*sigma12+C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1).item()


def _compute_PSNR_batch_cvg(img1, img2, data_range):
    """Compute PSNR for a batch."""
    eps = 1e-10
    mse_ = _compute_MSE_cvg(img1, img2)
    if mse_ == 0:
        mse_ += eps
    if torch.is_tensor(img1):
        return 10 * torch.log10((data_range ** 2) / mse_).item()
    else:
        return 10 * np.log10((data_range ** 2) / mse_)


def _compute_RMSE_batch_cvg(img1, img2):
    """Compute RMSE for a batch."""
    if type(img1) == torch.Tensor:
        return torch.sqrt(_compute_MSE_cvg(img1, img2)).item()
    else:
        return np.sqrt(_compute_MSE_cvg(img1, img2))


def get_mean_ssim_ct(ref_stack, pred_stack, data_range=1.0, spatial_dims=2):
    """Return mean ssim ct."""
    if not torch.is_tensor(ref_stack):
        ref_stack = torch.from_numpy(ref_stack).float()
    if not torch.is_tensor(pred_stack):
        pred_stack = torch.from_numpy(pred_stack).float()
    device = ref_stack.device if torch.is_tensor(ref_stack) else 'cpu'
    pred_stack = pred_stack.to(device)
    ssim_val = _compute_SSIM_batch_cvg(ref_stack, pred_stack, data_range=data_range, spatial_dims=spatial_dims)
    if np.isnan(ssim_val):
        ssim_val = 0.0
    return ssim_val


def get_mean_psnr_ct(ref_stack, pred_stack, data_range=1.0):
    """Return mean psnr ct."""
    if not torch.is_tensor(ref_stack):
        ref_stack = torch.from_numpy(ref_stack).float()
    if not torch.is_tensor(pred_stack):
        pred_stack = torch.from_numpy(pred_stack).float()
    device = ref_stack.device if torch.is_tensor(ref_stack) else 'cpu'
    pred_stack = pred_stack.to(device)
    psnr_val = _compute_PSNR_batch_cvg(ref_stack, pred_stack, data_range)
    if np.isnan(psnr_val):
        psnr_val = 0.0
    return psnr_val


def get_mean_rmse_ct(ref_stack, pred_stack):
    """Return mean rmse ct."""
    if not torch.is_tensor(ref_stack):
        ref_stack = torch.from_numpy(ref_stack).float()
    if not torch.is_tensor(pred_stack):
        pred_stack = torch.from_numpy(pred_stack).float()
    device = ref_stack.device if torch.is_tensor(ref_stack) else 'cpu'
    pred_stack = pred_stack.to(device)
    rmse_val = _compute_RMSE_batch_cvg(ref_stack, pred_stack)
    if np.isnan(rmse_val) or np.isinf(rmse_val):
        rmse_val = 0.0
    return rmse_val


def get_mean_nmse_mri(ref_stack, pred_stack):
    """Return mean nmse mri."""
    ref_norm = np.linalg.norm(ref_stack)
    if ref_norm == 0:
        pred_norm = np.linalg.norm(pred_stack)
        if pred_norm == 0:
            nmse_val = 0.0
        else:
            nmse_val = 1.0
    else:
        # NMSE = ||ref - pred||² / ||ref||²
        diff_norm = np.linalg.norm(ref_stack - pred_stack)
        nmse_val = (diff_norm ** 2) / (ref_norm ** 2)
    if np.isnan(nmse_val) or np.isinf(nmse_val):
        nmse_val = 0.0
    return nmse_val


def get_mean_psnr_mri(ref_stack, pred_stack):
    """Return mean psnr mri."""
    ref_max = 255.0
    if ref_max == 0:
        psnr_val = 0.0
    else:
        psnr_val = peak_signal_noise_ratio(ref_stack, pred_stack, data_range=ref_max)
    if np.isnan(psnr_val):
        psnr_val = 0.0
    return psnr_val


def get_mean_ssim_mri(ref_stack, pred_stack):
    """Return mean ssim mri."""
    maxval = 255.0
    if maxval == 0:
        return 0.0
    ssim_list = []
    for i in range(ref_stack.shape[0]):
        ref = ref_stack[i]
        pred = pred_stack[i]
        ssim_val = ssim(ref, pred, data_range=maxval)
        if np.isnan(ssim_val):
            ssim_val = 0
        ssim_list.append(ssim_val)
    return np.mean(ssim_list)


def get_mean_rmse_mri(ref_stack, pred_stack):
    """Return mean rmse mri."""
    rmse_list = []
    for i in range(ref_stack.shape[0]):
        ref = ref_stack[i]
        pred = pred_stack[i]
        mse = np.mean((ref - pred) ** 2)
        rmse_val = np.sqrt(mse)
        if np.isnan(rmse_val) or np.isinf(rmse_val):
            rmse_val = 0.0
        rmse_list.append(rmse_val)
    mean_rmse = np.mean(rmse_list)
    if np.isnan(mean_rmse) or np.isinf(mean_rmse):
        mean_rmse = 0.0
    return mean_rmse


def clip_grad_norm(optimizer, max_norm, norm_type=2):
    """
    Clip the norm of the gradients for all parameters under `optimizer`.
    Args:
    optimizer (torch.optim.Optimizer):
    max_norm (float): The maximum allowable norm of gradients.
    norm_type (int): The type of norm to use in computing gradient norms.
    """
    for group in optimizer.param_groups:
        torch.nn.utils.clip_grad_norm_(group['params'], max_norm, norm_type)


def draw_data(img):
    plt.imshow(img, cmap='jet')  # Use 'cmap' to specify the color map
    plt.colorbar()  # Add a colorbar to the plot (optional)
    plt.title('Matrix as Image')  # Add a title (optional)
    plt.show()


class Fourier_Utils():
    def __init__(self):
        pass

    @staticmethod
    def fft2d(x):
        """2D Fourier Transform"""
        orig_dtype = x.dtype
        if orig_dtype == torch.float16:
            x = x.float()
        result = torch.fft.fftshift(torch.fft.fft2(x))
        return result

    @staticmethod
    def ifft2d(x):
        """2D Inverse Fourier Transform"""
        x_unshifted = torch.fft.ifftshift(x)
        result = torch.fft.ifft2(x_unshifted)
        return result

    @staticmethod
    def get_amplitude(x):
        """Return the amplitude spectrum."""
        x_fft = Fourier_Utils.fft2d(x)
        return torch.abs(x_fft)

    @staticmethod
    def get_phase(x):
        """Return the phase spectrum."""
        x_fft = Fourier_Utils.fft2d(x)
        return torch.angle(x_fft)

    @staticmethod
    def normalize_amplitude(amp):
        # amp: (B, C, H, W)
        amp_log = torch.log(amp + 1)
        min_v = amp_log.amin(dim=[2,3], keepdim=True)
        max_v = amp_log.amax(dim=[2,3], keepdim=True)
        return (amp_log - min_v) / (max_v - min_v + 1e-8)

    @staticmethod
    def phase_to_sincos(phase):
        """Convert phase to sin and cos dual channels"""
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)

    @staticmethod
    def sincos_to_phase(sincos):
        """Handle to phase."""
        C2 = sincos.shape[1]
        assert C2 % 2 == 0, "Channel dim must be even"
        C = C2 // 2
        sin = sincos[:, :C, :, :]
        cos = sincos[:, C:, :, :]
        return torch.atan2(sin, cos)

    @staticmethod
    def get_phase_and_amp(x):
        fft = Fourier_Utils.fft2d(x)
        amp = torch.abs(fft)
        amp_norm = Fourier_Utils.normalize_amplitude(amp)
        phase = torch.angle(fft)
        phase_sincos = Fourier_Utils.phase_to_sincos(phase)
        return  phase_sincos, amp_norm

    @staticmethod
    def get_phase_amp_loss(pred, target):
        pred_phase, pred_amp = Fourier_Utils.get_phase_and_amp(pred)
        target_phase, target_amp = Fourier_Utils.get_phase_and_amp(target)
        phase_loss = torch.mean(torch.abs(pred_phase - target_phase))
        amp_loss = torch.mean(torch.abs(pred_amp - target_amp))
        return phase_loss, amp_loss

    @staticmethod
    def get_fourier_loss(pred, target):
        pred_f = Fourier_Utils.fft2d(pred)
        target_f = Fourier_Utils.fft2d(target)
        f_loss = torch.mean(torch.abs(pred_f - target_f))
        return f_loss
