import torch
import torch.nn as nn
import numpy as np
from torch_radon import ParallelBeam


class PET():
    """Parallel-beam PET operators and low-dose simulation."""
    def __init__(self, del_count, angles_count=180, circle_mask=True, device='cuda:0'):
        self.del_count = del_count
        self.angles_count = angles_count
        self.circle_mask = circle_mask
        self.device = device
        # Create angles (uniform from 0 to pi)
        angles = np.linspace(0, np.pi, angles_count, endpoint=False)
        # Create ParallelBeam projector
        self.radon = ParallelBeam(det_count=del_count, angles=angles, volume=del_count)
        # Expose volume for compatibility
        self.volume = self.radon.volume
        # Normalization factor (same as IRadon)
        self.scale_factor = np.pi / (2 * angles_count)
        # Create circular mask if needed
        if circle_mask:
            y_grid, x_grid = torch.meshgrid(
                torch.linspace(-1, 1, del_count),
                torch.linspace(-1, 1, del_count),
                indexing='ij'
            )
            self.circular_mask = (x_grid**2 + y_grid**2) <= 1
            self.circular_mask = self.circular_mask.to(device)
        else:
            self.circular_mask = None
    def A(self, x):
        """
        Forward projection (clean): image -> sinogram

        Args:
            x: Image tensor (B, C, H, W) - must be in image domain (C=1 for PET)
        Returns:
            Sinogram (B, C, Angle, Detector)
        """
        assert x.shape[1] == 1, (
            f"imaging_system.A() only works in image domain, but received {x.shape[1]}-channel input. "
            f"Latent domain data (typically 4 channels) must be decoded to image domain first. "
            f"Input shape: {x.shape}"
        )
        return self.radon.forward(x)

    def A_degrade(self, x, count=2e5, thresh=None):
        """
        Forward projection + low-dose degradation: image -> noisy sinogram

        Args:
            x: Full-dose image (B, C, H, W) - must be in image domain (C=1 for PET)
            count: Photon count (lower = more noise, e.g., 2e5, 1e5, 5e4)
            thresh: Optional threshold for clipping input
        Returns:
            Degraded sinogram (B, C, Angle, Detector)
        """
        assert x.shape[1] == 1, (
            f"imaging_system.A_degrade() only works in image domain, but received {x.shape[1]}-channel input. "
            f"Latent domain data (typically 4 channels) must be decoded to image domain first. "
            f"Input shape: {x.shape}"
        )
        # Optional thresholding/normalization
        if thresh is not None:
            x = torch.clip(x, min=0, max=thresh) / thresh
        # Forward projection (clean sinogram)
        proj = self.radon.forward(x)
        # Multiplicative factor (simulates attenuation/scatter variation ±10%)
        mul_factor = torch.ones_like(proj)
        mul_factor = mul_factor + (torch.rand_like(mul_factor) * 0.2 - 0.1)
        # Additive noise (simulates random coincidences, 20% of mean)
        noise = torch.ones_like(proj) * torch.mean(mul_factor * proj, dim=(-1, -2), keepdims=True) * 0.2
        # Apply degradation
        sinogram = mul_factor * proj + noise
        # Scale to desired count level
        cs = count / (1e-9 + torch.sum(sinogram, dim=(-1, -2), keepdim=True))
        sinogram = sinogram * cs
        mul_factor = mul_factor * cs
        noise = noise * cs
        # Apply Poisson noise (photon counting statistics)
        y = torch.poisson(sinogram)
        # Remove noise and correct for multiplicative factor
        sino = nn.ReLU()((y - noise) / mul_factor)
        return sino

    def A_degrade_physics_compliant(self, x, total_count=5e6, drf=100):
        """
        ICML Standard Degradation: Image -> Low Dose Sinogram -> Dirty FBP Image
        Args:
            x: Ground Truth Image (Batch, 1, H, W)
            count: Full Dose Count level (typically 5e6 ~ 1e7 for 2D slice)
            drf: Dose Reduction Factor (e.g., 100 for 1% dose)
        Returns:
            obs_image: The noisy, artifact-filled image input for the network (x_0 or condition)
        """
        # Forward-project the clean image.
        clean_proj = self.radon.forward(x)
        # Simulate attenuation and scatter.
        attenuation = torch.exp(-0.1 * clean_proj)
        # Scale to the target count level.
        target_count = total_count / drf
        scale_factor = target_count / (torch.sum(clean_proj) + 1e-8)
        lambda_sino = clean_proj * attenuation * scale_factor
        noisy_sino = torch.poisson(lambda_sino)
        corrected_sino = noisy_sino / (attenuation * scale_factor + 1e-8)
        # Transform the sinogram to the image domain.
        obs_image = self.radon.filter_sinogram(corrected_sino)
        obs_image = self.radon.backward(obs_image)
        return obs_image
    def AT(self, y):
        """
        Back projection (no filter): sinogram -> image

        Args:
            y: Sinogram (B, C, Angle, Detector)
        Returns:
            Image (B, C, H, W)
        """
        recon = self.radon.backward(y) * self.scale_factor
        if self.circular_mask is not None:
            recon = self._apply_mask(recon)
        return recon

    def AT_filtered(self, y):
        """
        Filtered back projection (FBP): sinogram -> image

        Args:
            y: Sinogram (B, C, Angle, Detector)
        Returns:
            Image (B, C, H, W)
        """
        # Apply ramp filter
        y_filtered = self.radon.filter_sinogram(y)
        # Back projection
        recon = self.radon.backward(y_filtered)
        # Apply scaling and mask
        if self.circular_mask is not None:
            recon = self._apply_mask(recon)
        return recon

    def A_split(self, x, p):
        """Split projected counts by binomial thinning with probability ``p``."""
        # Check if in image domain
        assert x.shape[1] == 1, (
            f"imaging_system.A_split() only works in image domain, but received {x.shape[1]}-channel input. "
            f"Latent domain data (typically 4 channels) must be decoded to image domain first. "
            f"Input shape: {x.shape}"
        )
        assert p > 0 and p < 1, "p must be between 0 and 1"
        # Step 1: Forward projection to get expected sinogram
        clean_proj = self.A(x)  # Expected value (B, C, Angle, Detector)
        attenuation = torch.rand_like(clean_proj) * 0.2 + 0.8
        scatter = torch.mean(clean_proj) * 0.1
        lambda_full = clean_proj * attenuation + scatter
        # Step 2: Sample Poisson counts from expected value
        sino_full = torch.poisson(lambda_full)  # Actual full counts (integers)
        sino_full = nn.ReLU()((sino_full))
        # Step 3: Apply Binomial thinning
        total = sino_full.float()
        probs = torch.full_like(total, p)
        binom = torch.distributions.Binomial(total_count=total, probs=probs)
        sino_part1 = binom.sample()
        sino_part2 = sino_full - sino_part1  # Perfect divisibility: part1 + part2 == full
        return sino_part1/p, sino_part2/(1-p)
    def _apply_mask(self, x):
        """Apply circular mask to image"""
        mask = self.circular_mask.unsqueeze(0).unsqueeze(0).expand_as(x)
        result = x.clone()
        result[~mask] = 0
        return result
