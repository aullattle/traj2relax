import torch
from torch import Tensor, nn
import math
from typing import Tuple


class Noiser(nn.Module):
    """
    Noise scheduler module for diffusion-style trajectory modeling.
    Implements a cosine-annealed interpolation between distributions.
    """

    def __init__(self, timesteps: int, gamma: float, alpha: float):
        """
        Initialize the noise scheduler.

        Args:
            timesteps (int): Number of training timesteps.
            gamma (float): Maximum noise scaling factor.
            alpha (float): Global noise strength.
        """
        super().__init__()

        self.timesteps = timesteps
        self.delta = 1 / (self.timesteps - 1)

        # Register timestep indices
        self.register_buffer("timestep_map", torch.arange(0, timesteps, 1, dtype=torch.long))

        # Generate cosine annealing noise schedule
        t = torch.linspace(0, 1, timesteps)
        noise_scales = 0.5 * alpha * gamma * (1 + torch.cos(math.pi * t))
        self.register_buffer("noise_scales", noise_scales)

    @torch.no_grad()
    def forward(
        self,
        l_mid: Tensor,
        x_mid: Tensor,
        x_mid_prev: Tensor,
        e_mid: Tensor,
        t: Tensor,
        num_atoms: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Compute the perturbed state and corresponding target for each timestep.

        Args:
            l_mid (Tensor): Intermediate lattice tensor, shape (B, 3, 3).
            x_mid (Tensor): Intermediate atomic positions, shape (ΣN, 3).
            x_mid_prev (Tensor): Previous step atomic positions, shape (ΣN, 3).
            e_mid (Tensor): Intermediate energy tensor, shape (B,).
            t (Tensor): Timestep indices for each sample, shape (B,).
            num_atoms (Tensor): Atom counts for each structure, shape (B,).

        Returns:
            Tuple of tensors:
                x_t (Tensor): Noised atomic positions.
                x_target (Tensor): Position velocity target.
                l_t (Tensor): Current lattice tensor.
                e_target (Tensor): Energy targets.
        """
        batch_size = t.size(0)
        device = l_mid.device

        x_t_list, x_target_list, l_t_list, e_target_list = [], [], [], []

        # Generate Gaussian noise
        noise_x = torch.randn_like(x_mid, device=device)

        # Split per structure
        x_mid_split = torch.split(x_mid, num_atoms.tolist())
        x_mid_prev_split = torch.split(x_mid_prev, num_atoms.tolist())
        noise_x_split = torch.split(noise_x, num_atoms.tolist())

        # Process each structure independently
        for i in range(batch_size):
            x_mid_sample = x_mid_split[i]
            x_mid_prev_sample = x_mid_prev_split[i]
            l_mid_sample = l_mid[i]
            noise_x_sample = noise_x_split[i]

            # Compute noised positions and velocity targets
            x_t_sample = x_mid_sample + self.noise_scales[t[i]] * noise_x_sample
            x_target_sample = (x_mid_prev_sample - x_t_sample) / self.delta

            # Energy target (no noise added)
            e_target_sample = e_mid[i]

            x_t_list.append(x_t_sample)
            x_target_list.append(x_target_sample)
            l_t_list.append(l_mid_sample.unsqueeze(0))
            e_target_list.append(e_target_sample)

        x_t = torch.cat(x_t_list, dim=0).float()
        x_target = torch.cat(x_target_list, dim=0).float()
        l_t = torch.cat(l_t_list, dim=0).float()
        e_target = torch.tensor(e_target_list, device=device).float()

        return x_t, x_target, l_t, e_target
