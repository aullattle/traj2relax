import torch
import torch.nn as nn
from torch import Tensor
from typing import Union
from torch_geometric.data import Batch
import hydra
import omegaconf
from model.gemnet.gemnet import GemNetTDenoiser
from model.gemnet.data import ChemGraph


class Traj2Relax(nn.Module):
    """
    Traj2Relax: Diffusion-based relaxation model.
    Combines a noiser module and a denoiser (GemNetT).
    """

    def __init__(self, noiser: nn.Module, denoiser_config: omegaconf.DictConfig):
        """
        Initialize Traj2Relax model.

        Args:
            noiser (nn.Module): The noise scheduler module.
            denoiser_config (DictConfig): Configuration for GemNetT denoiser.
        """
        super().__init__()
        self.denoiser: Union[GemNetTDenoiser, nn.Module] = hydra.utils.instantiate(denoiser_config)
        self.noiser = noiser
        self.num_timesteps = self.noiser.timesteps

    def forward(self, a: Tensor, l: Tensor, x: Tensor, n: Tensor, t: Tensor):
        """
        Forward pass of the diffusion model at a given timestep.

        Args:
            a (Tensor): Atomic numbers, shape (total_atoms,).
            l (Tensor): Lattice tensor, shape (batch_size, 3, 3).
            x (Tensor): Atomic positions, shape (total_atoms, 3).
            n (Tensor): Number of atoms per structure, shape (batch_size,).
            t (Tensor): Timestep indices, shape (batch_size,).

        Returns:
            Tuple[Tensor, Tensor]:
                pred_pos_v (Tensor): Predicted position velocities.
                pred_e (Tensor): Predicted energies.
        """
        batch_size = n.size(0)
        timestep = self.noiser.timestep_map[t]

        chem_graphs = []
        pos_start = 0

        # Build batched crystal graphs
        for i in range(batch_size):
            pos_end = pos_start + n[i].item()
            pos_slice = x[pos_start:pos_end]
            atomic_numbers_slice = a[pos_start:pos_end]

            chem_graphs.append(
                ChemGraph(
                    pos=pos_slice,
                    cell=l[i],
                    atomic_numbers=atomic_numbers_slice,
                    num_atoms=n[i],
                    pbc=torch.tensor([1, 1, 1], dtype=torch.bool, device=x.device),
                )
            )
            pos_start = pos_end

        # Create a batched graph object
        chem_graph_batch = Batch.from_data_list(chem_graphs)

        # Run the denoiser model
        pred_pos_v, pred_e = self.denoiser(x=chem_graph_batch, t=timestep)

        return pred_pos_v, pred_e
