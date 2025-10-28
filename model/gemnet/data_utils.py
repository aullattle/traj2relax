import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# import networkx as nx
import torch
from p_tqdm import p_umap
from pymatgen.analysis import local_env
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Element
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch_geometric.data import Data

# import itertools
from tqdm.auto import tqdm

# from materials.explorers.common.data.types import PropertyToConditionOn
from model.gemnet.data import ChemGraph
from model.gemnet.ocp_graph_utils import radius_graph_pbc as radius_graph_pbc_ocp
# from utilities.blob_storage import BlobClient
# from utilities.globals import DEFAULT_AZURE_STORAGE_ACCOUNT, DEFAULT_AZURE_SUBSCRIPTION_NAME
# from utilities.logging import logger

# from networkx.algorithms.components import is_connected

from typing import Dict, Literal, Sequence, Union

# supported properties that can be conditioned on in the generative model
PropertyToConditionOn = Literal[
    "dft_bulk_modulus",
    "dft_shear_modulus",
    "dft_poisson_ratio",
    "dft_band_gap",
    "dft_mag_density",
    "pred_bulk_modulus",
    "pred_shear_modulus",
    "azure_bulk_modulus",
    "azure_band_gap",
    "larsen_score_2d",
    "Si_100_mismatch",
    "formation_energy_per_atom",
    "chemical_system",
    "hhi_score",
    "csi_score_log10",
]

# this is the type of an object used to specify a single target to condition on when sampling
TargetProperty = Dict[PropertyToConditionOn, Union[float, Sequence[str]]]

CRYSTAL_RAW_DATA_CONTAINER = "crystal-raw-data"

# Tensor of unit cells. Assumes 27 cells in -1, 0, 1 offsets in the x and y dimensions
# Note that differing from OCP, we have 27 offsets here because we are in 3D
OFFSET_LIST = [
    [-1, -1, -1],
    [-1, -1, 0],
    [-1, -1, 1],
    [-1, 0, -1],
    [-1, 0, 0],
    [-1, 0, 1],
    [-1, 1, -1],
    [-1, 1, 0],
    [-1, 1, 1],
    [0, -1, -1],
    [0, -1, 0],
    [0, -1, 1],
    [0, 0, -1],
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, -1],
    [0, 1, 0],
    [0, 1, 1],
    [1, -1, -1],
    [1, -1, 0],
    [1, -1, 1],
    [1, 0, -1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, -1],
    [1, 1, 0],
    [1, 1, 1],
]

EPSILON = 1e-5

chemical_symbols = [
    # 0
    "X",
    # 1
    "H",
    "He",
    # 2
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    # 3
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    # 4
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    # 5
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    # 6
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    # 7
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]


CrystalNN = local_env.CrystalNN(distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)


def optionally_get_canonical_structure(
    structure: Structure, niggli: bool, primitive: bool
) -> Structure:
    if primitive:
        structure = structure.get_primitive_structure()
    if niggli:
        structure = structure.get_reduced_structure()

    structure = Structure(
        lattice=Lattice.from_parameters(*structure.lattice.parameters),
        species=structure.species,
        coords=structure.frac_coords,
        coords_are_cartesian=False,
    )

    return structure


def build_crystal(crystal_str: str, niggli: bool = True, primitive: bool = False) -> Structure:
    """Build crystal from cif string."""
    return optionally_get_canonical_structure(
        structure=Structure.from_str(crystal_str, fmt="cif"), niggli=niggli, primitive=primitive
    )


def build_crystal_graph(
    crystal: Structure, graph_method="crystalnn"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """ """

    if graph_method == "crystalnn":
        crystal_graph = StructureGraph.with_local_env_strategy(crystal, CrystalNN)
    elif graph_method == "none":
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers

    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(crystal.lattice.matrix, lattice_params_to_matrix(*lengths, *angles))

    edge_indices, to_jimages = [], []
    if graph_method != "none":
        for i, j, to_jimage in crystal_graph.graph.edges(data="to_jimage"):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms


def abs_cap(val: float, max_abs_val: float = 1) -> float:
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def lattice_params_to_matrix_torch(lengths, angles) -> torch.Tensor:
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    cos_angles = torch.clamp(torch.cos(torch.deg2rad(angles)), -1.0, 1.0)
    return lattice_params_to_matrix_partial_torch(lengths=lengths, cos_angles=cos_angles, eps=0.0)


def compute_volume(batch_lattice: torch.Tensor) -> torch.Tensor:
    """Compute volume from batched lattice matrix

    batch_lattice: (N, 3, 3)
    """
    vector_a, vector_b, vector_c = torch.unbind(batch_lattice, dim=1)
    return torch.abs(torch.einsum("bi,bi->b", vector_a, torch.cross(vector_b, vector_c, dim=1)))


def lengths_angles_to_volume(lengths, angles) -> torch.Tensor:
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    return compute_volume(lattice)


def lattice_matrix_to_params(matrix: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    lengths = np.sqrt(np.sum(matrix**2, axis=1)).tolist()

    angles = np.zeros(3)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[i] = abs_cap(np.dot(matrix[j], matrix[k]) / (lengths[j] * lengths[k]))
    angles = np.arccos(angles) * 180.0 / np.pi
    a, b, c = lengths
    alpha, beta, gamma = angles
    return a, b, c, alpha, beta, gamma


def lattice_matrix_to_params_torch(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a batch of lattice matrices into their corresponding unit cell vector lengths and angles.

    Args:
        matrix (torch.Tensor, [B, 3, 3]): The batch of lattice matrices.

    Returns:
        Tuple[torch.Tensor], ([B, 3], [B, 3]): Tuple whose first element is the lengths of the unit cell vectors, and the second one gives the angles between the vectors.
    """
    lengths, cos_angles = lattice_matrix_to_params_partial_torch(matrix=matrix, eps=0.0)
    return lengths, torch.arccos(cos_angles) * 180.0 / np.pi


def lattice_matrix_to_params_partial_torch(
    matrix: torch.Tensor, eps: float = 1e-3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a batch of lattice matrices into their corresponding unit cell vector lengths and angles.

    Args:
        matrix (torch.Tensor, [B, 3, 3]): The batch of lattice matrices.

    Returns:
        Tuple[torch.Tensor], ([B, 3], [B, 3]): Tuple whose first element is the lengths of the unit cell vectors, and the second one gives the angles between the vectors.
    """
    assert len(matrix.shape) == 3

    # derivatives of arccos(cos(theta)) are undefined for abs(cos(theta))=1
    # we should physically encounter lattices that have vectors that are
    # parallel to one another. NOTE: the value of eps may need tuning
    # if calculations are found to fail, reduce this magnitude

    lengths = matrix.norm(p=2, dim=-1)
    ix_j = torch.tensor([1, 2, 0], dtype=torch.long, device=matrix.device)
    ix_k = torch.tensor([2, 0, 1], dtype=torch.long, device=matrix.device)
    cos_angles = (torch.cosine_similarity(matrix[:, ix_j], matrix[:, ix_k], dim=-1)).clamp(
        -1 + eps, 1 - eps
    )
    if len(matrix.shape) == 2:
        cos_angles = cos_angles.squeeze(0)
        lengths = lengths.squeeze(0)
    return lengths, cos_angles


def lattice_params_to_matrix_partial_torch(lengths, cos_angles, eps: float = 1e-3) -> torch.Tensor:
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    coses = cos_angles
    sins = (1 - cos_angles**2).sqrt()

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    val = torch.clamp(val, -1.0 + eps, 1.0 - eps)

    vector_a = torch.stack(
        [
            lengths[:, 0] * sins[:, 1],
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 0] * coses[:, 1],
        ],
        dim=1,
    )
    vector_b = torch.stack(
        [
            -lengths[:, 1] * sins[:, 0] * val,
            lengths[:, 1] * sins[:, 0] * (1 - val**2).sqrt(),
            lengths[:, 1] * coses[:, 0],
        ],
        dim=1,
    )
    vector_c = torch.stack(
        [
            torch.zeros(lengths.size(0), device=lengths.device),
            torch.zeros(lengths.size(0), device=lengths.device),
            lengths[:, 2],
        ],
        dim=1,
    )

    return torch.stack([vector_a, vector_b, vector_c], dim=1)


def frac_to_cart_coords(
    frac_coords: torch.Tensor, lengths: torch.Tensor, angles: torch.Tensor, num_atoms: torch.Tensor
) -> torch.Tensor:
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    return frac_to_cart_coords_with_lattice(frac_coords, num_atoms, lattice)


def frac_to_cart_coords_with_lattice(
    frac_coords: torch.Tensor, num_atoms: torch.Tensor, lattice: torch.Tensor
) -> torch.Tensor:
    lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
    pos = torch.einsum("bi,bij->bj", frac_coords, lattice_nodes)  # cart coords
    # add warning if frac coords are not [0,1] inclusive: TODO
    return pos


def cart_to_frac_coords(
    cart_coords: torch.Tensor, lengths: torch.Tensor, angles: torch.Tensor, num_atoms: torch.Tensor
) -> torch.Tensor:
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    return cart_to_frac_coords_with_lattice(cart_coords, num_atoms, lattice)


def cart_to_frac_coords_with_lattice(
    cart_coords: torch.Tensor, num_atoms: torch.Tensor, lattice: torch.Tensor
) -> torch.Tensor:
    # use pinv in case the predicted lattice is not rank 3
    inv_lattice = torch.linalg.pinv(lattice)
    inv_lattice_nodes = torch.repeat_interleave(inv_lattice, num_atoms, dim=0)
    frac_coords = torch.einsum("bi,bij->bj", cart_coords, inv_lattice_nodes)
    return frac_coords % 1.0


def get_pbc_distances(
    coords,
    edge_index,
    lengths,
    angles,
    to_jimages,
    num_atoms,
    num_bonds,
    coord_is_cart=False,
    return_offsets=False,
    return_distance_vec=False,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    return get_pbc_distances_with_lattice(
        coords,
        edge_index,
        lattice,
        to_jimages,
        num_atoms,
        num_bonds,
        coord_is_cart,
        return_offsets,
        return_distance_vec,
    )


def get_pbc_distances_with_lattice(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    lattice: torch.Tensor,
    to_jimages: torch.Tensor,
    num_atoms: torch.Tensor,
    num_bonds: torch.Tensor,
    coord_is_cart: bool = False,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
) -> torch.Tensor:
    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = torch.repeat_interleave(lattice, num_atoms, dim=0)
        pos = torch.einsum("bi,bij->bj", coords, lattice_nodes)  # cart coords

    j_index, i_index = edge_index

    distance_vectors = pos[j_index] - pos[i_index]

    # correct for pbc
    lattice_edges = torch.repeat_interleave(lattice, num_bonds, dim=0)
    offsets = torch.einsum("bi,bij->bj", to_jimages.float(), lattice_edges)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors

    if return_offsets:
        out["offsets"] = offsets

    return out


def radius_graph_pbc_wrapper(
    data: Data, radius, max_num_neighbors_threshold, device, max_cell_images_per_dim: int
):
    cart_coords = frac_to_cart_coords(data.frac_coords, data.lengths, data.angles, data.num_atoms)
    return radius_graph_pbc(
        cart_coords=cart_coords,
        lengths=data.lengths,
        angles=data.angles,
        num_atoms=data.num_atoms,
        radius=radius,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        device=device,
        max_cell_images_per_dim=max_cell_images_per_dim,
    )


def radius_graph_pbc_wrapper_with_lattice(
    data: Data, radius, max_num_neighbors_threshold, device, lattice
):
    # pymatgen.core.structure.Structure does not have attribute num_atoms

    cart_coords = frac_to_cart_coords_with_lattice(
        data.frac_coords, data.num_atoms, lattice=lattice
    )
    return radius_graph_pbc_with_lattice(
        cart_coords=cart_coords,
        lattice=lattice,
        num_atoms=data.num_atoms,
        radius=radius,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        device=device,
    )


def radius_graph_pbc(
    cart_coords,
    lengths,
    angles,
    num_atoms,
    radius,
    max_num_neighbors_threshold,
    device,
    max_cell_images_per_dim: int,
    topk_per_pair=None,
):
    lattice = lattice_params_to_matrix_torch(lengths, angles)
    return radius_graph_pbc_with_lattice(
        cart_coords=cart_coords,
        lattice=lattice,
        num_atoms=num_atoms,
        radius=radius,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        device=device,
        topk_per_pair=topk_per_pair,
        max_cell_images_per_dim=max_cell_images_per_dim,
    )


def radius_graph_pbc_with_lattice(
    cart_coords: torch.Tensor,
    lattice: torch.Tensor,
    num_atoms: torch.Tensor,
    radius: float,
    max_num_neighbors_threshold: int,
    device,
    max_cell_images_per_dim: int = 10,
    topk_per_pair: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes pbc graph edges under pbc.

    topk_per_pair: (num_atom_pairs,), select topk edges per atom pair

    Note: topk should take into account self-self edge for (i, i)

        Keyword arguments
        -----------------
        cart_cords.shape=[Ntotal, 3] -- concatenate all atoms over all crystals
        lattice.shape=[Ncrystal, 3, 3]
        num_atoms.shape=[Ncrystal]
        max_cell_images_per_dim -- constrain the max. number of cell images per dimension in event
                                that infinitesimal angles between lattice vectors are encountered.

    WARNING: It is possible (and has been observed) that for rare cases when periodic atom images are
    on or close to the cut off radius boundary, doing these operations in 32 bit floating point can
    lead to atoms being spuriously considered within or outside of the cut off radius. This can lead
    to invariance of the neighbour list under global translation of all atoms in the unit cell. For
    the rare cases where this was observed, switching to 64 bit precision solved the issue. Since all
    graph embeddings should taper messages from neighbours to zero at the cut off radius, the effect
    of these errors in 32-bit should be negligible in practice.
    """
    assert topk_per_pair is None, "non None values of topk_per_pair is not supported"
    edge_index, unit_cell, num_neighbors_image, _, _ = radius_graph_pbc_ocp(
        pos=cart_coords,
        cell=lattice,
        natoms=num_atoms,
        pbc=torch.BoolTensor([True, True, True]),
        radius=radius,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        max_cell_images_per_dim=max_cell_images_per_dim,
    )
    return edge_index, unit_cell, num_neighbors_image


def min_distance_sqr_pbc(
    cart_coords1,
    cart_coords2,
    lengths,
    angles,
    num_atoms,
    device,
    return_vector=False,
    return_to_jimages=False,
):
    """Compute the pbc distance between atoms in cart_coords1 and cart_coords2.
    This function assumes that cart_coords1 and cart_coords2 have the same number of atoms
    in each data point.
    returns:
        basic return:
            min_atom_distance_sqr: (N_atoms, )
        return_vector == True:
            min_atom_distance_vector: vector pointing from cart_coords1 to cart_coords2, (N_atoms, 3)
        return_to_jimages == True:
            to_jimages: (N_atoms, 3), position of cart_coord2 relative to cart_coord1 in pbc
    """
    batch_size = len(num_atoms)

    # Get the positions for each atom
    pos1 = cart_coords1
    pos2 = cart_coords2

    unit_cell = torch.tensor(OFFSET_LIST, device=device).float()
    num_cells = len(unit_cell)
    # unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(cart_coords2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # lattice matrix
    lattice = lattice_params_to_matrix_torch(lengths, angles)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(pbc_offsets, num_atoms, dim=0)

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the vector between atoms
    # shape (num_atom_squared_sum, 3, 27)
    atom_distance_vector = pos1 - pos2
    atom_distance_sqr = torch.sum(atom_distance_vector**2, dim=1)

    min_atom_distance_sqr, min_indices = atom_distance_sqr.min(dim=-1)

    return_list = [min_atom_distance_sqr]

    if return_vector:
        min_indices = min_indices[:, None, None].repeat([1, 3, 1])

        min_atom_distance_vector = torch.gather(atom_distance_vector, 2, min_indices).squeeze(-1)

        return_list.append(min_atom_distance_vector)

    if return_to_jimages:
        to_jimages = unit_cell.T[min_indices].long()
        return_list.append(to_jimages)

    return return_list[0] if len(return_list) == 1 else return_list


class StandardScalerTorch(object):
    """Normalizes the targets of a dataset."""

    def __init__(self, means=None, stds=None):
        self.means = means
        self.stds = stds

    def fit(self, X):
        self.means = torch.nanmean(X, dim=0)
        self.stds = torch_nanstd(X, dim=0, unbiased=False) + EPSILON

    def transform(self, X):
        return (X - self.means) / self.stds

    def inverse_transform(self, X):
        return X * self.stds + self.means

    def match_device(self, tensor):
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)

    def copy(self):
        return StandardScalerTorch(
            means=self.means.clone().detach(), stds=self.stds.clone().detach()
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means.tolist()}, "
            f"stds: {self.stds.tolist()})"
        )


def get_scaler_from_data_list(data_list, key, log10_prop=False) -> StandardScalerTorch:
    targets = torch.tensor(np.array([d[key] for d in data_list]))
    if log10_prop:
        targets = torch.log10(targets)
    scaler = StandardScalerTorch()
    scaler.fit(targets)
    return scaler


def torch_nanstd(x: torch.Tensor, dim: int, unbiased: bool) -> torch.Tensor:
    data_is_present = torch.all(
        torch.reshape(torch.logical_not(torch.isnan(x)), (x.shape[0], -1)),
        dim=1,
    )
    # https://github.com/pytorch/pytorch/issues/29372
    return torch.std(x[data_is_present], dim=dim, unbiased=unbiased)


# @dataclass(frozen=True)
# class CrystDatasetPreprocessor:
#     raw_data_path: Path
#     niggli: bool
#     primitive: bool
#     graph_method: str
#     # prop_list can be None as it is set from CrystDataset.props: Optional[Literal[...]]
#     prop_list: Optional[List[PropertyToConditionOn]]
#     num_workers: int
#     lattice_scale_method: str
#     include_charge: bool = False

#     @property
#     def dataset_path(self) -> Path:
#         return self.raw_data_path.resolve().parent

#     @property
#     def dataset_name(self) -> str:
#         return self.dataset_path.name

#     @property
#     def raw_data_blob_root(self) -> str:
#         return f"csv/{self.dataset_name}"

#     @property
#     def preprocessed_blob_root(self) -> str:
#         return f"csv/{self.dataset_name}/preprocessed"

#     @property
#     def default_cache_path(self) -> Path:
#         return self.dataset_path / "preprocessed"

#     @property
#     def split_name(self) -> str:
#         filename = self.raw_data_path.name
#         if filename.endswith(".csv"):
#             return filename.replace(".csv", "")
#         else:
#             return filename

#     def get_cached_file_name(self) -> str:
#         assert self.prop_list is not None  # for mypy
#         prop = "_".join(self.prop_list)
#         return f"{self.split_name}_niggli_{self.niggli}_primitive_{self.primitive}_graph_method_{self.graph_method}_prop_{prop}_lattice_scale_method_{self.lattice_scale_method}_include_charge_{self.include_charge}.pkl"

#     def get_raw_file_name(self) -> str:
#         return f"{self.split_name}.csv"

#     def preprocess_or_download(self, prefer_cached: bool = True) -> List[Dict[str, Any]]:
#         """Preprocesses or loads already preprocessed dataset.

#         Args:
#             prefer_cached: If True, first tries to load a local copy of preprocessed dataset, if that fails,
#                 tries to download the preprocessed dataset from blob storage. If that fails again or if
#                 `prefer_cached = False`, falls back to preprocessing (default: True).

#         Returns:
#             a list of dictionaries containing preprocessed data.
#         """
#         if prefer_cached:
#             path = self.default_cache_path
#             filename = self.get_cached_file_name()
#             if os.path.exists(path / filename):
#                 loaded_cache = torch.load(path / filename)
#                 logger.info(f"Loaded cached data from '{path / filename}'.")
#                 return loaded_cache

#             with BlobClient(
#                 DEFAULT_AZURE_SUBSCRIPTION_NAME,
#                 DEFAULT_AZURE_STORAGE_ACCOUNT,
#                 CRYSTAL_RAW_DATA_CONTAINER,
#                 self.preprocessed_blob_root,
#             ) as downloader:
#                 if downloader.blob_exists(filename):
#                     local_file_path = downloader.download_file(filename)
#                     return torch.load(local_file_path)
#             logger.info(
#                 f"No cached data found at '{path / filename}'. Proceeding with preprocessing."
#             )
#         # If we get here, either prefer_cached = False or we failed to load the cached data.
#         # In both cases, we need to preprocess.
#         if self.raw_data_path.suffix == ".csv":
#             return self.preprocess()
#         raise ValueError(f"Don't know how to preprocess {self.raw_data_path}.")

#     @classmethod
#     def process_one_structure(
#         cls,
#         structure: Structure,
#         niggli: bool,
#         primitive: bool,
#         graph_method,
#         properties: Optional[Dict[PropertyToConditionOn, Union[str, float]]],
#         include_charge: bool,
#         mp_id: Optional[str],
#         cif_str: Optional[str],
#     ) -> Dict[str, Any]:
#         crystal = optionally_get_canonical_structure(
#             structure=structure, niggli=niggli, primitive=primitive
#         )
#         graph_arrays = build_crystal_graph(crystal, graph_method)
#         result_dict: Dict[Union[str, PropertyToConditionOn], Any] = {"graph_arrays": graph_arrays}
#         if mp_id is not None:
#             result_dict.update({"mp_id": mp_id})
#         if cif_str is not None:
#             result_dict.update({"cif": cif_str})
#         if properties is not None:
#             result_dict.update(properties)  # type: ignore
#         if include_charge:
#             charges_dict = get_charges(crystal)
#             charges = charges_dict["charges"]
#             charge_method = charges_dict["charge_method"]
#             assert len(charges) == len(result_dict["graph_arrays"][1])
#             charges = np.array(charges)
#             result_dict["charges"] = charges
#             result_dict["charge_method"] = charge_method
#         return result_dict

#     def preprocess(self) -> List[Dict[str, Any]]:
#         try:
#             df = pd.read_csv(self.raw_data_path)
#         except FileNotFoundError:
#             logger.info(
#                 f"Raw data file '{self.raw_data_path}' not found locally. Downloading from blob storage."
#             )
#             # Download raw data from blob
#             filename = self.get_raw_file_name()
#             with BlobClient(
#                 DEFAULT_AZURE_SUBSCRIPTION_NAME,
#                 DEFAULT_AZURE_STORAGE_ACCOUNT,
#                 CRYSTAL_RAW_DATA_CONTAINER,
#                 self.raw_data_blob_root,
#             ) as downloader:
#                 df = pd.read_csv(downloader.download_file(filename))

#         def process_one(row, niggli, primitive, graph_method, prop_list, include_charge):
#             cif_str = row["cif"]
#             properties = {k: row[k] for k in prop_list if k in row.keys()}
#             return CrystDatasetPreprocessor.process_one_structure(
#                 structure=CifParser.from_string(cif_str).get_structures()[0],
#                 niggli=niggli,
#                 primitive=primitive,
#                 graph_method=graph_method,
#                 properties=properties,
#                 include_charge=include_charge,
#                 mp_id=row["material_id"],
#                 cif_str=cif_str,
#             )

#         unordered_results = p_umap(
#             process_one,
#             [df.iloc[idx] for idx in range(len(df))],
#             [self.niggli] * len(df),
#             [self.primitive] * len(df),
#             [self.graph_method] * len(df),
#             [self.prop_list] * len(df),
#             [self.include_charge] * len(df),
#             num_cpus=self.num_workers,
#         )
#         mpid_to_results = {result["mp_id"]: result for result in unordered_results}
#         ordered_results = [mpid_to_results[df.iloc[idx]["material_id"]] for idx in range(len(df))]
#         add_scaled_lattice_prop(ordered_results, self.lattice_scale_method)
#         return ordered_results

#     def upload_preprocessed_dataset(self, local_file_path: str, overwrite: bool = False) -> None:
#         BlobClient(
#             DEFAULT_AZURE_SUBSCRIPTION_NAME,
#             DEFAULT_AZURE_STORAGE_ACCOUNT,
#             CRYSTAL_RAW_DATA_CONTAINER,
#             self.preprocessed_blob_root,
#             local_dir=os.path.dirname(local_file_path),
#         ).upload_file(local_file_path, overwrite=overwrite)


def preprocess_tensors(
    crystal_array_list,
    niggli,
    primitive,
    graph_method,
    include_charge: bool = False,
):
    def process_one(batch_idx, crystal_array, niggli, primitive, graph_method, include_charge):
        frac_coords = crystal_array["frac_coords"]
        atom_types = crystal_array["atom_types"]
        lengths = crystal_array["lengths"]
        angles = crystal_array["angles"]
        crystal = Structure(
            lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False,
        )
        graph_arrays = build_crystal_graph(crystal, graph_method)
        result_dict = {
            "batch_idx": batch_idx,
            "graph_arrays": graph_arrays,
        }
        if include_charge:
            charges_dict = get_charges(crystal)
            charges = charges_dict["charges"]
            charge_method = charges_dict["charge_method"]
            assert len(charges) == len(result_dict["graph_arrays"][1])
            charges = np.array(charges)
            result_dict["charges"] = charges
            result_dict["charge_method"] = charge_method
        return result_dict

    unordered_results = p_umap(
        process_one,
        list(range(len(crystal_array_list))),
        crystal_array_list,
        [niggli] * len(crystal_array_list),
        [primitive] * len(crystal_array_list),
        [graph_method] * len(crystal_array_list),
        [include_charge] * len(crystal_array_list),
        num_cpus=30,
    )
    ordered_results = list(sorted(unordered_results, key=lambda x: x["batch_idx"]))
    return ordered_results


def add_scaled_lattice_prop(data_list, lattice_scale_method):
    for dict in data_list:
        graph_arrays = dict["graph_arrays"]
        # the indexes are brittle if more objects are returned
        lengths = graph_arrays[2]
        angles = graph_arrays[3]
        num_atoms = graph_arrays[-1]
        assert lengths.shape[0] == angles.shape[0] == 3
        assert isinstance(num_atoms, int)

        if lattice_scale_method == "scale_length":
            lengths = lengths / float(num_atoms) ** (1 / 3)

        dict["scaled_lattice"] = np.concatenate([lengths, angles])


def mard(targets, preds):
    """Mean absolute relative difference."""
    assert torch.all(targets > 0.0)
    return torch.mean(torch.abs(targets - preds) / targets)


def batch_accuracy_precision_recall(pred_edge_probs, edge_overlap_mask, num_bonds):
    if pred_edge_probs is None and edge_overlap_mask is None and num_bonds is None:
        return 0.0, 0.0, 0.0
    pred_edges = pred_edge_probs.max(dim=1)[1].float()
    target_edges = edge_overlap_mask.float()

    start_idx = 0
    accuracies, precisions, recalls = [], [], []
    for num_bond in num_bonds.tolist():
        pred_edge = pred_edges.narrow(0, start_idx, num_bond).detach().cpu().numpy()
        target_edge = target_edges.narrow(0, start_idx, num_bond).detach().cpu().numpy()

        accuracies.append(accuracy_score(target_edge, pred_edge))
        precisions.append(precision_score(target_edge, pred_edge, average="binary"))
        recalls.append(recall_score(target_edge, pred_edge, average="binary"))

        start_idx = start_idx + num_bond

    return np.mean(accuracies), np.mean(precisions), np.mean(recalls)


class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.
        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.
        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan
        )

        return transformed_with_none

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan
        )

        return transformed_with_none


def make_hash(d: dict):
    """
    Generate a hash for the input dictionary.
    From: https://stackoverflow.com/a/22003440
    Parameters
    ----------
    d: input dictionary
    Returns
    -------
    hash (hex encoded) of the input dictionary.
    """
    import hashlib
    import json

    return hashlib.md5(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


def filter_dataset(
    dataset: Iterable[Data], chemical_system: List[str], strict: bool = False
) -> List[Data]:
    """
    Takes an iterable of datapoints and filters them using the input chemical system.
    That is, this returns a list containing the datapoints that are from the input chemical system.

    Arguments:
        dataset: Iterable[Data]
            The input datapoints.

        chemical_system: List[str]
            The chemical system, e.g., ['Na', 'Cl'].

        strict: bool
            If True, only returns datapoints where the chemical system exactly matches the input one.
            If False, we also return datapoints from a subset of the chemical system. E.g., if the input
            chemical system is ['Li', 'P', 'S'], we also return samples from ['Li'], ['Li', 'P'], and so on.

    Returns:
        filtered_data: List[Data]
            The filtered datapoints.
    """
    chemical_system_numbers = torch.tensor(
        sorted([chemical_symbols.index(x) for x in chemical_system])
    )
    chemical_system_set = set(chemical_system_numbers.tolist())
    filtered_data = []
    for ix, data in tqdm(enumerate(dataset)):
        atom_types_set = set(data.atom_types.tolist())
        check = (
            atom_types_set == chemical_system_set
            if strict
            else atom_types_set.difference(chemical_system_set) == set()
        )
        if check:
            filtered_data.append(data)
    return filtered_data


def get_niggli_lattice(lattice: torch.Tensor, volume_cutoff: float = 1e-3) -> torch.Tensor:
    # Returns tensor of shape [3, 3]
    if lattice.det().abs() < volume_cutoff:
        # sometimes niggli takes ages when the volume is very small.
        warnings.warn(
            f"Skipped niggli computation of a lattice because its volume was very small ({lattice.det().abs():.2f})"
        )
        return lattice
    return torch.from_numpy(
        Lattice(lattice.cpu().numpy()).get_niggli_reduced_lattice()._matrix.copy()
    )


def niggli_reduce_lattice_batch(
    x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Performs a Niggli reduction on the input batch's lattice matrix and updates the fractional coordinates accordingly.

    Args:
        x (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): Input batch consisting of
            * Fractional coordinates: (num_atoms, 3)
            * Lattice matrix: (num_crystals, 3, 3)
            * Atom types: (num_atoms, )
            * Num atoms: (num_crystals,)
            * Batch pointer: (num_atoms,)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The updated Niggli-reduced batch with same shapes as input.
    """
    lattices = x[1]
    frac_coords = x[0]
    num_atoms = x[3]
    cart_coords = frac_to_cart_coords_with_lattice(
        frac_coords=frac_coords, num_atoms=num_atoms, lattice=lattices
    )
    # A simple `map()` was much faster than multiprocessing in my experiments, so we're not parallelizing here.
    reduced_lattices = (
        torch.stack(list(map(get_niggli_lattice, lattices.cpu())), dim=0)
        .to(lattices.device)
        .to(lattices.dtype)
    )
    new_frac_coords = cart_to_frac_coords_with_lattice(
        cart_coords=cart_coords, num_atoms=num_atoms, lattice=reduced_lattices
    )
    return new_frac_coords, reduced_lattices, *x[2:]


def compute_lattice_polar_decomposition(lattice_matrix: torch.Tensor) -> torch.Tensor:
    # Polar decomposition via SVD, see https://en.wikipedia.org/wiki/Polar_decomposition
    # lattice_matrix: [batch_size, 3, 3]
    # Computes the (unique) symmetric lattice matrix that is equivalent (up to rotation) to the input lattice.

    if lattice_matrix.device.type == "cuda":
        # because of this https://statics.teams.cdn.office.net/evergreen-assets/safelinks/1/atp-safelinks.html issue
        # there is an issue running torch.linalg.svd on cuda tensors with driver version 450.*

        try:
            W, S, V_transp = torch.linalg.svd(lattice_matrix)
        except torch._C._LinAlgError:
            # move to cpu and try again
            W, S, V_transp = torch.linalg.svd(lattice_matrix.to("cpu"))
            W = W.to(lattice_matrix.device.type)
            S = S.to(lattice_matrix.device.type)
            V_transp = V_transp.to(lattice_matrix.device.type)
    else:
        W, S, V_transp = torch.linalg.svd(lattice_matrix)
    S_square = torch.diag_embed(S)
    V = V_transp.transpose(1, 2)
    U = W @ V_transp
    P = V @ S_square @ V_transp
    P_prime = U @ P @ U.transpose(1, 2)
    # symmetrized lattice matrix
    symm_lattice_matrix = P_prime
    return symm_lattice_matrix


def get_charges(crystal: Structure) -> Dict[str, Any]:
    """Returns an estimaton of the partial charges for each individual atom in the form of a list.

    Args:
        structure (Structure): Structure describing the crystal we want to assign charges to.

    Returns:
        Dict[str, Any]: A dictionary containing the partial charges and the charge attribution method used
    """

    # Try to assign partial charges with the bond valence analyzer from pymatgen (it takes structure into account)
    try:
        bv_analyzer = BVAnalyzer()
        proposed_structure = bv_analyzer.get_oxi_state_decorated_structure(crystal)
        return {
            "charges": [site.specie._oxi_state for site in proposed_structure._sites],
            "charge_method": "bv_analyzer",
        }
    # The bv analyzer will throw an exception if it fails to assign charges. In this case, try the oxi state guess method
    except ValueError:
        guesses = crystal.composition.oxi_state_guesses()
        # If the method had a guess, pick the best one
        if guesses:
            proposed_composition = crystal.composition.add_charges_from_oxi_state_guesses()
            return {
                "charges": [
                    elem.oxi_state
                    for elem in proposed_composition.elements
                    for _ in range(proposed_composition[elem])
                ],  # Repeat charges according to composition
                "charge_method": "oxi_state_guess",
            }
        # No guess often corresponds to metallic crystals (92% for mp_20).
        # We choose to handle this exception by assigning zero charge to each atom.
        else:
            return {
                "charges": [0] * len(crystal.atomic_numbers),
                "charge_method": "zero_everywhere",
            }


def create_chem_graph_from_composition(target_composition_dict: dict[str, float]) -> ChemGraph:
    atomic_numbers = []
    for element_name, number_of_atoms in target_composition_dict.items():
        atomic_numbers += [Element(element_name).Z] * int(number_of_atoms)

    return ChemGraph(
        atomic_numbers=torch.tensor(atomic_numbers, dtype=torch.long),
        num_atoms=torch.tensor([len(atomic_numbers)], dtype=torch.long),
        cell=torch.eye(3, dtype=torch.float).reshape(1, 3, 3),
        pos=torch.zeros((len(atomic_numbers), 3), dtype=torch.float),
    )
