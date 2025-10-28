from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union
from model.gemnet.utils import ragged_range,repeat_blocks,mask_neighbors,inner_product_normalized
import torch
import torch.nn as nn
import math
from model.gemnet.base_layers import Dense
from model.gemnet.scaling import AutomaticFit
from model.gemnet.radial_basis import RadialBasis,CircularBasisLayer
from model.gemnet.efficient import EfficientInteractionDownProjection
from model.gemnet.interaction_block import InteractionBlockTripletsOnly
from model.gemnet.atom_update_block import OutputBlock
from model.gemnet.data import ChemGraph
from model.gemnet.embedding_block import (
    AtomEmbedding,
    ChargeEmbedding,
    EdgeEmbedding,
)
from model.gemnet.data_utils import (
    get_pbc_distances_with_lattice,
    lattice_matrix_to_params_partial_torch,
    radius_graph_pbc_with_lattice,
)
from torch_sparse import SparseTensor
from torch_scatter import scatter, scatter_add

def edge_score_to_lattice_score_frac(
    score_d: torch.Tensor,
    edge_index: torch.Tensor,
    edge_vectors: torch.Tensor,
    lattice_matrix: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    batch_edge = batch[edge_index[0]]
    unit_edge_vectors_cart = edge_vectors / edge_vectors.norm(dim=-1, keepdim=True)
    edge_vectors_frac = (
        lattice_matrix.inverse().transpose(-1, -2)[batch_edge] @ edge_vectors[:, :, None]
    ).squeeze(-1)
    score_lattice = scatter_add(
        score_d[:, None, None]
        * (unit_edge_vectors_cart[:, :, None] @ edge_vectors_frac[:, None, :]),
        batch_edge,
        dim=0,
        dim_size=batch.max() + 1,
    ).transpose(-1, -2)
    return score_lattice

def edge_score_to_lattice_score_frac_symmetric(
    score_d: torch.Tensor,
    edge_index: torch.Tensor,
    edge_vectors: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    batch_edge = batch[edge_index[0]]
    unit_edge_vectors_cart = edge_vectors / edge_vectors.norm(dim=-1, keepdim=True)
    score_lattice = scatter_add(
        score_d[:, None, None]
        * (unit_edge_vectors_cart[:, :, None] @ unit_edge_vectors_cart[:, None, :]),
        batch_edge,
        dim=0,
        dim_size=batch.max() + 1,
    ).transpose(-1, -2)
    return score_lattice

def edge_score_to_coord_and_lattice_score_with_dists(
    score_d: torch.Tensor,
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    edge_length: torch.Tensor,
    edge_vectors: torch.Tensor,
    to_jimages: torch.Tensor,
    batch: torch.Tensor,
    coord_score: bool = True,
    lattice_score: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    N = coords.size(0)
    batch_edge = batch[edge_index[0]]
    dd_dx = (1 / edge_length[:, None]) * edge_vectors
    score_pos = None
    score_lattice = None
    if coord_score:
        score_pos = -scatter_add(
            dd_dx * score_d[:, None], edge_index[1], dim=0, dim_size=N
        ) + scatter_add(dd_dx * score_d[:, None], edge_index[0], dim=0, dim_size=N)
    if lattice_score:
        dd_dl_per_edge = dd_dx[:, None] * to_jimages[:, :, None]
        score_lattice = scatter_add(
            score_d[:, None, None] * dd_dl_per_edge, batch_edge, dim=0, dim_size=batch.max() + 1
        )
    return score_pos, score_lattice

class GemNetT(torch.nn.Module):
    def __init__(
        self,
        num_targets: int,
        latent_dim: int,
        num_spherical: int = 7,
        num_radial: int = 128,
        num_blocks: int = 3,
        emb_size_atom: int = 512,
        emb_dim_atomic_number: int = 512,
        emb_size_edge: int = 512,
        emb_size_trip: int = 64,
        emb_size_rbf: int = 16,
        emb_size_cbf: int = 16,
        emb_size_bil_trip: int = 64,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_concat: int = 1,
        num_atom: int = 3,
        cutoff: float = 6.0,
        max_neighbors: int = 50,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        cbf: dict = {"name": "spherical_harmonics"},
        otf_graph: bool = False,
        output_init: str = "HeOrthogonal",
        activation: str = "swish",
        with_mask_type: bool = False,
        max_cell_images_per_dim: int = 5,
        conservative_stress_is_invariant: bool = False,
        concat_angles_to_edges: bool = True,
        concat_lengths_to_edges: bool = True,
        include_charge: bool = False,
    ):
        super().__init__()
        self.num_targets = num_targets
        assert num_blocks > 0
        self.num_blocks = num_blocks
        self.emb_size_atom = emb_size_atom
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.max_cell_images_per_dim = max_cell_images_per_dim
        self.otf_graph = otf_graph
        self.conservative_stress_is_invariant = conservative_stress_is_invariant
        self.concat_angles_to_edges = concat_angles_to_edges
        self.concat_lengths_to_edges = concat_lengths_to_edges
        if self.concat_angles_to_edges:
            self.angle_update_mlp = nn.Sequential(
                nn.Linear(emb_size_edge, emb_size_edge),
                nn.ReLU(),
                nn.Linear(emb_size_edge, 3),
            )
            self.angle_edge_emb = nn.Sequential(
                nn.Linear(emb_size_edge + 3, emb_size_edge),
                nn.ReLU(),
                nn.Linear(emb_size_edge, emb_size_edge),
            )
        if self.concat_lengths_to_edges:
            self.length_edge_emb = nn.Sequential(
                nn.Linear(emb_size_edge + 3, emb_size_edge),
                nn.ReLU(),
                nn.Linear(emb_size_edge, emb_size_edge),
            )
        AutomaticFit.reset()
        self.radial_basis = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )
        radial_basis_cbf3 = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )
        self.cbf_basis3 = CircularBasisLayer(
            num_spherical,
            radial_basis=radial_basis_cbf3,
            cbf=cbf,
            efficient=True,
        )
        self.mlp_rbf3 = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_cbf3 = EfficientInteractionDownProjection(num_spherical, num_radial, emb_size_cbf)
        self.mlp_rbf_h = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.mlp_rbf_out = Dense(
            num_radial,
            emb_size_rbf,
            activation=None,
            bias=False,
        )
        self.atom_emb = AtomEmbedding(emb_dim_atomic_number, with_mask_type=with_mask_type)
        self.charge_emb = (
            ChargeEmbedding(emb_dim_atomic_number, with_mask_type=with_mask_type)
            if include_charge
            else None
        )
        self.atom_latent_emb = nn.Linear(
            emb_dim_atomic_number * (1 + int(include_charge)) + latent_dim, emb_size_atom
        )
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )
        out_blocks = []
        int_blocks = []
        interaction_block = InteractionBlockTripletsOnly
        for i in range(num_blocks):
            int_blocks.append(
                interaction_block(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_trip=emb_size_trip,
                    emb_size_rbf=emb_size_rbf,
                    emb_size_cbf=emb_size_cbf,
                    emb_size_bil_trip=emb_size_bil_trip,
                    num_before_skip=num_before_skip,
                    num_after_skip=num_after_skip,
                    num_concat=num_concat,
                    num_atom=num_atom,
                    activation=activation,
                    name=f"IntBlock_{i+1}",
                )
            )
        for i in range(num_blocks + 1):
            out_blocks.append(
                OutputBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    emb_size_rbf=emb_size_rbf,
                    nHidden=num_atom,
                    num_targets=num_targets,
                    activation=activation,
                    output_init=output_init,
                    direct_forces=True,
                    name=f"OutBlock_{i}",
                )
            )
        self.out_blocks = torch.nn.ModuleList(out_blocks)
        self.int_blocks = torch.nn.ModuleList(int_blocks)
        self.shared_parameters = [
            (self.mlp_rbf3, self.num_blocks),
            (self.mlp_cbf3, self.num_blocks),
            (self.mlp_rbf_h, self.num_blocks),
            (self.mlp_rbf_out, self.num_blocks + 1),
        ]

    def get_triplets(
        self, edge_index: torch.Tensor, num_atoms: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx_s, idx_t = edge_index
        value = torch.arange(idx_s.size(0), device=idx_s.device, dtype=idx_s.dtype)
        adj = SparseTensor(
            row=idx_t,
            col=idx_s,
            value=value,
            sparse_sizes=(num_atoms, num_atoms),
        )
        row, col, val = adj.coo()
        mask = torch.isin(row, idx_t)
        adj_edges = SparseTensor(
            row=row[mask],
            col=col[mask],
            value=val[mask] if val is not None else None,
            sparse_sizes=adj.sparse_sizes(),
            is_sorted=True,
            trust_data=True
        )
        id3_ba = adj_edges.storage.value()
        id3_ca = adj_edges.storage.row()
        mask = id3_ba != id3_ca
        id3_ba = id3_ba[mask]
        id3_ca = id3_ca[mask]
        num_triplets = torch.bincount(id3_ca, minlength=idx_s.size(0))
        id3_ragged_idx = ragged_range(num_triplets)
        return id3_ba, id3_ca, id3_ragged_idx

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        tensor_directed = tensor[mask]
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(
        self,
        edge_index: torch.Tensor,
        cell_offsets: torch.Tensor,
        neighbors: torch.Tensor,
        edge_dist: torch.Tensor,
        edge_vector: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_sep_atoms = edge_index[0] < edge_index[1]
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] == 0) & (cell_offsets[:, 2] < 0))
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(batch_edge, minlength=neighbors.size(0))
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(cell_offsets, mask, edge_reorder_idx, True)
        edge_dist_new = self.select_symmetric_edges(edge_dist, mask, edge_reorder_idx, False)
        edge_vector_new = self.select_symmetric_edges(edge_vector, mask, edge_reorder_idx, True)
        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_dist_new,
            edge_vector_new,
        )

    def select_edges(
        self,
        edge_index: torch.Tensor,
        cell_offsets: torch.Tensor,
        neighbors: torch.Tensor,
        edge_dist: torch.Tensor,
        edge_vector: torch.Tensor,
        cutoff: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if cutoff is not None:
            edge_mask = edge_dist <= cutoff
            edge_index = edge_index[:, edge_mask]
            cell_offsets = cell_offsets[edge_mask]
            neighbors = mask_neighbors(neighbors, edge_mask)
            edge_dist = edge_dist[edge_mask]
            edge_vector = edge_vector[edge_mask]
        empty_image = neighbors == 0
        if torch.any(empty_image):
            import pdb
            pdb.set_trace()
        return edge_index, cell_offsets, neighbors, edge_dist, edge_vector

    def generate_interaction_graph(
        self,
        cart_coords: torch.Tensor,
        lattice: torch.Tensor,
        num_atoms: torch.Tensor,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        edge_index, to_jimages, num_bonds = radius_graph_pbc_with_lattice(
            cart_coords=cart_coords,
            lattice=lattice,
            num_atoms=num_atoms,
            radius=self.cutoff,
            max_num_neighbors_threshold=self.max_neighbors,
            device=num_atoms.device,
            max_cell_images_per_dim=self.max_cell_images_per_dim,
        )
        out = get_pbc_distances_with_lattice(
            cart_coords,
            edge_index,
            lattice,
            to_jimages,
            num_atoms,
            num_bonds,
            coord_is_cart=True,
            return_offsets=True,
            return_distance_vec=True,
        )
        edge_index = out["edge_index"]
        D_st = out["distances"]
        V_st = -out["distance_vec"] / D_st[:, None]
        (
            edge_index,
            cell_offsets,
            neighbors,
            D_st,
            V_st,
        ) = self.reorder_symmetric_edges(edge_index, to_jimages, num_bonds, D_st, V_st)
        block_sizes = neighbors // 2
        block_sizes = torch.masked_select(block_sizes, block_sizes > 0)
        id_swap = repeat_blocks(
            block_sizes,
            repeats=2,
            continuous_indexing=False,
            start_idx=block_sizes[0],
            block_inc=block_sizes[:-1] + block_sizes[1:],
            repeat_inc=-block_sizes,
        )
        id3_ba, id3_ca, id3_ragged_idx = self.get_triplets(
            edge_index,
            num_atoms=num_atoms.sum(),
        )
        return (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
            cell_offsets,
        )

    def forward(
        self,
        z: torch.Tensor,
        lattice: torch.Tensor,
        cart_coords: torch.Tensor,
        atom_types: torch.Tensor,
        num_atoms: torch.Tensor,
        batch: torch.Tensor
    ) -> torch.Tensor:
        distorted_lattice = lattice
        pos = cart_coords
        atomic_numbers = atom_types
        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
            to_jimages,
        ) = self.generate_interaction_graph(
            pos, distorted_lattice, num_atoms
        )
        idx_s, idx_t = edge_index
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        rad_cbf3, cbf3 = self.cbf_basis3(D_st, cosφ_cab, id3_ca)
        rbf = self.radial_basis(D_st)
        h = self.atom_emb(atomic_numbers)
        if z is not None:
            z_per_atom = z[batch]
            h = torch.cat([h, z_per_atom], dim=1)
        h = self.atom_latent_emb(h)
        m = self.edge_emb(h, rbf, idx_s, idx_t)
        if self.concat_angles_to_edges:
            batch_edge = batch[edge_index[0]]
            cosines = torch.cosine_similarity(V_st[:, None], distorted_lattice[batch_edge], dim=-1)
            m = torch.cat([m, cosines], dim=-1)
            m = self.angle_edge_emb(m)
        if self.concat_lengths_to_edges:
            lattice_lengths, _ = lattice_matrix_to_params_partial_torch(distorted_lattice)
            relative_lengths = D_st[:, None] / lattice_lengths[batch_edge]
            m = torch.cat([m, relative_lengths], dim=-1)
            m = self.length_edge_emb(m)
        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)
        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)
        E_t, F_st = self.out_blocks[0](h, m, rbf_out, idx_t)
        for i in range(self.num_blocks):
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf3=rbf3,
                cbf3=cbf3,
                id3_ragged_idx=id3_ragged_idx,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
            )
            E, F = self.out_blocks[i + 1](h, m, rbf_out, idx_t)
            F_st += F
            E_t += E
        nMolecules = torch.max(batch) + 1
        E_t = scatter(
            E_t, batch, dim=0, dim_size=nMolecules, reduce="sum"
        )
        F_st_vec = F_st[:, :, None] * V_st[:, None, :]
        F_t = scatter(
            F_st_vec,
            idx_t,
            dim=0,
            dim_size=num_atoms.sum(),
            reduce="add",
        )
        F_t = F_t.squeeze(1)
        return F_t, E_t

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

class NoiseLevelEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        self.register_buffer("div_term", div_term)
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = torch.zeros((t.shape[0], self.d_model), device=self.div_term.device)
        x[:, 0::2] = torch.sin(t[:, None] * self.div_term[None])
        x[:, 1::2] = torch.cos(t[:, None] * self.div_term[None])
        return self.dropout(x)

class GemNetTDenoiser(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        max_neighbors: int = 50,
        radius: float = 7.0,
        num_targets: int = 1,
        otf_graph: bool = True,
        num_blocks: int = 4,
        max_cell_images_per_dim: int = 5,
        concat_angles_to_edges: bool = True,
        concat_lengths_to_edges: bool = True,
        include_charge: bool = False,
        gnn_class: nn.Module = GemNetT,
    ):
        super(GemNetTDenoiser, self).__init__()
        self.noise_level_encoding = NoiseLevelEncoding(hidden_dim)
        self.gemnet = gnn_class(
            num_targets=num_targets,
            latent_dim=hidden_dim,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            emb_dim_atomic_number=hidden_dim,
            cutoff=radius,
            max_neighbors=max_neighbors,
            max_cell_images_per_dim=max_cell_images_per_dim,
            otf_graph=otf_graph,
            num_blocks=num_blocks,
            concat_angles_to_edges=concat_angles_to_edges,
            concat_lengths_to_edges=concat_lengths_to_edges,
            include_charge=include_charge,
        )

    def forward(
        self,
        x: ChemGraph,
        t: torch.Tensor
    ) -> ChemGraph:
        x_tuple = (
            x["pos"],
            x["cell"],
            x["atomic_numbers"],
            x["num_atoms"],
            x.get_batch_idx("pos"),
        )
        assert isinstance(x_tuple, tuple) and len(x_tuple) == 5
        for tensor in x_tuple:
            assert isinstance(tensor, torch.Tensor)
        t_enc = self.noise_level_encoding(t).to(x["cell"].device)
        pred_eps, pred_e = self.gemnet(
            z=t_enc,
            cart_coords=x_tuple[0],
            atom_types=x_tuple[2],
            num_atoms=x_tuple[3],
            batch=x_tuple[4],
            lattice=x_tuple[1],
        )
        pred_pos_v = x["cell"].inverse().transpose(1, 2)[x.get_batch_idx("pos")] @ pred_eps.unsqueeze(-1)
        return pred_pos_v, pred_e