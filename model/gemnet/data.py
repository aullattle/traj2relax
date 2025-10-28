import copy
from typing import Optional, Union, List, Tuple
from torch import LongTensor, IntTensor, Tensor
import torch_geometric.data as pyg_data
from torch_geometric.typing import OptTensor
from torch_geometric import utils
import torch
from model.gemnet.unit import NormalizationStats
import pandas as pd
from typing import Union
import torch

class ChemGraph(pyg_data.Data):
    r"""A ChemGraph is a Pytorch Geometric Data object describing a MLPotential molecular graph with atoms in 3D space.
    The data object can hold node-level, and graph-level attributes, as well as (pre-computed) edge information.
    In general, :class:`~torch_geometric.data.Data` tries to mimic the
    behaviour of a regular Python dictionary.
    In addition, it provides useful functionality for analyzing graph
    structures, and provides basic PyTorch tensor functionalities.
    See `here <https://pytorch-geometric.readthedocs.io/en/latest/notes/
    introduction.html#data-handling-of-graphs>`__ for the accompanying
    tutorial.

    Args:
        atomic_numbers (LongTensor, optional): Atomic numbers following ase.Atom, (Unknown=0, H=1) with shape
            :obj:`[num_nodes]`. (default: :obj:`None`)
        pos (Tensor, optional): Node position matrix, only set one position value.
            :obj:`[num_nodes, 3]`. (default: :obj:`None`)
        pbc (BoolTensor, optional): Periodic Boundary Conditions
            :obj:`[1, 3]`. (default: :obj:`None`)
        cell (Tensor, optional): Cell matrix if pbc = True, has shape
            :obj:`[1, 3, 3]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Edge indexes (sender, receiver)
            :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge attributes
            :obj:`[num_edges, num_edge_attr]`. (default: :obj:`None`)
        cell_offsets (IntTensor, optional): Which periodic image does the end of the edge belong to.
            :obj:`[num_edges, 3]`. (default: :obj:`None`)
        node_features (Tensor, optional): Node feature matrix with shape
            :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
        node_descriptors (Tensor, optional): Node descriptor matrix, used for e.g. atomic number embeddings with shape
            :obj:`[num_nodes, num_node_descriptors]`. (default: :obj:`None`)
        node_vector_features (Tensor, optional): Node vector feature matrix, currently only used for TorchMDNet's
             equivariant representation
            :obj:`[num_nodes, num_node_vector_features]`. (default: :obj:`None`)
        node_orientations (Tensor, optional): Node orientation expressed as a 3x3 rotation matrix. Used in some protein representations,
            where each node is a residue with a position and orientation.
            :obj:`[num_nodes, 3, 3]`. (default: :obj:`None`)
        atom_37 (Tensor, optional): Used in an all-atom representation of proteins, where each node is a residue and for each residue
            this item contains 37 positions. If an atom is present in this residue then its corresponding position is included otherwise
            the row is set to 0.
            :obj:`[num_nodes, 37, 3]`. (default: :obj:`None`)
        atom_37_mask (Tensor, optional): Used in all-atom representation of proteins. For each node (residue) this item contains a mask
            for whether that atom is present in this residue.
            :obj:`[num_nodes, 37]`. (default: :obj:`None`)
        energy (Tensor, optional): Graph-level energy label
            :obj:`[1]`. (default: :obj:`None`)
        energy_stats (NormalizationStats, optional): energy normalization stats
        forces (Tensor, optional): Node forces matrix with shape
            :obj:`[num_nodes, 3]`. (default: :obj:`None`)
        forces_stats (NormalizationStats, optional): forces normalization stats
        velocities (Tensor, optional): Node velocities matrix with shape.
            :obj:`[num_nodes, 3]`. (default: :obj:`None`)
        velocities_stats (NormalizationStats, optional): velocities normalization stats
        stress (Tensor, optional): Graph stress matrix with shape
            :obj:`[1, 3, 3]`. (default: :obj:`None`)
        stress_stats (NormalizationStats, optional):  stress normalization stats
        **kwargs (optional): Additional attributes to be stored in the data object.
    """

    def __init__(
        self,
        atomic_numbers: Optional[IntTensor] = None,
        pos: OptTensor = None,
        pbc: Optional[Tuple[bool, bool, bool]] = None,
        cell: OptTensor = None,
        edge_index: Optional[LongTensor] = None,
        edge_attr: OptTensor = None,
        cell_offsets: Optional[IntTensor] = None,
        node_features: OptTensor = None,
        node_descriptors: OptTensor = None,
        node_vector_features: OptTensor = None,
        node_orientations: OptTensor = None,
        atom_37: OptTensor = None,
        atom_37_mask: OptTensor = None,
        energy: OptTensor = None,
        energy_stats: Optional[Union[NormalizationStats, List[NormalizationStats]]] = None,
        forces: OptTensor = None,
        forces_stats: Optional[Union[NormalizationStats, List[NormalizationStats]]] = None,
        velocities: OptTensor = None,
        velocities_stats: Optional[Union[NormalizationStats, List[NormalizationStats]]] = None,
        stress: OptTensor = None,
        stress_stats: Optional[Union[NormalizationStats, List[NormalizationStats]]] = None,
        **kwargs,
    ):
        super().__init__(x=None, edge_index=edge_index, edge_attr=edge_attr, pos=pos, **kwargs)
        if atomic_numbers is not None:
            self.atomic_numbers = atomic_numbers
        if cell is not None:
            self.cell = cell
        if pbc is not None:
            self.pbc = pbc
        if cell_offsets is not None:
            self.cell_offsets = cell_offsets
        if node_features is not None:
            self.node_features = node_features
        if node_descriptors is not None:
            self.node_descriptors = node_descriptors
        if node_vector_features is not None:
            self.node_vector_features = node_vector_features
        if node_orientations is not None:
            self.node_orientations = node_orientations
        if atom_37 is not None:
            self.atom_37 = atom_37
        if atom_37_mask is not None:
            self.atom_37_mask = atom_37_mask
        if energy is not None:
            self.energy = energy
        if energy_stats is not None:
            self.energy_stats = energy_stats
        if forces is not None:
            self.forces = forces
        if forces_stats is not None:
            self.forces_stats = forces_stats
        if velocities is not None:
            self.velocities = velocities
        if velocities_stats is not None:
            self.velocities_stats = velocities_stats
        if stress is not None:
            self.stress = stress
        if stress_stats is not None:
            self.stress_stats = stress_stats
        self.__dict__["_frozen"] = True

    def __setattr__(self, attr, value):
        if self.__dict__.get("_frozen", False) and attr not in (
            "_num_graphs",
            "_slice_dict",
            "_inc_dict",
            "_collate_structure",
        ):
            raise AttributeError(
                f"Replacing ChemGraph.{attr} in-place. Consider using the self.replace method to create a shallow copy."
            )
        return super().__setattr__(attr, value)

    def replace(
        self, **kwargs: Union[OptTensor, str, int, float, NormalizationStats, list]
    ) -> "ChemGraph":
        """Returns a shallow copy of the ChemGraph with updated fields."""
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__["_store"] = copy.copy(self._store)
        for key, value in kwargs.items():
            out._store[key] = value
        out._store._parent = out
        return out

    def get_batch_idx(self, field_name: str) -> Optional[LongTensor]:
        """Used by diffusion library to retrieve batch indices for a given field."""
        assert isinstance(
            self, pyg_data.Batch
        )  # ChemGraphBatch subclass is dynamically defined by PyG
        if field_name == "cell":
            # Graph-level attributes become 'dense' fields where the first dimension is batch dimension.
            return None
        elif field_name in [
            "pos",
            "atomic_numbers",
            "node_features",
            "node_descriptors",
            "node_vector_features",
            "forces",
            "velocities",
            "node_orientations",
            "atom_37",
            "atom_37_mask",
        ]:
            # per-node attributes
            return self.batch
        else:
            try:
                # This happens if 'follow_batch' kwarg was used when constructing the batch
                return self[f"{field_name}_batch"]
            except KeyError:
                raise NotImplementedError(f"Unable to determine batch index for {field_name}")

    def get_batch_size(self):
        # For diffusion library. Only works if self is a ChemGraphBatch
        assert isinstance(self, pyg_data.Batch)
        return self.num_graphs

    def subgraph(self, subset: Tensor) -> "ChemGraph":
        """
        Returns the induced subgraph given by the node indices :obj:`subset`. If no edge indices are
        present, subsets will only be created for node features.

        Args:
            subset (LongTensor or BoolTensor): The nodes to keep.
        """
        # Check for boolean mask or index array.
        if subset.dtype == torch.bool:
            num_nodes = int(subset.sum())
        else:
            num_nodes = subset.size(0)
            subset = torch.unique(subset, sorted=True)

        # If edge indices are provided, determine subgraph components. Otherwise use only `subset`
        # to select relevant nodes of node attributes.
        if self.edge_index is not None:
            out = utils.subgraph(
                subset,
                self.edge_index,
                relabel_nodes=True,
                num_nodes=self.num_nodes,
                return_edge_mask=True,
            )
            edge_index, _, edge_mask = out
        else:
            edge_index = None
            edge_mask = None

        # Create dictionary of the subsets of all quantities.
        masked_data = {}
        for key, value in self:
            if value is None:
                continue
            if key == "edge_index":
                masked_data[key] = edge_index
            if key == "num_nodes":
                masked_data[key] = num_nodes
            elif self.is_node_attr(key):
                cat_dim = self.__cat_dim__(key, value)
                masked_data[key] = utils.select(value, subset, dim=cat_dim)
            elif self.is_edge_attr(key) and edge_index is not None:
                cat_dim = self.__cat_dim__(key, value)
                masked_data[key] = utils.select(value, edge_mask, dim=cat_dim)

        # Generate final graph.
        data = self.replace(**masked_data)

        return data
    
    def clone_with_pos_cell(graph: "ChemGraph") -> "ChemGraph":
        """
        创建一个新的 ChemGraph 实例，只拷贝 pos 和 cell 属性。
        
        Args:
            graph (ChemGraph): 要拷贝的 ChemGraph 实例。
        
        Returns:
            ChemGraph: 仅包含 pos 和 cell 属性的新 ChemGraph 实例。
        """
        return ChemGraph(
            pos=graph.pos.clone() if graph.pos is not None else None,
            cell=graph.cell.clone() if graph.cell is not None else None
        )


MLPotentialData = ChemGraph  # To enable unpickling legacy processed dataset caches.
# Retrieve a pointer for the DynamicInheritance-based PYG Batch class.
# For typing reasons only, use isinstance(pyg_data.Batch) for runtime checks.
ChemGraphBatch = pyg_data.Batch(_base_cls=ChemGraph).__class__
