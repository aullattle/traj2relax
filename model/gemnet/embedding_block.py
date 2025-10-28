"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from dataclasses import dataclass

import numpy as np
import torch
from pymatgen.analysis.bond_valence import ICSD_BV_DATA
from pymatgen.core.periodic_table import Element

# from materials.explorers.common.embeddings import MAX_ATOMIC_NUM, MAX_CHARGE, MIN_CHARGE
from model.gemnet.base_layers import Dense
MAX_ATOMIC_NUM = 100
# Minimal and maximal charges are set according to the table https://github.com/WMD-group/SMACT/blob/master/smact/data/oxidation_states.txt
MIN_CHARGE = -4
MAX_CHARGE = 8
# This is the the number of existing ions that have an atomic number <= 100.
# This can be deduced from the oxidation states in https://pymatgen.org/pymatgen.core.periodic_table.html &
# https://github.com/materialsproject/pymatgen/blob/89fc2d6f37dd4677e8f861f3227985daf6d3f812/pymatgen/analysis/icsd_bv.yaml
NUM_IONS = 505


class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size, with_mask_type=False):
        super().__init__()
        self.emb_size = emb_size

        # Atom embeddings: We go up to Bi (83).
        self.embeddings = torch.nn.Embedding(MAX_ATOMIC_NUM + int(with_mask_type), emb_size)
        # init by uniform distribution
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, Z):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(Z - 1)  # -1 because Z.min()=1 (==Hydrogen)
        return h


class ChargeEmbedding(torch.nn.Module):
    """
    Initial charge embeddings based on the partial charge number

    Parameters
    ----------
        emb_size: int
            Charge embeddings size
    """

    def __init__(
        self,
        emb_size: int,
        with_mask_type: bool = False,
        min_charge: int = MIN_CHARGE,
        max_charge: int = MAX_CHARGE,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.min_charge = min_charge
        self.max_charge = max_charge

        # Charge embeddings:
        self.embeddings = torch.nn.Embedding(
            max_charge - min_charge + 1 + int(with_mask_type), emb_size
        )
        # init by uniform distribution
        torch.nn.init.uniform_(self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3))

    def forward(self, C: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(C - self.min_charge)  # The minimal charge is assign a token 0
        return h


class EdgeEmbedding(torch.nn.Module):
    """
    Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

    Parameters
    ----------
        emb_size: int
            Embedding size after the dense layer.
        activation: str
            Activation function used in the dense layer.
    """

    def __init__(
        self,
        atom_features,
        edge_features,
        out_features,
        activation=None,
    ):
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(in_features, out_features, activation=activation, bias=False)

    def forward(
        self,
        h,
        m_rbf,
        idx_s,
        idx_t,
    ):
        """

        Arguments
        ---------
        h
        m_rbf: shape (nEdges, nFeatures)
            in embedding block: m_rbf = rbf ; In interaction block: m_rbf = m_st
        idx_s
        idx_t

        Returns
        -------
            m_st: torch.Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        h_s = h[idx_s]  # shape=(nEdges, emb_size)
        h_t = h[idx_t]  # shape=(nEdges, emb_size)

        m_st = torch.cat([h_s, h_t, m_rbf], dim=-1)  # (nEdges, 2*emb_size+nFeatures)
        m_st = self.dense(m_st)  # (nEdges, emb_size)
        return m_st


@dataclass
class IonData:
    atom_types: torch.Tensor
    charges: torch.Tensor


class IonHandler:
    """This class handles the mapping between (atom_type, charge) pairs and ion states represented by integers.
    The set of possible ion states is built through pymatgen, by leveraging statistics from the ICSD data.
    """

    def __init__(self, add_mask: bool = True) -> None:
        ion_states = {}
        atom_charges_pairs = []
        # Itterate over al possible elements
        ion_state = 0
        all_elems = list(Element)
        for atom_type in range(1, MAX_ATOMIC_NUM + 1):
            el = all_elems[atom_type - 1]
            # Get all possible oxidation states for the given atom type
            ox_ls = list(el.oxidation_states)
            ox_ls.append(0)
            for charge in ox_ls:
                ion_states[(atom_type, charge)] = ion_state
                ion_state += 1
                atom_charges_pairs.append((atom_type, charge))
        for ion in ICSD_BV_DATA:
            atom_type = ion.element.Z
            charge = ion.oxi_state
            if (atom_type, charge) not in ion_states.keys():
                ion_states[(atom_type, charge)] = ion_state
                ion_state += 1
                atom_charges_pairs.append((atom_type, charge))
        if add_mask:
            ion_states[(MAX_ATOMIC_NUM + 1, MAX_CHARGE + 1)] = ion_state
            atom_charges_pairs.append((MAX_ATOMIC_NUM + 1, MAX_CHARGE + 1))
        self.ion_states = ion_states
        self.atom_charges_pairs = atom_charges_pairs
        self.n_ions = len(self.ion_states)
        self.add_mask = add_mask

    def get_ion_states(self, atom_types: torch.Tensor, charges: torch.Tensor) -> torch.Tensor:
        """Maps the  pairs (atom_types, charges) to a single ion state

        Args:
            atom_types (torch.Tensor): tensor of shape (nAtoms,) that specifies each ion's atom species
            charges (torch.Tensor): tensor of shape (nAtoms,) that specifies each ion's oxidation state

        Returns:
            torch.Tensor: tensor of shape (nAtoms,) that specifies each ion state
        """
        assert atom_types.shape == charges.shape
        assert 1 <= torch.min(atom_types) and torch.max(atom_types) <= MAX_ATOMIC_NUM + 1
        assert MIN_CHARGE <= torch.min(charges) and torch.max(charges) <= MAX_CHARGE + 1
        # Charge and atom type should be set to their absorbing state simultaneously
        assert torch.equal(atom_types == MAX_ATOMIC_NUM + 1, charges == MAX_CHARGE + 1)
        ion_states = torch.tensor(
            [
                self.ion_states[(atom_type.item(), charge.item())]
                for atom_type, charge in zip(atom_types, charges)
            ]
        )  # This returns the integer index of the ion state on a (Z,C) matrix
        return ion_states.long().to(atom_types.device)

    def unpack_ion_states(self, ion_states: torch.Tensor) -> IonData:
        """Extracts atom species from ion states

        Args:
            ion_states (torch.Tensor): tensor of shape (nAtoms,) that specifies each ion state

        Returns:
            IonData: data structure containing atom species and charges
        """
        assert 0 <= torch.min(ion_states) and torch.max(ion_states) <= self.n_ions
        atom_types = (
            torch.tensor([self.atom_charges_pairs[ion_state.item()][0] for ion_state in ion_states])
            .long()
            .to(ion_states.device)
        )
        charges = (
            torch.tensor([self.atom_charges_pairs[ion_state.item()][1] for ion_state in ion_states])
            .long()
            .to(ion_states.device)
        )
        return IonData(atom_types=atom_types, charges=charges)
