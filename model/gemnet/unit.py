import enum
from typing import Union, Optional, NamedTuple

import torch


class Unit(enum.Enum):
    def __str__(self) -> str:
        return str(self.value)


class EnergyUnit(Unit):
    ev = "ev"
    hartree = "hartree"
    kcal_per_mole = "kcal_per_mole"
    unknown = "unknown"


class LengthUnit(Unit):
    bohr = "bohr"
    angstrom = "angstrom"
    unknown = "unknown"


class ForceUnit(Unit):
    kcal_per_mole_per_angstrom = "kcal_per_mole_per_angstrom"
    ev_per_angstrom = "ev_per_angstrom"
    hartree_per_angstrom = "hartree_per_angstrom"
    unknown = "unknown"


class StressUnit(Unit):
    ev_per_angstrom_cube = "ev_per_angstrom_cube"
    unknown = "unknown"


class DipoleMomentUnit(Unit):
    coulomb_meter = "coulomb_meter"
    debye = "debye"  # also known as D
    unknown = "unknown"


class NormalizationStats(NamedTuple):
    """Describes the (potentially normalized) unit of a value, and the statistics to (de)normalize it."""

    unit: Unit
    normalized: bool
    mean: Optional[Union[float, torch.FloatTensor]] = None
    std: Optional[Union[float, torch.FloatTensor]] = None

    def __eq__(self, o: object) -> bool:
        if isinstance(o, NormalizationStats):
            return (
                self.unit == o.unit
                and self.normalized == o.normalized
                and torch.equal(self.mean, torch.tensor(o.mean))
                if isinstance(self.mean, torch.Tensor)
                else self.mean == o.mean and torch.equal(self.std, torch.tensor(o.std))
                if isinstance(self.std, torch.Tensor)
                else self.std == o.std
            )
        else:
            return self == o
