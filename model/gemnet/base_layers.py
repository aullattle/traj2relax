"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import abc
import math
from collections.abc import Callable
from typing import Optional
from typing import (
    Callable,
    Optional,
    Protocol,
    TypeVar,
    Generic,
    TypeVar,
    runtime_checkable,
)
import torch

def _standardize(kernel):
    """
    Makes sure that N*Var(W) = 1 and E[W] = 0
    """
    eps = 1e-6

    if len(kernel.shape) == 3:
        axis = (0, 1)  # last dimension is output dimension
    else:
        axis = 1

    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel

def he_orthogonal_init(tensor: torch.Tensor) -> torch.Tensor:
    """
    Generate a weight matrix with variance according to He (Kaiming) initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
    """
    tensor = torch.nn.init.orthogonal_(tensor)

    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor


T = TypeVar("T")
@runtime_checkable
class BatchedData(Protocol):
    def replace(self: T, **vals: torch.Tensor) -> T:
        """Return a copy of self with some fields replaced with new values."""

    def get_batch_idx(self, field_name: str) -> Optional[torch.LongTensor]:
        """Get the batch index (i.e., which row belongs to which sample) for a given field.
        For 'dense' type data, where every sample has the same shape and the first dimension is the
        batch dimension, this method should return None. Mathematically,
        returning None will be treated the same as returning a tensor [0, 1, 2, ..., batch_size - 1]
        but I expect memory access in other functions to be more efficient if you return None.
        """

    def get_batch_size(self) -> int:
        """Get the batch size."""

    def device(self) -> torch.device:
        """Get the device of the batch."""

    def __getitem__(self, field_name: str) -> torch.Tensor:
        """Get a field from the batch."""

    def to(self: T, device: torch.device) -> T:
        """Move the batch to a given device."""

    def clone(self: T) -> T:
        """Return a copy with all the tensors cloned."""


Diffusable = TypeVar("Diffusable", bound=BatchedData)


class ScoreModel(torch.nn.Module, Generic[Diffusable], abc.ABC):
    """Abstract base class for score models."""

    @abc.abstractmethod
    def forward(self, x: Diffusable, t: torch.Tensor) -> Diffusable:
        """Args:
        x: batch of noisy data
        t: timestep. Shape (batch_size, 1)
        """
        ...

class Dense(torch.nn.Module):
    """
    Combines dense layer with scaling for swish activation.

    Parameters
    ----------
        units: int
            Output embedding size.
        activation: str
            Name of the activation function to use.
        bias: bool
            True if use bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation: Optional[str] = None,
    ):
        super().__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

        if isinstance(activation, str):
            activation = activation.lower()
        if activation in ["swish", "silu"]:
            self._activation = ScaledSiLU()
        elif activation == "siqu":
            self._activation = SiQU()
        elif activation is None:
            self._activation = torch.nn.Identity()
        else:
            raise NotImplementedError("Activation function not implemented for GemNet (yet).")

    def reset_parameters(self, initializer: Callable = he_orthogonal_init):
        initializer(self.linear.weight)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = self._activation(x)
        return x

class ScaledSiLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_factor = 1 / 0.6
        self._activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor):
        return self._activation(x) * self.scale_factor

class SiQU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._activation = torch.nn.SiLU()

    def forward(self, x: torch.Tensor):
        return x * self._activation(x)

class ResidualLayer(torch.nn.Module):
    """
    Residual block with output scaled by 1/sqrt(2).

    Parameters
    ----------
        units: int
            Output embedding size.
        nLayers: int
            Number of dense layers.
        layer_kwargs: str
            Keyword arguments for initializing the layers.
    """

    def __init__(self, units: int, nLayers: int = 2, layer: Callable = Dense, **layer_kwargs):
        super().__init__()
        self.dense_mlp = torch.nn.Sequential(
            *[
                layer(in_features=units, out_features=units, bias=False, **layer_kwargs)
                for _ in range(nLayers)
            ]
        )
        self.inv_sqrt_2 = 1 / math.sqrt(2)

    def forward(self, input: torch.Tensor):
        x = self.dense_mlp(input)
        x = input + x
        x = x * self.inv_sqrt_2
        return x
