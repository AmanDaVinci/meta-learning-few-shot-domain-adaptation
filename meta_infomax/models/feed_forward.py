
import torch
import torch.nn as nn
from typing import Union, List, Dict
from torch.nn import functional as F


class FeedForward(nn.Module):
    """
    This `Module` is a feed-forward neural network, just a sequence of `Linear` layers with
    activation functions in between.
    # Parameters
    input_dim : `int`, required
        The dimensionality of the input.  We assume the input has shape `(batch_size, input_dim)`.
    num_layers : `int`, required
        The number of `Linear` layers to apply to the input.
    hidden_dims : `Union[int, List[int]]`, required
        The output dimension of each of the `Linear` layers.  If this is a single `int`, we use
        it for all `Linear` layers.  If it is a `List[int]`, `len(hidden_dims)` must be
        `num_layers`.
    activations : `Union[Callable, List[Callable]]`, required
        The activation function to use after each `Linear` layer.  If this is a single function,
        we use it after all `Linear` layers.  If it is a `List[Callable]`,
        `len(activations)` must be `num_layers`.
    dropout : `Union[float, List[float]]`, optional (default = 0.0)
        If given, we will apply this amount of dropout after each layer.  Semantics of `float`
        versus `List[float]` is the same as with other parameters.
    """

    def __init__(
            self,
            input_dim: int,
            num_layers: int,
            hidden_dims: Union[int, List[int]],
            activations: Union['Activation', List['Activation']],
            dropout: Union[float, List[float]] = 0.0,
    ) -> None:

        super().__init__()
        if not isinstance(hidden_dims, list):
            hidden_dims = [hidden_dims] * num_layers  # type: ignore
        if not isinstance(activations, list):
            activations = [activations] * num_layers  # type: ignore
        if not isinstance(dropout, list):
            dropout = [dropout] * num_layers  # type: ignore
        if len(hidden_dims) != num_layers:
            raise ConfigurationError(
                "len(hidden_dims) (%d) != num_layers (%d)" % (len(hidden_dims), num_layers)
            )
        if len(activations) != num_layers:
            raise ConfigurationError(
                "len(activations) (%d) != num_layers (%d)" % (len(activations), num_layers)
            )
        if len(dropout) != num_layers:
            raise ConfigurationError(
                "len(dropout) (%d) != num_layers (%d)" % (len(dropout), num_layers)
            )
        self._activations = activations
        input_dims = [input_dim] + hidden_dims[:-1]
        linear_layers = []
        for layer_input_dim, layer_output_dim in zip(input_dims, hidden_dims):
            linear_layers.append(torch.nn.Linear(layer_input_dim, layer_output_dim))
        self._linear_layers = torch.nn.ModuleList(linear_layers)
        dropout_layers = [torch.nn.Dropout(p=value) for value in dropout]
        self._dropout = torch.nn.ModuleList(dropout_layers)
        self._output_dim = hidden_dims[-1]
        self.input_dim = input_dim

    def get_output_dim(self):
        return self._output_dim

    def get_input_dim(self):
        return self.input_dim

    def forward(self, inputs: torch.Tensor, custom_params = None) -> torch.Tensor:
        if not custom_params:
            output = inputs
            for layer, activation, dropout in zip(self._linear_layers[:-1], self._activations[:-1], self._dropout[:-1]):
                output = dropout(activation(layer(output)))

            output = self._linear_layers[-1](output)
        else:

            output = inputs
            layer_ind = 0
            for layer, activation, dropout in zip(self._linear_layers[:-1], self._activations[:-1], self._dropout[:-1]):
                output = F.dropout(F.relu(F.linear(output, custom_params[layer_ind], custom_params[layer_ind+1])), 0.5)
                layer_ind += 2
            
            output = F.linear(output, custom_params[-2], custom_params[-1])
   
        return output
