import torch
import torch.nn as nn
from typing import Union, List, Dict


class Classifier(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        return self.layers(x)


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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = inputs
        for layer, activation, dropout in zip(
            self._linear_layers[:-1], self._activations[:-1], self._dropout[:-1]
        ):
            output = dropout(activation(layer(output)))
        output = self._linear_layers[-1](output)
        return output

class SentimentClassifier(nn.Module):
    def __init__(self,
                 text_encoder,
                 classifier_feedforward,
                ):
        """
        Parameters
        ----------
        """
        super().__init__()

        self.text_encoder = text_encoder
        self.num_classes = 2
        self.classifier_feedforward = classifier_feedforward
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self,
                text: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        text : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.
            
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        encoded_text = self.text_encoder(text)[0][:,0] # todo: make abstraction model that

        logits = self.classifier_feedforward(encoded_text)
        output_dict = {'logits': logits}

        if label is not None:
            loss = self.criterion(logits, label)
#             acc = accuracy(logits, label)
            output_dict["loss"] = loss
#             output_dict["acc"] = acc

        return output_dict
    
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        predictions = output_dict['class_probabilities'].cpu().data.numpy()
        argmax_indices = np.argmax(predictions, axis=-1)
#         labels = [self.vocab.get_token_from_index(x, namespace="labels")
#                   for x in argmax_indices]
        labels = -1 # TODO
        output_dict['label'] = labels
        return output_dict
    
    
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
