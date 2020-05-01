import torch
import torch.nn as nn
from typing import Dict

from meta_infomax.models.feed_forward import FeedForward


def accuracy(y_pred: torch.Tensor, y: torch.Tensor) -> float:
    return (y_pred.argmax(dim=1) == y).float().mean().item()


class SentimentClassifier(nn.Module):
    def __init__(self, encoder, head: FeedForward):
        """
        Parameters
        ----------
        """
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.num_classes = 2
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self,
                text: torch.Tensor,
                labels: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        text : torch.Tensor, required
            The output of ``TextField.as_array()``.
        labels : Variable, optional (default = None)
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
        encoded_text = self.encoder(text)[0][:, 0]  # todo: make abstraction model that

        logits = self.head(encoded_text)
        output_dict = {'logits': logits}

        if labels is not None:
            output_dict["loss"] = self.criterion(logits, labels)
            output_dict["acc"] = accuracy(logits, labels)

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

    def encoder_unfreeze_layers(self, layers=(10, 11)):
        for name, param in self.encoder.named_parameters():
            if name.startswith(f"encoder"):
                layer_index = int(name.split(".")[2])
                if layer_index in layers:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            elif name.startswith(f"pooler"):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}
