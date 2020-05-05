import torch
import torch.nn as nn
from typing import Dict, List

from meta_infomax.models.feed_forward import FeedForward


def accuracy(y_pred: torch.Tensor, y: torch.Tensor) -> float:
    """
    Return predictive accuracy.

    Parameters
    ---
    y_pred: torch.Tensor (BATCH, NUM_CLASS)
        Predicted values.
    y: torch.Tensor (BATCH,)
        Real class values.
    """
    return (y_pred.argmax(dim = 1) == y).float().mean().item()


class SentimentClassifier(nn.Module):
    def __init__(self, encoder, head: FeedForward, pooler=None):
        """
        Parameters
        ----------
        encoder: Transformer model
        head: FeedForward
            Classifier on top of encoder.
        pooler: function, optional
            Takes output of BERT model and returns a sentence embedding. Default takes the embedding of
            the CLS token.
        """
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.num_classes = 2
        self.criterion = torch.nn.CrossEntropyLoss()
        self.pooler = pooler if pooler is not None else lambda x: x[0][:,0] # get CLS embedding for each sentence in batch

    def forward(self,
                x: torch.Tensor,
                masks: torch.Tensor = None,
                labels: torch.LongTensor = None,
                domains: List[str]=None) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor, (BATCH, max_seq_len), required
            Input ids of words. They are already encoded, batched, padded, and special tokens added. 
        masks: torch.Tensor (BATCH, max_seq_len), optional
            Array where values indicate whether the transformer should consider this token or not.
        labels : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.
        domains: List[str]
            Domain of each sample in batch.
            
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """

        sentence_embedding = self.encode(x, masks)
        return self.classify_encoded(sentence_embedding, labels)



    def encode(self,
                x: torch.Tensor,
                masks: torch.Tensor = None):

        ### run the data through the encoder part of the model (Transformer)
        if masks is None:
            # if masks not provided, we don't mask any observation
            masks = torch.ones_like(x)
        encoded_text = self.encoder(input_ids=x, attention_mask=masks)
        sentence_embedding = self.pooler(encoded_text)

        return sentence_embedding

    def classify_encoded(self, sentence_embedding, labels = None, custom_params = None):
     
        logits = self.head(sentence_embedding, custom_params = custom_params)
        output_dict = {'logits': logits}

        print("logits shape")
        print(logits.shape)
        print(labels.shape)

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
        """Make layer of a huggingface Transformer model require a gradient.
        
        Parameters
        ---
        layers: iterable(int)
            Layers to unfreeze.
        """
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
