import torch
import torch.nn as nn


class Protonet(nn.Module):
    ''' Prototypical Network '''

    def __init__(self, bert_encoder, bert_embed_dim = 768):
        super().__init__()
        self.bert_encoder = bert_encoder
        self.head_encoder = nn.Sequential(
            nn.Linear(bert_embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
    
    def forward(self, input_seq, attn_mask):
        bert_output = self.bert_encoder(input_seq, attn_mask)
        bert_embed = bert_output[0][:, 0] # CLS embedding for sequence
        embed = self.head_encoder(bert_embed)
        return embed

    def freeze_bert(self, until_layer):
        """ Freezes layers of BERT until given layer index
        
        Parameters
        ---
        until_layer: int
            layers below until_layer index will be frozen 
        """
        # start by un-freezing everything
        # (not the optimal way, but makes the later computations simpler)
        for param in self.bert_encoder.parameters():
            param.requires_grad = True

        # always freeze embedding layer
        for param in self.bert_encoder.embeddings.parameters():
            param.requires_grad = False

        layers_to_freeze = range(0, until_layer)
        for layer_idx in layers_to_freeze:
            for param in self.bert_encoder.encoder.layer[layer_idx].parameters():
                param.requires_grad = False