import torch
import torch.nn as nn

from meta_infomax.models.classifier import Classifier
from meta_infomax.models.encoder import Encoder


class MultiTaskInfoMax(nn.Module):

    def __init__(self, shared_encoder, embeddings, vocab, encoder_dim, encoder_layers, classifier_dim, out_dim):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(embeddings, freeze=True, padding_idx=vocab.stoi["<pad>"])
        self.shared_encoder = shared_encoder
        self.private_encoder = Encoder(embeddings.shape[-1], encoder_dim, encoder_layers)
        self.classifier = Classifier(encoder_dim * 4, classifier_dim, out_dim)

    def forward(self, sentences, lengths):
        sent_embed = self.emb(sentences)
        shared_out = self.shared_encoder(sent_embed)
        private_out = self.private_encoder(sent_embed)
        h = torch.cat((shared_out, private_out), dim=1)
        out = self.classifier(h)
        return out, shared_out, private_out
