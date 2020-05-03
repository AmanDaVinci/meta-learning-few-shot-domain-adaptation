import torch
import numpy as np
from numpy.random import choice
from torch.nn.utils.rnn import pad_sequence


class FudanMetaSampler(object):
    ''' yields a batch of indexes for each episode of a meta-learning iteration '''

    def __init__(self, labels: torch.Tensor, domains: torch.Tensor, n_classes: int, n_samples: int, n_episodes: int):
        ''' Initialize the index matrix to sample episodes from

        Parameters
        ---
        labels:
        vector of sentiment labels

        domains:
        vector of domains to sample each episode from

        n_classes:
        number of classes in one episode

        n_samples:
        number of samples from each class

        n_episodes:
        number of episodes per epoch
        '''
        super().__init__()
        self.labels = labels
        self.domains = domains
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_episodes = n_episodes

        # concatenate domains and labels to form unique classes
        domain_label = np.array([str(domain.item())+str(label.item()) \
                                for domain, label in zip(self.domains, self.labels)])
        self.classes, self.class_counts = np.unique(domain_label, return_counts=True)
        self.class_indexes = [np.where(domain_label == c)[0] for c in self.classes]

    def __iter__(self):
        ''' yield a batch of indexes '''
        for i in range(self.n_episodes):
            sampled_classes = choice(len(self.classes), size=self.n_classes)
            batch = [choice(self.class_indexes[c], size=self.n_samples) for c in sampled_classes]
            yield torch.tensor(batch).flatten()

    def __len__(self):
        return self.n_episodes
