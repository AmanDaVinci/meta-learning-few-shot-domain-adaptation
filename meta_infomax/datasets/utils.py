"""
Utility functions.
"""
from pathlib import Path
import pandas as pd

from typing import List, Dict, Tuple, Union
import torch
import numpy as np

from torchtext.data import Field, LabelField, TabularDataset, Dataset, Iterator
from pathlib import Path

GLOVE_DIR = Path("../data/glove")


def sample_domains(data: Dict[str, Dict[str, List[Instance]]], n_samples: int=5, strategy: str='uniform') -> np.ndarray:
    """Sample domains from data according to strategy
    
    Parameters
    ----------
    data: contains all data splits for domains
    n_samples: number of samples to draw
    strategy: in {'uniform', 'domain_size'}
        specifies sampling strategy:
        'uniform': sample uniformly
        'domain_size': sample more from domain if size of domain is larger
        
    Returns
    -------
    array of domain names
    """
    assert strategy in ('uniform', 'domain_size'), 'specify correct strategy'
    domains = np.array([d for d in data.keys()])
    n_domains = len(domains)
    if strategy == 'uniform':
        weights = [1] * n_domains
    elif strategy == 'domain_size':
        weights = [len(data[domain]['train']) for domain in domains]

    sampler = torch.utils.data.WeightedRandomSampler([1 / n_domains] * n_domains, num_samples=n_samples,
                                                     replacement=False)
    return domains[list(sampler)]


def get_iterators(data: Dict[str, Dict[str, Dataset]], split: str='train',
                  collapse_domains: bool=False, batch_size: int=64, device='cpu', **kwargs) -> Dict[str, 'iterator']:
    """
    Generate iterators of each domain from split.
    
    Parameters
    ----------
    data: ``Dict[str, Dict[str, List[Instance]]]`` contains all datasets.
    split: in {'train', 'val', 'test'}
    collapse_domains: bool
        If True, make one big iterator from all datasets. This means that different domains will be present
        in the same batch.
    batch_size: batch size
    device: device of tensor
    kwargs: kwargs passed to iterators.
    
    Returns
    -------
    iterators: ``Dict[str, iterator]``, where it is indexed by domain. If collapse_domain=True,
    we collapse all the domains into one single iterator, which is indexed by 'all' in the returned dict.
    """
    iterators = {}
    if collapse_domains:
        # collect instances from `split` of every domain
        all_examples = [example for domain, splits in data.items() for example in splits[split].examples]
        arbitrary_split_fields = list(data.values())[0][split].fields
        all_dataset = Dataset(all_examples, fields=arbitrary_split_fields)
        iterators['all'] = Iterator(all_dataset, batch_size=batch_size, device=device)
    else:
        for dataset, splits in data.items():
            iterators[dataset] = Iterator(splits[split], batch_size=batch_size, device=device)
    return iterators


def remove_outlier_lengths(data, quantile: float=0.995):
    """Remove outliers in place and return thrown away indices."""
    throw_away_ixs = {}
    for dataset, splits in data.items():
        throw_away_ixs[dataset] = {}
        for split, split_set in splits.items():
            if split == 'test':
                # we don't want to remove test samples.
                continue
            lengths = np.array([len(example.text) for example in split_set.examples])
            keep_lengths = lengths < np.quantile(lengths, quantile)
            throw_away = (keep_lengths == 0).nonzero()[0]
            for ix in throw_away:
                del split_set.examples[ix]
            throw_away_ixs[dataset][split] = throw_away
    return throw_away_ixs


from transformers import BertTokenizer, BertModel, BertForMaskedLM
#                     Model          | Tokenizer          | Pretrained weights shortcut
TRANSFORMER_MODELS = {'bert-base-uncased': (BertModel,       BertTokenizer,   'bert-base-uncased')}
TRANSFORMER_EMBEDDING_DIMS = {'bert-base-uncased': 768}

# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)

def get_embedding_dim(model_name):
    """
    Get embedding dim of transformer model.

    Parameters
    ----------
    model_name: str
        Name of model's embedding dim to fetch.

    Returns
    -------
    embedding_dim, int
    """
    return TRANSFORMER_EMBEDDING_DIMS[model_name]


def get_transformer(model_name):
    """
    Return model from transformers library.

    Parameters
    ----------
    model_name: str
        Name of model to fetch. Also called `pretrained weights shortcut' by huggingface people.

    Returns
    -------
    tuple model, tokenizer, embeddding_dim according to model_name
    """
    model_class, tokenizer_class, pretrained_weights = TRANSFORMER_MODELS[model_name]
    model = model_class.from_pretrained(pretrained_weights,
                                        output_hidden_states=True)
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)

    return model, tokenizer, get_embedding_dim(model_name)


def get_vocab(data):
    """Not necessary if we use a pretrained model."""
    return Vocabulary.from_instances([instance for splits in data.values()
                                    for instance in splits['train']],
                                     min_count={'tokens': min_count})