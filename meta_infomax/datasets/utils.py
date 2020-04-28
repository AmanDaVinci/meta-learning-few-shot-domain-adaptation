"""
Utility functions.
"""
from pathlib import Path
import pandas as pd

from typing import Iterator, List, Dict, Tuple, Union
import torch
import numpy as np

from allennlp.common.params import Params
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data.iterators import BucketIterator, BasicIterator
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list


from pathlib import Path

GLOVE_DIR = Path("../data/glove")


def get_glove_embeddings(vocab: Vocabulary, glove_dir: Union[Path, str]=GLOVE_DIR,
                        glove_file: Union[Path, str]='glove.840B.300d.txt'):
    """Load Glove embeddings from file and generate embeddings."""
    glove_dir = Path(glove_dir)
    glove_params = Params({
        'pretrained_file': (glove_dir / glove_file).as_posix(),
        'embedding_dim': 300,
        'trainable': False
    })
    return Embedding.from_params(vocab, glove_params)


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


def get_iterators(data: Dict[str, Dict[str, List[Instance]]], vocab: Vocabulary, split: str='train',
                  collapse_domains: bool=False, batch_size: int=64, use_bucket: bool=False, **kwargs) -> Dict[str, Iterator]:
    """
    Generate iterators of each domain from split.
    
    Parameters
    ----------
    data: ``Dict[str, Dict[str, List[Instance]]]`` contains all datasets.
    vocab: ``Vocabulary``
    split: in {'train', 'val', 'test'}
    collapse_domains: bool
        If True, make one big iterator from all datasets. This means that different domains will be present
        in the same batch.
    batch_size: batch size
    use_bucket:
        If true, use bucket iterator. Otherwise use normal one.
    kwargs: kwargs passed to iterators.
    
    Returns
    -------
    iterators: ``Dict[str, iterator]``, where it is indexed by domain. If collapse_domain=True,
    we collapse all the domains into one single iterator, which is indexed by 'all' in the returned dict.
    """
    iterators = {}
    if use_bucket:
        iterator = BucketIterator(
            batch_size=batch_size,
            sorting_keys=[('text', 'num_tokens')], # sort by num_tokens of sentence field for efficient batching
            padding_noise=kwargs.get('padding_noise', 0.4) # so that we put a bit of randomness when sorting by padding length
        )
    else:
        iterator = BasicIterator(batch_size=batch_size)
    iterator.index_with(vocab) # for determining padding lengths
    # generate generators
    if collapse_domains:
        # collect instances from `split` of every domain
        all_instances = [instance for domain, splits in data.items() for instance in splits[split]]
        iterators['all'] = iterator(all_instances)
    else:
        for dataset, splits in data.items():
            iterators[dataset] = iterator(splits[split])
    return iterators


def remove_outlier_lengths(data, quantile: float=0.995):
    """Remove outliers in place and return thrown away indices."""
    throw_away_ixs = {}
    for dataset, splits in data.items():
        throw_away_ixs[dataset] = {}
        for split, instances in splits.items():
            if split == 'test':
                # we don't want to remove test samples.
                continue
            lengths = np.array([len(instance.get('text').tokens) for instance in instances])
            keep_lengths = lengths < np.quantile(lengths, quantile)
            throw_away = (keep_lengths == 0).nonzero()[0]
            for ix in throw_away:
                del instances[ix]
            throw_away_ixs[dataset][split] = throw_away
    return throw_away_ixs
