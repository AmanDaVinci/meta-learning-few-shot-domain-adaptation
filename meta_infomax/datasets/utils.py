"""
Utility functions.
"""

import numpy as np
import torch
from pathlib import Path
from torchtext.data import Dataset, Iterator
from typing import List, Dict
import logging

GLOVE_DIR = Path("../data/glove")
LOG_PATH = Path('logs')


def sample_domains(data: Dict[str, Dict[str, List[Dataset]]], n_samples: int = 5, strategy: str = 'uniform') -> np.ndarray:
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
    return domains[list(sampler)].tolist()


def get_iterators(data: Dict[str, Dict[str, Dataset]], include_domains: List[str], split: str = 'train', 
                  collapse_domains: bool = False, batch_size: int = 64, device='cpu', **kwargs) -> Dict[str, 'Iterator']:
    """
    Generate iterators of each domain from split.
    
    Parameters
    ----------
    data: ``Dict[str, Dict[str, List[Dataset]]]`` contains all datasets.
    include_domains: list of domains to include
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
        all_examples = [example for domain, splits in data.items()\
                                for example in splits[split].examples\
                                if domain in include_domains]
        arbitrary_split_fields = list(data.values())[0][split].fields
        all_dataset = Dataset(all_examples, fields=arbitrary_split_fields)
        iterators['all'] = Iterator(all_dataset, batch_size=batch_size, device=device)
    else:
        for domain, splits in data.items():
            if domain in include_domains:
                iterators[domain] = Iterator(splits[split], batch_size=batch_size, device=device)
    return iterators


def remove_outlier_lengths(data, quantile: float = 0.995):
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


from transformers import BertTokenizer, BertModel

#                                           Model     | Tokenizer   | Pretrained weights shortcut
TRANSFORMER_MODELS = {'bert-base-uncased': (BertModel, BertTokenizer, 'bert-base-uncased')}
TRANSFORMER_EMBEDDING_DIMS = {'bert-base-uncased': 768}


# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
# import logging
# logging.basicConfig(level=logging.INFO)


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

    return model, tokenizer, TRANSFORMER_EMBEDDING_DIMS[model_name]


def init_logging(log_path=LOG_PATH, log_level=logging.DEBUG):
    """Initialize logger."""
    log_path = Path(log_path)
    logging.basicConfig(level=log_level) # set minimum log level
    # format how the log will be printed. Some parameters are already defined in LogRecord like asctime used below
    # more info at https://docs.python.org/3/library/logging.html#logrecord-attributes
    logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger()

    # FileHandler created with log file path.
    # TODO: log file name adaptive to experiment
    log_path.mkdir(exist_ok=True, parents=True)
    fileHandler = logging.FileHandler('{0}/{1}.log'.format(log_path, 'logs'))
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    # This allows logs to be printed on console as well.
    # consoleHandler = logging.StreamHandler()
    # consoleHandler.setFormatter(logFormatter)
    # rootLogger.addHandler(consoleHandler)
