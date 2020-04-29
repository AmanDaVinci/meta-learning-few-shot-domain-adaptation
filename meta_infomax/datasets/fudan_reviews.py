from pathlib import Path
import pandas as pd

from typing import Iterator, List, Dict, Tuple, Union
import torch
import numpy as np

import torchtext
from torchtext.data import Field, LabelField, TabularDataset, Dataset

from sklearn.model_selection import train_test_split

DATA_DIR = Path("../data/mtl-dataset")
DATASETS = ['apparel', 'baby', 'books', 'camera_photo',  'electronics', 
      'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines', 
      'music', 'software', 'sports_outdoors', 'toys_games', 'video']
MODEL_NAME = 'bert-base-uncased'


def get_fudan_datasets(tokenizer, datasets: List[str]=DATASETS, data_dir: Union[str, Path]=DATA_DIR,
                       validation_size: float=0.2, min_count: int=0,
                       random_state: int=11) -> Dict[str, Dict[str, List[Dataset]]]:
    """
    Get all fudan datasets specified in a dictionary. Also return vocabulary.
    Gets all train, val, and test tests, with corresponding keys in the returned dict for each dataset.

    Parameters
    ----------
    tokenizer: bert
    datasets:
        Specify domain such as ['apparel', 'baby'].
    data_dir:
        Directory of fudan datasets.
    validation_size: should be in [0, 1].
        Fraction of validation samples in train set.
    min_count:
        Token should appear min_count times to be counted in vocabulary.
    random_state:
        Used for validation split.

    Returns
    -------
    dict mapping dataset to dict containing train, val, and test sets.
    """
    data_dir = Path(data_dir)
    result = {}

    text_field = Field(sequential=True, use_vocab=False, tokenize=tokenizer.encode, batch_first=True, include_lengths=True,
                        pad_token=tokenizer.pad_token_id, unk_token=tokenizer.unk_token_id)
    label_field = LabelField(use_vocab=False)
    data_fields = [("label", label_field), ("text", text_field)]
    for idx, dataset in enumerate(datasets):
        print(f'processing dataset: {dataset}')
        train_file = dataset+".task.train"
        test_file = dataset+".task.test"
        train_val_set, test_set, = TabularDataset.splits(path=data_dir,
                                                            train=train_file,
                                                            test=test_file,
                                                            fields=data_fields, skip_header=True, format="tsv")
        # tabulardataset reads it in as a string
        for example in train_val_set.examples:
            example.label = int(example.label)
        for example in test_set.examples:
            example.label = int(example.label)
        train_examples, val_examples = train_test_split(train_val_set.examples, test_size=validation_size,
                                                        random_state=random_state)
        train_set, val_set = Dataset(train_examples, data_fields), Dataset(val_examples, data_fields)
        result[dataset] = {'train': train_set, 'val': val_set, 'test': test_set}

    return result
    