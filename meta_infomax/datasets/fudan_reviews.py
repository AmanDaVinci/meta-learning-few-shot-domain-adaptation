from pathlib import Path
from sklearn.model_selection import train_test_split
from torchtext.data import Field, LabelField, TabularDataset, Dataset
from typing import List, Dict, Union
import shutil
import urllib.request
import tarfile
import os
from pathlib import Path
import logging

DATASETS = ['apparel', 'baby', 'books', 'camera_photo', 'electronics',
            'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines',
            'music', 'software', 'sports_outdoors', 'toys_games', 'video']


def get_fudan_datasets(tokenizer, data_dir: Union[str, Path], datasets: List[str] = DATASETS,
                       validation_size: float = 0.2, min_count: int = 0,
                       random_state: int = 11) -> Dict[str, Dict[str, List[Dataset]]]:
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

    if not os.path.exists(data_dir):
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        download_and_extract_fudan(data_dir.parent)

    result = {}

    truncated_encode = lambda x: tokenizer.encode(x, max_length=512)
    text_field = Field(sequential=True, use_vocab=False, tokenize=truncated_encode, batch_first=True, include_lengths=True,
                       pad_token=tokenizer.pad_token_id, unk_token=tokenizer.unk_token_id)
    label_field = LabelField(use_vocab=False)
    data_fields = [("label", label_field), ("text", text_field)]
    for idx, dataset in enumerate(datasets):
        print(f'processing dataset: {dataset}')
        train_file = dataset + ".task.train"
        test_file = dataset + ".task.test"
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


def download_and_extract_fudan(data_dir):
    """
    Downloads and extracts fudan data

    Parameters
    ----------
    data_dir:
        Directory of fudan datasets.
    """
    logging.info(f'Downloading data to {data_dir}..')
    data_dir = Path(data_dir)
    if data_dir.exists():
        logging.info('data directory already exists, stopping download.')
        return
    data_dir.mkdir(parents=True)
    data_url = "https://github.com/FrankWork/fudan_mtl_reviews/raw/master/data/fudan-mtl-dataset.tar.gz"

    tar_filename = data_dir / "fudan-mtl-dataset.tar.gz"
    with urllib.request.urlopen(data_url) as response, open(tar_filename, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
    with tarfile.open(tar_filename, 'r:gz') as tar_file:
        tar_file.extractall(data_dir)
    os.remove(tar_filename)
    for path in (data_dir / 'mtl-dataset').glob('*'):
        shutil.move(path.as_posix(), data_dir.as_posix())
    (data_dir / 'mtl-dataset').rmdir()

