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
from allennlp.data.iterators import BucketIterator
from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
import pandas as pd

from sklearn.model_selection import train_test_split

DATA_DIR = Path("../data/mtl-dataset")
DATASETS = ['apparel', 'baby', 'books', 'camera_photo',  'electronics', 
      'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines', 
      'music', 'software', 'sports_outdoors', 'toys_games', 'video']


def get_fudan_datasets(datasets: List[str]=DATASETS, data_dir: Union[str, Path]=DATA_DIR,
                       validation_size: float=0.2, min_count: int=0,
                       random_state: int=11) -> Tuple[Dict[str, Dict[str, List[Instance]]], Vocabulary]:
    """
    Get all fudan datasets specified in a dictionary. Also return vocabulary.
    Gets all train, val, and test tests, with corresponding keys in the returned dict for each dataset.

    Vocab is created from ALL training datasets combined.

    Parameters
    ----------
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

    reader = FudanReviewDatasetReader()

    for idx, dataset in enumerate(datasets):
        train_file = dataset+".task.train"
        test_file = dataset+".task.test"
        train_val_set = reader.read(data_dir / train_file)
        test_set = reader.read(data_dir / test_file)
        train_set, validation_set = train_test_split(train_val_set, test_size=validation_size, random_state=random_state)
        result[dataset] = {'train': train_set, 'val': validation_set, 'test': test_set}

    vocab = Vocabulary.from_instances([example for splits in result.values()
                                    for example in splits['train']],
                                    min_count={'tokens': min_count})
    return result, vocab


# @DatasetReader.register("fudan_review") # allows us to specify reader using JSON
class FudanReviewDatasetReader(DatasetReader):
    """
    DatasetReader for Fudan Review data, one sentence per line, like

        <sentiment_label in (0,1)><tab><sentence>
        
    No need to tokenize sentence, since that's already been done.
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer(lowercase_tokens=False)}
    
    def text_to_instance(self, tokens: List[Token], label: int = None) -> Instance:
        text_field = TextField(tokens, self.token_indexers)
        fields = {'text': text_field}

        if label is not None:
            # our labels are already 0-indexed ints -> skip indexing
            label_field = LabelField(label=label, skip_indexing=True)
            fields['label'] = label_field

        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        df = pd.read_csv(file_path, sep="\t", header=None, names=['label', 'text'])
        for i, row in enumerate(df.itertuples()):
            label, text = row.label, row.text
            # our examples are already tokenized, so just use split
            try:
                yield self.text_to_instance([Token(word) for word in text.split()], label)
            except AttributeError as e:
                print(e, f'skipping row {i}')



