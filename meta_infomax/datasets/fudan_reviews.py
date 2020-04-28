from pathlib import Path
import pandas as pd
from torchtext.data import Field, TabularDataset, BucketIterator
from torchtext.vocab import GloVe

from typing import Iterator, List, Dict
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

torch.manual_seed(1)
%load_ext autoreload
%autoreload 2

DATA_DIR = Path("../data/mtl-dataset")
GLOVE_DIR = Path("../data")
DATASETS = ['apparel', 'baby', 'books', 'camera_photo',  'electronics', 
      'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines', 
      'music', 'software', 'sports_outdoors', 'toys_games', 'video']

def prepare_data(dataset_list: list=DATASETS, data_dir: Path=DATA_DIR,
                 devset_fraction: float=0.2) -> None:
    col_names=['label', 'text']
    train_dfs = []
    test_dfs = []
    for idx, dataset in enumerate(dataset_list):
        train_file = dataset+".task.train"
        test_file = dataset+".task.test"
        train_df = pd.read_csv(data_dir / train_file, sep="\t", header=None, names=col_names)
        test_df = pd.read_csv(data_dir / test_file, sep="\t", header=None, names=col_names)
        train_df['task_id'] = idx
        train_df['task_name'] = dataset
        test_df['task_id'] = idx
        test_df['task_name'] = dataset
        train_dfs.append(train_df)
        test_dfs.append(test_df)
    # concatenate then shuffle in place
    df = pd.concat(train_dfs)
    train_dfs = df.sample(frac=1-devset_fraction).reset_index(drop=True)
    dev_dfs = df.sample(frac=devset_fraction).reset_index(drop=True)
    test_dfs = pd.concat(test_dfs).sample(frac=1).reset_index(drop=True)
    train_dfs.to_csv(DATA_DIR / "train.csv", index=False)
    dev_dfs.to_csv(DATA_DIR / "dev.csv", index=False)
    test_dfs.to_csv(DATA_DIR / "test.csv", index=False)

def get_data(train_file: str="train.csv", dev_file: str="dev.csv", test_file: str="test.csv",
             tokenizer: any=lambda x: x.split(), include_lengths: bool=True):
    text_field = Field(sequential=True, tokenize=tokenizer, lower=True, batch_first=True, include_lengths=True)
    label_field = Field(sequential=False, use_vocab=False)
    task_field = Field(sequential=False, use_vocab=False)
    data_fields = [("label", label_field), ("text", text_field), ("task", task_field), ("task_name", None)]
    train_set, dev_set, test_set = TabularDataset.splits(path=DATA_DIR, root=DATA_DIR,
                                                        train=train_file, validation=dev_file, test=test_file,
                                                        fields=data_fields, skip_header=True, format="csv")
    text_field.build_vocab(train_set, vectors=GloVe(cache=GLOVE_DIR))
    return train_set, dev_set, test_set, text_field.vocab

@DatasetReader.register("fudan_review") # allows us to specify reader using JSON
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
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {'sentence': sentence_field}

        if label is not None:
            # our labels are already 0-indexed ints -> skip indexing
            label_field = LabelField(label=label, skip_indexing=True)
            fields['label'] = label_field

        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        with open(file_path) as f:
            for line in f:
                label, sentence = line.strip(' ').split('\t')
                yield self.text_to_instance([Token(word) for word in sentence.split()], int(label))
