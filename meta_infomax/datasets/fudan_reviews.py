from pathlib import Path
from sklearn.model_selection import train_test_split
from torchtext.data import Field, LabelField, TabularDataset, Dataset
from typing import List, Dict, Union
import shutil
import urllib.request
import tarfile
import os
import logging
import tqdm
import torch    
from torch.utils.data import Dataset, DataLoader, TensorDataset, SequentialSampler,\
        RandomSampler, BatchSampler
import pandas as pd
pd.options.mode.chained_assignment = None # ignore annoying pandas warnings

DATASETS = ['apparel', 'baby', 'books', 'camera_photo', 'electronics',
            'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines',
            'music', 'software', 'sports_outdoors', 'toys_games', 'video']
    
class SingleTaskDataset(Dataset):
    def __init__(self, data):
        """Simple wrapper dataset around a DataFrame."""
        self.data = data
        
    def __getitem__(self, idx):
        return self.data.loc[idx, :].to_dict()
    
    def __len__(self):
        return len(self.data)


class MultiTaskDataset(Dataset):
    def __init__(self, tokenizer, data_dir='data/mtl-dataset/', split: str='train', collapse_domains: bool=True,
                 keep_datasets: List[str] = DATASETS, validation_size: float = 0.2, min_count: int = 0,
                 random_state: int = 42, load: bool=True, save: bool=True):
        """
        Dataset class for multi task learning.
        
        Reads data from Fudan review dataset, containing data for sentiment classification in different
        domains.
        
        Parameters
        ---
        tokenizer: huggingface bert tokenizer
        data_dir: Union[str, Path]
            Directory for Fudan valdata.
        split: str in {'train', 'val', 'test', 'all'}
            Indicate which datasplit we want. This is not a split over domain, but over samples.
        collapse_domains: bool. Not used at the moment, since we have methods to return individual domains anyway.
            If True, make one big iterator from all datasets. This means that different domains will be present
            in the same batch.
        keep_datasets: List[str]
            Which domains to keep in memory. Specify domain such as ['apparel', 'baby'].
            By default, all known datasets (found in DATASETS) are read in and tokenized, so that
            we can save them and not have to tokenize again. We then filter on keep_datasets.
        validation_size: should be in [0, 1].
            Fraction of validation samples in train set.
        min_count: NOT IMPLEMENTED
            Token should appear min_count times to be counted in vocabulary.
        random_state:
            Used for validation split.
        load:
            Flag indicating whether to load set from a file if it exists.
            TODO: implement different filename for different datasets loaded.
        save:
            Whether to save processed data to a file. Currently always saves to data_dir / 'processed_data.pt'
        """
        assert split in ('train', 'val', 'test', 'all'), 'provide correct data split'
        self.data_dir = Path(data_dir)
        self.split = split
        self.collapse_domains = collapse_domains
        self.keep_datasets = keep_datasets
        self.random_state = random_state
        # load and process data
        store_processed = self.data_dir / (self.split + f'_random-{random_state}' + f'_valsize-{validation_size}' + '_processed_data.pt')
        if store_processed.exists() and load:
            logging.info(f'loading data from file: {store_processed}')
            self.data = torch.load(store_processed)
        else:
            self.data = self._read_datasets(validation_size=validation_size)
            self.data['tokenized'] = self._tokenize_data(self.data['text'], tokenizer)
            if save:
                torch.save(self.data, store_processed)
        # filter rows with domain in keep_datasets
        self.data = self.data.loc[self.data['domain'].isin(keep_datasets), :].reset_index(drop=True)
        self.collator = MultiTaskCollator(tokenizer)
        
    def _read_datasets(self, validation_size=0.2):
        """
        Read datasets from file. If data directory does not exist, downloads data.
        
        Parameters
        ----------
        validation_size: float in [0, 1]
        
        Returns
        ---
        pd.DataFrame
            Appropriate datasplit with fields 'label', 'text', and 'domain'.
        """
        dfs = []
        col_names=['label', 'text']
        if not self.data_dir.exists():
            download_and_extract_fudan(self.data_dir)
        for idx, dataset in enumerate(DATASETS):
            logging.info(f'processing dataset: {dataset}')
            train_set, val_set, test_set = None, None, None
            if self.split in ('train', 'val', 'all'):
                train_file = dataset + '.task.train'
                train_val_set = pd.read_csv(self.data_dir / train_file, sep='\t', header=None, names=col_names)
                if validation_size == 0: # only do split when validation_size > 0
                    train_set = train_val_set
                else:
                    train_set, val_set = train_test_split(train_val_set, test_size=validation_size,
                                                                random_state=self.random_state)
                    val_set['domain'] = dataset
                train_set['domain'] = dataset # record which domain it is in dataframe
            elif self.split in ('test', 'all'):
                test_file = dataset + '.task.test'
                test_set = pd.read_csv(self.data_dir / test_file, sep='\t', header=None, names=col_names)
                test_set['domain'] = dataset
            if self.split == 'all':
                dfs.extend([train_set, val_set, test_set])
            else:
                dfs.append({'train': train_set, 'val': val_set, 'test': test_set}[self.split])
        return pd.concat(dfs, ignore_index=True).dropna().reset_index(drop=True) # ignore nan values
    
    def _tokenize_data(self, texts, tokenizer):
        """
        Tokenize and map to int (aka encode) iterable of texts with tokenizer.
        WITHOUT adding special tokens.
        
        Parameters
        ---
        texts: Iterable[str]
        tokenizer: huggingface tokenizer
        
        Returns
        ---
        Iterable[List[int]]
        """
        # surpress >512 tokens tokenization warning
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
        logging.info('tokenizing data..')
        tokenized = []
        for text in tqdm.tqdm(texts):
            tokenized.append(
                tokenizer.encode(text, add_special_tokens=False)
            )
        # set back to original logging
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.WARNING)
        return tokenized
    
    def get_domain(self, domain):
        """Get all the data for a single domain.
        
        Parameters
        ---
        domain : str
            Domain of which we want to get the data.
        
        Returns
        ---
        SingleTaskDataset for the given domain
        """
        assert domain in self.keep_datasets, 'ensure domain is present in data'
        return SingleTaskDataset(self.data[self.data.domain == domain].reset_index(drop=True))
    
    def domain_datasets(self, domains):
        """
        Create datasets for each domain given in domains.
        
        Parameters
        ---
        domains : List[str]
            Domains for which we want datasets.
            
        Returns
        ---
        Dict[str, SingleTaskDataset]
        """
        result = {}
        for domain in domains:
            assert domain in self.keep_datasets, 'make sure domain is in available domains.'
            result[domain] = self.get_domain(domain)
        return result
            
    def domain_dataloaders(self, domains=None, **kwargs):
        """
        Create dataloaders for each domain given in domains.
        
        Parameters
        ---
        domains : List[str]
            Domains for which we want datasets. If not provided, default to all datasets provided in
            keep_datasets.
        kwargs: keyword arguments for DataLoader.
            
        Returns
        ---
        Dict[str, DataLoader]
        """
        result = {}
        if domains is None:
            domains = self.keep_datasets
        domain_datasets = self.domain_datasets(domains)
        for domain in domains:
            assert domain in self.keep_datasets, 'make sure domain is in available domains.'
            result[domain] = DataLoader(domain_datasets[domain], **kwargs)
        return result
    
    def __getitem__(self, idx):
        return self.data.loc[idx, :].to_dict()
    
    def __len__(self):
        return len(self.data)

class MultiTaskCollator:
    def __init__(self, tokenizer):
        """
        Class to pass to Dataloader collate_fn argument. Dataloader calls __call__ method.
        
        Parameters
        ---
        tokenizer: transformer tokenizer supporting prepare_for_model method.
        """
        self.tokenizer = tokenizer

    def __call__(self, inputs):
        """
        Function to pass into Dataloader collate_fn argument.
        
        Parameters
        ---
        inputs: List[Dict[str, info]]
            Input dicts should contain fields 'label', 'domain', and 'tokenized'.

        Returns
        ---
        Dict[str, Tensor], with fields 'x', 'masks', 'labels', 'domains'.
        Masks is fed to the BERT model to know which tokens to mask.
        """
        labels = [x['label'] for x in inputs]
        domains = [x['domain'] for x in inputs]
        tokenizeds = [x['tokenized'] for x in inputs]

        # will be fed to model
        input_ids = []
        attention_masks = []

        # calculate max batch len for batching
        max_batch_len = max(len(x) for x in tokenizeds)
        max_batch_len = min(max_batch_len, 512) # 512 is our absolute max

        for tokenized_sequence in tokenizeds:
            input_dict = self.tokenizer.prepare_for_model(tokenized_sequence,
                                                     max_length=max_batch_len, pad_to_max_length=True)

            input_ids.append(input_dict['input_ids'])
            attention_masks.append(input_dict['attention_mask'])
        return {'x': torch.tensor(input_ids), 'masks': torch.tensor(attention_masks), 'labels': torch.tensor(labels),
               'domains': domains}

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
