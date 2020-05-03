import torch
import pandas as pd
from typing import List
from pathlib import Path
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class FudanMetaDataset(Dataset):
    ''' Fudan Reviews pytorch dataset for meta-learning '''

    def __init__(self, domain_list: List[str], data_dir: Path, tokenizer: PreTrainedTokenizer, is_train: bool=True):
        ''' Initialize the fudan review dataset

        Parameters
        ---
        domain_list:
        list of domains to be included

        data_dir: 
        path to the data directory

        tokenizer:
        PreTrainedTokenizer from one of the transformer models

        is_train: 
        is this for train dataset or test dataset?
        '''
        super().__init__()
        self.is_train = is_train
        self.data_dir = data_dir    
        self.domain_list = domain_list

        df = self._prepare_df()
        batch = tokenizer.batch_encode_plus(df['text'].tolist(),
                                            max_length=tokenizer.max_len,
                                            pad_to_max_length=True,
                                            return_tensors='pt', 
                                            return_attention_masks=True)
        self.x = batch['input_ids']
        self.attn_mask = batch['attention_mask']
        self.domains = torch.tensor(df['domain'].tolist())
        self.y = torch.tensor(df['label'].tolist())
    
    def __getitem__(self, idx):
        return self.x[idx], self.attn_mask[idx], self.y[idx]
    
    def __len__(self):
        return len(self.x)

    def _prepare_df(self):
        col_names=['label', 'text']
        df = pd.DataFrame(columns=['text', 'domain', 'label'])
        for idx, domain in enumerate(self.domain_list):
            file = domain+".task.train" if self.is_train else domain+".task.test" 
            domain_df = pd.read_csv(self.data_dir / file, sep="\t", header=None, names=col_names)
            domain_df["domain"] = idx
            df = df.append(domain_df, sort=False, ignore_index=True)
        return df