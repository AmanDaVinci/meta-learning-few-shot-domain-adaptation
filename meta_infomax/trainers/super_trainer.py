from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from typing import Dict

from meta_infomax.datasets import utils, fudan_reviews
from meta_infomax.models.feed_forward import FeedForward
from meta_infomax.models.sentiment_classifier import SentimentClassifier

RESULTS = Path("results")
CHECKPOINTS = Path("checkpoints")
LOG_DIR = Path("logs")


class TrainerParent():
    """Train to classify sentiment across different domains/tasks"""

    def __init__(self, config: Dict, collapse_domains = True):
        """Initialize the trainer with data, models and optimizers

        Parameters
        ---
        config:
            dictionary of configurations with the following keys:
            {
                'exp_name': "multitask_test",
                'epochs': 10,
                'batch_size': 64,
                'valid_freq': 50, 
                'save_freq': 100,
                'device': 'cpu',
                'data_dir': 'data/mtl-dataset/',
                'transformer_name': "bert-base-uncased",
                'domains': ['apparel', 'baby', 'books', 'camera_photo', 'electronics',
                            'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines',
                            'music', 'software', 'sports_outdoors', 'toys_games', 'video'], 
                'train_domains': ['apparel', 'baby', 'books', 'camera_photo', 'health_personal_care',
                                  'imdb', 'kitchen_housewares', 'magazines', 'sports_outdoors', 'toys_games'], 
                'valid_domains': ['software', 'electronics'], 
                'test_domains': ['music', 'video'],
            }
        """
        self.config = config

        self.checkpoint_dir = CHECKPOINTS / config['exp_name']
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.exp_dir = RESULTS / config['exp_name']
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.exp_dir / LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.BEST_MODEL_FNAME = "best-model.pt"


        bert, tokenizer, embedding_dim = utils.get_transformer(config['transformer_name'])

        data = fudan_reviews.get_fudan_datasets(tokenizer, data_dir=config['data_dir'])  
        ###thrown_away = utils.remove_outlier_lengths(data) # call only once!

        print("Data Summary")
        for domain in config['domains']:
            summary = f"{domain} \t\t\t Train: {len(data[domain]['train'])} Val: {len(data[domain]['val'])} Test: {len(data[domain]['test'])}"
            print(summary)

        self.train_iter = utils.get_iterators(data, include_domains=config['train_domains'], device=config['device'],
                                              split='train', batch_size=config['batch_size'], collapse_domains=collapse_domains)
        self.valid_iter = utils.get_iterators(data, include_domains=config['valid_domains'], device=config['device'],
                                              split='val', batch_size=config['batch_size']*2, collapse_domains=collapse_domains)
        self.test_iter = utils.get_iterators(data, include_domains=config['test_domains'], device=config['device'],
                                              split='test', batch_size=config['batch_size']*2, collapse_domains=collapse_domains)

        ffn = FeedForward(768, 3, [512, 256, 2], nn.ReLU())
        self.model = SentimentClassifier(bert, ffn)
        print(f"Using device: {config['device']}")
        self.model.to(config['device'])

        self.model.encoder_unfreeze_layers(layers=(10, 11))
        self.ffn_opt = optim.Adam(ffn.parameters())
        self.bert_opt = optim.AdamW(bert.parameters(), lr=2e-5)
        self.bert_scheduler = get_linear_schedule_with_warmup(self.bert_opt,
                                                              num_warmup_steps=len(self.train_iter)/10,
                                                              num_training_steps=len(self.train_iter))

        # Init trackers
        self.current_iter = 0
        self.current_epoch = 0
        self.best_accuracy = 0.

    def run(self):
        """ Run the train-eval loop
        
        If the loop is interrupted manually, finalization will still be executed
        """
        try:
            print(f"Begin training for {self.config['epochs']} epochs")
            self.train()
        except KeyboardInterrupt:
            print("Manual interruption registered. Please wait to finalize...")
            self.save_checkpoint()

    def train(self):
        """ Main training loop """
        print("implement class specific training main")
    
    def validate(self):
        """ Main validation loop """
        print("implement class specific validation")

    def _batch_iteration(self, batch: tuple, training: bool):
        """ Iterate over one batch """

        print("implement class specific batch iteration")

    def save_checkpoint(self, file_name: str = None):
        """Save checkpoint in the checkpoint directory.

        Checkpoint directory and checkpoint file need to be specified in the configs.

        Parameters
        ----------
        file_name: str
            Name of the checkpoint file.
        """
        if file_name is None:
            file_name = f"Epoch[{self.current_epoch}]-Step[{self.current_iter}].pt"

        file_name = self.checkpoint_dir / file_name
        state = {
            'epoch': self.current_epoch,
            'iter': self.current_iter,
            'best_accuracy': self.best_accuracy,
            'model_state': self.model.state_dict(),
        }
        torch.save(state, file_name)
        print(f"Checkpoint saved @ {file_name}")
