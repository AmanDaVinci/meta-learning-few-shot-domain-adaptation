import logging
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW
from typing import Dict

from meta_infomax.datasets import utils
from meta_infomax.models.feed_forward import FeedForward
from meta_infomax.models.sentiment_classifier import SentimentClassifier

RESULTS = Path("results")
CHECKPOINTS = Path("checkpoints")
LOG_DIR = Path("logs")


class BaseTrainer:
    """Train to classify sentiment across different domains/tasks"""

    def __init__(self, config: Dict):
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
        self.BEST_MODEL_FNAME = "best-model.pt"

        self.checkpoint_dir = CHECKPOINTS / config['exp_name']
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.exp_dir = RESULTS / config['exp_name']
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.exp_dir / LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        utils.init_logging(log_path=self.log_dir, log_level=config['log_level'])  # initialize logging
        logging.info(f'Config Setting:\n{config}')

        # TODO: set dropout
        bert, self.tokenizer, embedding_dim = utils.get_transformer(config['transformer_name'])

        # TODO: parameterize feedforward from config
        # TODO: initialize with sampling from a normal distribution with mean 0 and standard deviation 0.02
        ffn = FeedForward(embedding_dim, 3, [512, 256, 2], activations=nn.ReLU())
        self.model = SentimentClassifier(bert, ffn)
        logging.info(f"Using device: {config['device']}")
        self.model.to(config['device'])

        self.model.encoder_unfreeze_layers(layers=config['unfreeze_layers'])
        self.ffn_opt = optim.Adam(ffn.parameters())
        self.bert_opt = AdamW(bert.parameters(), lr=config['lr'], correct_bias=False, weight_decay=config['weight_decay'])

        # Init trackers
        self.current_iter = 0
        self.current_epoch = 0
        self.best_accuracy = 0.

    def run(self):
        """ Run the train-eval loop
        
        If the loop is interrupted manually, finalization will still be executed
        """
        try:
            logging.info(f"Begin training for {self.config['epochs']} epochs")
            self.train()
        except KeyboardInterrupt:
            logging.info("Manual interruption registered. Please wait to finalize...")
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
            # TODO: Save optimizer states?
            'epoch': self.current_epoch,
            'iter': self.current_iter,
            'best_accuracy': self.best_accuracy,
            'model_state': self.model.state_dict(),
        }
        torch.save(state, file_name)
        logging.info(f"Checkpoint saved @ {file_name}")

    def load_checkpoint(self, experiment_name: str, file_name: str = None):
        if file_name is None:
            file_name = self.BEST_MODEL_FNAME

        try:
            file_name = CHECKPOINTS / experiment_name / file_name
            logging.info(f"Loading checkpoint...")
            checkpoint = torch.load(file_name, map_location=self.config["device"])

            self.current_epoch = checkpoint['epoch']
            self.current_iter = checkpoint['iter']
            self.best_accuracy = checkpoint['best_accuracy']
            self.model.load_state_dict(checkpoint['model_state'])
            logging.info(f"Checkpoint loaded successfully from '{file_name}'\n")

        except OSError:
            logging.info(f"No checkpoint exists @ {self.checkpoint_dir}")
            logging.info("**Training for the first time**")
