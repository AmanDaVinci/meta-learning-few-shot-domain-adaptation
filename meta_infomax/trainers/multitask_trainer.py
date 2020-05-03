from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW
from typing import Dict
import logging
from tqdm import tqdm

from meta_infomax.models.feed_forward import FeedForward
from meta_infomax.models.sentiment_classifier import SentimentClassifier
from meta_infomax.datasets import utils
from meta_infomax.datasets.fudan_reviews import MultiTaskDataset


RESULTS = Path("results")
CHECKPOINTS = Path("checkpoints")
LOG_DIR = Path("logs")
BEST_MODEL_FNAME = "best-model.pt"


class MultitaskTrainer():
    """Train to classify sentiment across different domains/tasks"""

    def __init__(self, config: Dict):
        """Initialize the trainer with data, models and optimizers

        Parameters
        ---
        config:
            dictionary of configurations with the following keys: (TODO: and more)
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
        utils.init_logging(log_path=self.log_dir, log_level=config['log_level']) # initialize logging
        logging.info(f'Config Setting:\n{config}')

        # TODO: set dropout
        bert, tokenizer, embedding_dim = utils.get_transformer(config['transformer_name'])

        # for now, we say that the training data, is the train split of every train domain
        # we could eventually also include the test split of the train_domain
        train_data = MultiTaskDataset(tokenizer=tokenizer, data_dir=config['data_dir'], split='train',
                        keep_datasets=config['train_domains'],
                        random_state=config['random_state'], validation_size=0)
        val_data = MultiTaskDataset(tokenizer=tokenizer, data_dir=config['data_dir'], split='train',
                        keep_datasets=config['val_domains'],
                        random_state=config['random_state'], validation_size=0)
        test_data = MultiTaskDataset(tokenizer=tokenizer, data_dir=config['data_dir'], split='train',
                        keep_datasets=config['test_domains'],
                        random_state=config['random_state'], validation_size=0)

        # logging.info('Data summary\n' + '-' * 12)
        # for domain in config['domains']:
        #     summary = f"{domain} -- Train: {len(train_data.get_domain(domain))} Val: {len(val_data.get_domain(domain))} Test: {len(test_data.get_domain(domain))}"
        #     logging.info(summary)

        if config['collapse_domains']:
            self.train_loader = DataLoader(train_data, batch_size=config['batch_size'],
                                           collate_fn=train_data.collator, shuffle=True)
            self.val_loader = DataLoader(val_data, batch_size=config['batch_size'],
                                           collate_fn=train_data.collator, shuffle=False)
            self.test_loader = DataLoader(test_data, batch_size=config['batch_size'],
                                           collate_fn=train_data.collator, shuffle=False)
        else:
            # loaders are now dicts mapping from domains to individual loaders
            self.train_loader = train_data.domain_dataloaders(batch_size=config['batch_size'], collate_fn=train_data.collator,
                                                            shuffle=True)
            self.val_loader = val_data.domain_dataloaders(batch_size=config['batch_size'], collate_fn=val_data.collator,
                                                            shuffle=False)
            self.test_loader = test_data.domain_dataloaders(batch_size=config['batch_size'], collate_fn=test_data.collator,
                                                            shuffle=False)

        # TODO: parameterize feedforward from config
        # TODO: initialize with sampling from a normal distribution with mean 0 and standard deviation 0.02
        ffn = FeedForward(768, 3, [512, 256, 2], activations=nn.ReLU())
        self.model = SentimentClassifier(bert, ffn)
        logging.info(f"Using device: {config['device']}")
        self.model.to(config['device'])

        self.model.encoder_unfreeze_layers(layers=config['unfreeze_layers'])
        self.ffn_opt = optim.Adam(ffn.parameters())
        # self.bert_opt = optim.AdamW(bert.parameters(), lr=2e-5, correct_bias=False)
        self.bert_opt = AdamW(bert.parameters(), lr=config['lr'], correct_bias=False,
                            weight_decay=config['weight_decay']) # use transformers AdamW
        self.bert_scheduler = get_linear_schedule_with_warmup(self.bert_opt,
                                                              num_warmup_steps=config['warmup_steps'],
                                                              num_training_steps=len(self.train_loader) *
                                                              config['epochs'])

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
            print("Manual interruption registered. Please wait to finalize...")
            self.save_checkpoint()

    def train(self):
        """Main training loop."""
        assert self.config['collapse_domains'] == True, 'only implemented for collapse_domains=True'
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(self.train_loader))
        logging.info("  Num Epochs = %d", self.config['epochs'])
        logging.info("  Batch size = %d", self.config['batch_size'])
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch

            for i, batch in enumerate(tqdm(self.train_loader)):
                self.current_iter += 1
                results = self._batch_iteration(batch, training=True)
                
                # TODO: also write to csv file every log_freq steps
                self.writer.add_scalar('Accuracy/Train', results['accuracy'], self.current_iter)
                self.writer.add_scalar('Loss/Train', results['loss'], self.current_iter)
                # TODO: only every log_freq steps
                logging.info(f"EPOCH:{epoch} STEP:{i}\t Accuracy: {results['accuracy']:.3f} Loss: {results['loss']:.3f}")

                if self.current_iter % self.config['valid_freq'] == 0:
                    self.validate()

    
    def validate(self):
        """ Main validation loop """
        losses = []
        accuracies = []

        print("Begin evaluation over validation set")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader)):
                results = self._batch_iteration(batch, training=False)
                self.writer.add_scalar('Accuracy/Valid', results['accuracy'], self.current_iter)
                self.writer.add_scalar('Loss/Valid', results['loss'], self.current_iter)
                losses.append(results['loss'])
                accuracies.append(results['accuracy'])
            
        mean_accuracy = np.mean(accuracies)
        if mean_accuracy > self.best_accuracy:
            self.best_accuracy = mean_accuracy
            self.save_checkpoint(BEST_MODEL_FNAME)
        
        report = (f"[Validation]\t"
                  f"Accuracy: {mean_accuracy:.3f} "
                  f"Total Loss: {np.mean(losses):.3f}")
        logging.info(report)

    def _batch_iteration(self, batch: tuple, training: bool):
        """ Iterate over one batch """

        # send tensors to model device
        x, masks, labels, domains = batch['x'], batch['masks'], batch['labels'], batch['domains']
        x = x.to(self.config['device'])
        masks = masks.to(self.config['device'])
        labels = labels.to(self.config['device'])

        if training:
            self.bert_opt.zero_grad()
            self.ffn_opt.zero_grad()
            output = self.model(x=x, masks=masks, labels=labels, domains=domains) # domains is ignored for now
            logits = output['logits']
            loss = output['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad_norm'])
            self.bert_opt.step()
            self.bert_scheduler.step()
            self.ffn_opt.step()
        else:
            with torch.no_grad():
                output = self.model(x=x, masks=masks, labels=labels, domains=domains) # domains is ignored for now
                logits = output['logits']
                loss = output['loss']

        results = {'accuracy': output['acc'], 'loss': loss.item()}
        return results

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
