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
BEST_MODEL_FNAME = "best-model.pt"


class MultitaskTrainer():
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

        self.checkpoint_dir = CHECKPOINTS / config['exp_name']
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.exp_dir = RESULTS / config['exp_name']
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.exp_dir / LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        bert, tokenizer, embedding_dim = utils.get_transformer(config['transformer_name'])

        data = fudan_reviews.get_fudan_datasets(tokenizer, data_dir=config['data_dir'])  
        thrown_away = utils.remove_outlier_lengths(data) # call only once!

        print("Data Summary")
        for domain in config['domains']:
            summary = f"{domain} \t\t\t Train: {len(data[domain]['train'])} Val: {len(data[domain]['val'])} Test: {len(data[domain]['test'])}"
            print(summary)

        self.train_iter = utils.get_iterators(data, include_domains=config['train_domains'], device=config['device'],
                                              split='train', batch_size=config['batch_size'], collapse_domains=True)
        self.valid_iter = utils.get_iterators(data, include_domains=config['valid_domains'], device=config['device'],
                                              split='val', batch_size=config['batch_size']*2, collapse_domains=True)
        self.test_iter = utils.get_iterators(data, include_domains=config['test_domains'], device=config['device'],
                                              split='test', batch_size=config['batch_size']*2, collapse_domains=True)

        ffn = FeedForward(768, 3, [512, 256, 2], nn.ReLU())
        self.model = SentimentClassifier(bert, ffn)
        print(f"Using device: {config['device']}")
        self.model.to(config['device'])

        self.model.encoder_unfreeze_layers(layers=(10, 11))
        self.ffn_opt = optim.Adam(ffn.parameters())
        self.bert_opt = optim.AdamW(bert.parameters(), lr=2e-5, correct_bias=False)
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
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch

            for i, batch in enumerate(self.train_iter['all']):
                self.current_iter += 1
                results = self._batch_iteration(batch, training=True)
                
                self.writer.add_scalar('Accuracy/Train', results['accuracy'], self.current_iter)
                self.writer.add_scalar('Loss/Train', results['loss'], self.current_iter)
                print(f"EPOCH:{epoch} STEP:{i}\t Accuracy: {results['accuracy']:.3f} Loss: {results['loss']:.3f}")

                if i % self.config['valid_freq'] == 0:
                    self.validate()
                if i % self.config['save_freq'] == 0:
                    self.save_checkpoint()
    
    def validate(self):
        """ Main validation loop """
        losses = []
        accuracies = []

        print("Begin evaluation over validation set")
        with torch.no_grad():
            for i, batch in enumerate(self.valid_iter['all']):
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
        print(report)

    def _batch_iteration(self, batch: tuple, training: bool):
        """ Iterate over one batch """

        # send tensors to model device
        (text, text_len), label = batch.text, batch.label
        text = text.to(self.config['device'])
        label = label.to(self.config['device'])

        if training:
            self.bert_opt.zero_grad()
            self.ffn_opt.zero_grad()
            output = self.model(text, label)
            logits = output['logits']
            loss = output['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.bert_opt.step()
            self.bert_scheduler.step()
            self.ffn_opt.step()
        else:
            with torch.no_grad():
                output = self.model(text, label)
                logits = output['logits']
                loss = output['loss']

        acc = (logits.argmax(dim=1) == label).float().mean().item()
        results = {'accuracy': acc, 'loss': loss.item()}
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
