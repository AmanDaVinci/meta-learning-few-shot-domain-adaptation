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


from meta_infomax.trainers.super_trainer import BaseTrainer


class MultitaskTrainer(BaseTrainer):
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
        super().__init__(config)

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
            self.save_checkpoint(self.BEST_MODEL_FNAME)
        
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
