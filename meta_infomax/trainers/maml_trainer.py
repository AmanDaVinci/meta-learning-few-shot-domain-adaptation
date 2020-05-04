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
from meta_infomax.datasets.fudan_reviews import MultiTaskDataset


from meta_infomax.trainers.super_trainer import BaseTrainer
from meta_infomax.datasets.utils import sample_domains

from copy import deepcopy

class MAMLTrainer(BaseTrainer):
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

        # for now, we say that the training data, is the train split of every train domain
        # we could eventually also include the test split of the train_domain
        train_data = MultiTaskDataset(tokenizer=self.tokenizer, data_dir=config['data_dir'], split='train',
                        keep_datasets=config['train_domains'],
                        random_state=config['random_state'], validation_size=0)
        val_data = MultiTaskDataset(tokenizer=self.tokenizer, data_dir=config['data_dir'], split='train',
                        keep_datasets=config['val_domains'],
                        random_state=config['random_state'], validation_size=0)
        test_data = MultiTaskDataset(tokenizer=self.tokenizer, data_dir=config['data_dir'], split='train',
                        keep_datasets=config['test_domains'],
                        random_state=config['random_state'], validation_size=0)

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

        self.bert_scheduler = get_linear_schedule_with_warmup(self.bert_opt,
                                                              num_warmup_steps=config['warmup_steps'],
                                                              num_training_steps=len(self.train_loader) *
                                                              config['epochs'])
        self.current_episode = 0

    def train(self):
        """Main training loop."""
        assert self.config['collapse_domains'] == False, 'only implemented for collapse_domains=False'
        logging.info("***** Running training *****")
        logging.info("  Num examples = %d", len(self.train_loader))
        logging.info("  Num Episodes = %d", self.config['episodes'])
        logging.info("  K-shot = %d", self.config['k_shot_num'])
        logging.info("  N-way = %d", self.config['n_domains'])

        for episode in range(self.current_episode, self.config['episodes']):
            self.current_episode = episode
            episode_domains = sample_domains(self.train_loader, n_samples=self.config['n_domains'], strategy=self.config['domain_sampling_strategy'])
            results = self.outer_loop(episode_domains, training=True)

            # TODO: fix acc and res logging, also write to csv file every log_freq steps
            #self.writer.add_scalar('Accuracy/Train', results['accuracy'], self.current_iter)
            #self.writer.add_scalar('Loss/Train', results['loss'], self.current_iter)
            # TODO: only every log_freq steps
            logging.info(f"EPSIODE:{episode} Accuracy: {results['accuracy']:.3f} Meta Loss: {results['loss']:.3f}")

            #TODO add validation
            #if self.current_iter % self.config['valid_freq'] == 0:
                #self.validate()
        


    def outer_loop(self, domains, training: bool):
        """ Iterate over one batch """
        meta_loss = 0
        meta_acc = 0
        for domain in domains:
            support_batch = next(iter(self.train_loader[domain]))
            query_batch = next(iter(self.train_loader[domain]))
            results = self.inner_loop(support_batch, query_batch, training=training)

            meta_loss += results["loss"]
            meta_acc += results["accuracy"]


        meta_results = {"loss": meta_loss, "accuracy" : meta_acc}
        return meta_results





    def inner_loop(self, support_batch, query_batch, training):
        # send tensors to model device
        support_x, support_masks, support_labels, support_domains = support_batch['x'], support_batch['masks'], support_batch['labels'], support_batch['domains']
        support_x = support_x.to(self.config['device'])
        support_masks = support_masks.to(self.config['device'])
        support_labels = support_labels.to(self.config['device'])

        query_x, query_masks, query_labels, query_domains = query_batch['x'], query_batch['masks'], query_batch['labels'], query_batch['domains']
        query_x = query_x.to(self.config['device'])
        query_masks = query_masks.to(self.config['device'])
        query_labels = query_labels.to(self.config['device'])

        if training:
            
            ##create copy of self.model
            fast_model = deepcopy(self.model)

            for grad_step in self.config['inner_gd_steps']:
                ## TODO implement for bert weights
                ##self.bert_opt.zero_grad()
                fast_model.zero_grad()
                output = fast_model(x=support_x, masks=support_masks, labels=support_labels, domains=support_domains) # domains is ignored for now
                logits = output['logits']
                loss = output['loss']
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad_norm'])
                ##self.bert_opt.step()
                ##self.bert_scheduler.step()
                self.ffn_opt.step()
            
        else:
            ##TODO update this part
            with torch.no_grad():
                output = self.model(x=support_x, masks=support_masks, labels=support_labels, domains=support_domains) # domains is ignored for now
                logits = output['logits']
                loss = output['loss']

        results = {'accuracy': output['acc'], 'loss': loss.item()}
        return results