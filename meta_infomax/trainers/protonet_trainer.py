import logging
from tqdm import tqdm
from typing import Dict
from pathlib import Path

import torch
import numpy as np
from numpy.random import choice
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW

from meta_infomax.datasets import utils
from meta_infomax.trainers.super_trainer import BaseTrainer
from meta_infomax.datasets.fudan_reviews import MultiTaskDataset
from meta_infomax.losses.prototypical_loss import PrototypicalLoss
from meta_infomax.models.protonet import Protonet


class ProtonetTrainer(BaseTrainer):
    """Train Prototypical Networks to classify sentiment across different domains"""

    def __init__(self, config: Dict):
        """Initialize the trainer with data, models and optimizers

        Parameters
        ---
        config:
            dictionary of configurations with the following keys: 
            {
                'exp_name': "protonet_defaut",
                'n_episodes': 100,
                'n_support': 3,
                'n_query': 2,
                'val_episodes': 5,
            }
        """
        super().__init__(config)
        bert, self.tokenizer, embedding_dim = utils.get_transformer(config['transformer_name'])
        self.model = Protonet(bert, bert_embed_dim=embedding_dim)
        self.model.freeze_bert(until_layer=config['freeze_until_layer'])
        self.model.to(config['device'])

        train_data = MultiTaskDataset(tokenizer=self.tokenizer, data_dir=config['data_dir'], split='train',
                                      keep_datasets=config['train_domains'], random_state=config['random_state'],
                                      validation_size=0, const_len=True)
        val_data = MultiTaskDataset(tokenizer=self.tokenizer, data_dir=config['data_dir'], split='train',
                                    keep_datasets=config['val_domains'], random_state=config['random_state'],
                                    validation_size=0, const_len=True)
        test_data = MultiTaskDataset(tokenizer=self.tokenizer, data_dir=config['data_dir'], split='train',
                                     keep_datasets=config['test_domains'], random_state=config['random_state'],
                                     validation_size=0, const_len=True)
        self.train_dls = train_data.episodic_dataloaders(batch_size=config['n_support']+config['n_query'],
                                                         collate_fn=train_data.collator, shuffle=True)
        self.val_dls = val_data.episodic_dataloaders(batch_size=config['n_support']+config['n_query'],
                                                     collate_fn=val_data.collator, shuffle=True)
        self.test_dls = test_data.episodic_dataloaders(batch_size=config['n_support']+config['n_query'],
                                                       collate_fn=test_data.collator, shuffle=True)
        self.train_dls = np.array(self.train_dls)
        self.val_dls = np.array(self.val_dls)
        self.test_dls = np.array(self.test_dls)

        self.prototypical_loss = PrototypicalLoss()
        self.ffn_opt = optim.Adam(self.model.head_encoder.parameters())
        self.bert_opt = AdamW(self.model.bert_encoder.parameters(),
                              lr=config['lr'], correct_bias=False, 
                              weight_decay=config['weight_decay'])
        self.bert_scheduler = get_linear_schedule_with_warmup(self.bert_opt,
                                                              num_warmup_steps=config['warmup_steps'],
                                                              num_training_steps=config['n_episodes'])
        self.seen_examples = 0

    def train(self):
        """Main training loop."""
        logging.info("***** Start Training *****")
        logging.info(f"  Number of Episodes = {self.config['n_episodes']}")
        logging.info(f"  Support Set Size = {self.config['n_support']}")
        logging.info(f"  Query Set Size = {self.config['n_query']}")

        self.validate()
        for i in range(self.config['n_episodes']):

            for domain_dataloader in self.train_dls:
                self.current_iter += 1
                episode, domain = self._prepare_episode(domain_dataloader) 
                results = self._episode_iteration(episode, training=True)
                self.writer.add_scalar(f'{domain}/Accuracy', results['accuracy'], self.current_iter)
                self.writer.add_scalar(f'{domain}/Loss', results['loss'], self.current_iter)
                logging.info(f"EPISODE:{i} \t Domain: {domain} "
                            f"Accuracy: {results['accuracy']:.3f} "
                            f"Prototypical Loss: {results['loss']:.3f}")
                self.seen_examples += self.config['n_support'] + self.config['n_query']

            self.validate()
            if self.seen_examples >= self.config['total_seen_examples']:
                logging.info(f"Seen {self.seen_examples} examples. Stop training.")
                break

    def validate(self):
        """ Main validation loop """
        total_losses = []
        total_accuracies = []

        for domain_dataloader in self.val_dls:
            domain_losses = []
            domain_accuracies = []

            for i in range(self.config['val_episodes']):
                episode, domain = self._prepare_episode(domain_dataloader) 
                results = self._episode_iteration(episode, training=False)
                domain_losses.append(results['loss'])
                domain_accuracies.append(results['accuracy'])
            
            domain_mean_accuracy = np.mean(domain_accuracies)
            domain_mean_loss = np.mean(domain_losses)
            self.writer.add_scalar(f'Validation-{domain}/Accuracy', domain_mean_accuracy, self.current_iter)
            self.writer.add_scalar(f'Validation-{domain}/Loss', domain_mean_loss, self.current_iter)
            self.writer.add_scalar(f'Validation-{domain}/Accuracy-vs-Seen-Examples', domain_mean_accuracy, self.seen_examples)
            self.writer.add_scalar(f'Validation-{domain}/Loss-vs-Seen-Examples', domain_mean_loss, self.seen_examples)
            report = (f"[Validation] \t Domain: {domain} "
                      f"Average accuracy: {domain_mean_accuracy:.3f} "
                      f"Average loss: {domain_mean_loss:.3f}")
            logging.info(report)
            total_losses.append(domain_mean_loss)
            total_accuracies.append(domain_mean_accuracy)

        mean_loss = np.mean(total_losses)
        mean_accuracy = np.mean(total_accuracies)
        if mean_accuracy > self.best_accuracy:
            self.best_accuracy = mean_accuracy
            self.save_checkpoint(self.BEST_MODEL_FNAME)

    def test(self, checkpoint_name=None):
        """ Main validation loop """

        self.load_checkpoint(self.config['exp_name'], checkpoint_name)
        for domain_dataloader in self.test_dls:
            domain_losses = []
            domain_accuracies = []
            for i in range(self.config['val_episodes']):
                episode, domain = self._prepare_episode(domain_dataloader) 
                results = self._episode_iteration(episode, training=False)
                domain_losses.append(results['loss'])
                domain_accuracies.append(results['accuracy'])
            domain_mean_accuracy = np.mean(domain_accuracies)
            domain_mean_loss = np.mean(domain_losses)
            report = (f"Domain: {domain} "
                      f"Average accuracy: {domain_mean_accuracy:.3f} "
                      f"Average loss: {domain_mean_loss:.3f}")
            print(report)

    def _episode_iteration(self, episode: tuple, training: bool):
        """ Iterate over one episode """

        # send tensors to model device
        x, masks = episode
        x = x.to(self.config['device'])
        masks = masks.to(self.config['device'])

        if training:
            self.bert_opt.zero_grad()
            self.ffn_opt.zero_grad()
            x_embeds = self.model(x, masks)
            loss, acc = self.prototypical_loss(x_embeds,
                                               n_support=self.config['n_support'],
                                               n_query=self.config['n_query'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad_norm'])
            self.ffn_opt.step()
            self.bert_opt.step()
            self.bert_scheduler.step()
        else:
            with torch.no_grad():
                x_embeds = self.model(x, masks)
                loss, acc = self.prototypical_loss(x_embeds,
                                                   n_support=self.config['n_support'],
                                                   n_query=self.config['n_query'])

        results = {'accuracy': acc.item(), 'loss': loss.item()}
        return results

    def _prepare_episode(self, domain_dataloader):
        ''' Prepare an episode by concatenating two batches

        One batch of support + query set with label-0 from the domain
        One batch of support + query set with label-1 from the domain

        Parameters
        ---
        domain_dataloader:
            list of two dataloaders: label-0 and label-1 from one domain
        
        Returns
        ---
        episode:
            tuple of tokenized text, attention masks 

        domain_name:
            name of the domain
        '''
        neg_batch = next(iter(domain_dataloader[0]))
        pos_batch = next(iter(domain_dataloader[1]))

        x = torch.cat([neg_batch['x'], pos_batch['x']])
        masks = torch.cat([neg_batch['masks'], pos_batch['masks']])
        episode = (x, masks)

        domain = pos_batch['domains'][0]
        return episode, domain