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

        # loaders are now dicts mapping from domains to individual loaders
        self.train_loader = train_data.domain_dataloaders(batch_size=config['k_shot_num'], collate_fn=train_data.collator,
                                                        shuffle=True)
        self.val_loader = val_data.domain_dataloaders(batch_size=config['k_shot_num'], collate_fn=val_data.collator,
                                                        shuffle=False)
        self.test_loader = test_data.domain_dataloaders(batch_size=config['k_shot_num'], collate_fn=test_data.collator,
                                                        shuffle=False)

        self.bert_scheduler = get_linear_schedule_with_warmup(self.bert_opt,
                                                              num_warmup_steps=config['warmup_steps'],
                                                              num_training_steps=len(self.train_loader) *
                                                              config['epochs'])
        self.current_episode = 0

        self.ffn_opt = optim.Adam(self.model.head.parameters(), lr=self.config['meta_lr'])

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
            results = self.outer_loop(episode_domains, mode='training')

            results['loss'].backward()

            self.ffn_opt.step()

            # TODO:  also write to csv file every log_freq steps
            self.writer.add_scalar('Query_Accuracy/Train', results['accuracy'], self.current_episode)
            self.writer.add_scalar('Meta_Loss/Train', results['loss'].item(), self.current_episode)
            # TODO: only every log_freq steps
            logging.info(f"EPSIODE:{episode} Query_Accuracy: {results['accuracy']:.3f} Meta_Loss: {results['loss'].item():.3f}")

            #TODO add validation
            if self.current_episode % self.config['valid_freq'] == 0:
                self.fine_tune(mode = 'validate')
        

    def fine_tune(self, mode):
        """ Main validation loop """

        
        if mode == 'validate':
            logging.info("***** Running evaluation *****")
            domains = self.config['val_domains']
            episodes = range(self.config['val_episodes'])
        elif mode == 'test':
            logging.info("***** Running test *****")
            domains = self.config['test_domains']
            episodes = range(self.config['test_episodes'])

        for episode in episodes:
            results = self.outer_loop(domains, mode=mode)
            
        mean_accuracy = results['accuracy']
        mean_loss = results['loss']
        if mean_accuracy > self.best_accuracy:
            self.best_accuracy = mean_accuracy
            self.save_checkpoint(self.BEST_MODEL_FNAME)
        self.writer.add_scalar('Query_Accuracy/' + mode, mean_accuracy, self.current_episode)
        self.writer.add_scalar('Meta_Loss/' + mode, mean_loss, self.current_episode)
        
        report = (f"[Validation]\t"
                  f"Query_Accuracy: {mean_accuracy:.3f} "
                  f"Total Meta_Loss: {mean_loss:.3f}")
        logging.info(report)

    def outer_loop(self, domains, mode: str):
        """ Iterate over one batch """
        meta_loss = 0
        meta_acc = 0

        if mode == 'training':
            loader = self.train_loader
        elif mode == 'validate':
            loader = self.val_loader
        elif mode == 'test':
            loader = self.test_loader

        for domain in domains:
            support_batch = next(iter(loader[domain]))
            query_batch = next(iter(loader[domain]))
            results = self.inner_loop(support_batch, query_batch)

            meta_loss += results["loss"]
            meta_acc += results["accuracy"]


        meta_results = {"loss": meta_loss/len(domains), "accuracy" : meta_acc/len(domains)}
        return meta_results


    def inner_loop(self, support_batch, query_batch):
        # send tensors to model device
        support_x, support_masks, support_labels, support_domains = support_batch['x'], support_batch['masks'], support_batch['labels'], support_batch['domains']
        support_x = support_x.to(self.config['device'])
        support_masks = support_masks.to(self.config['device'])
        support_labels = support_labels.to(self.config['device'])

        query_x, query_masks, query_labels, query_domains = query_batch['x'], query_batch['masks'], query_batch['labels'], query_batch['domains']
        query_x = query_x.to(self.config['device'])
        query_masks = query_masks.to(self.config['device'])
        query_labels = query_labels.to(self.config['device'])

        ### initial update based on the net's weights
        ## TODO implement for bert weights
        ##self.bert_opt.zero_grad()

        self.model.zero_grad()
        output = self.model(x=support_x, masks=support_masks, labels=support_labels, domains=support_domains) # domains is ignored for now
        logits = output['logits']
        loss = output['loss']
        grad = torch.autograd.grad(loss, self.model.head.parameters(), create_graph=True)


        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad_norm'])
        fast_weights_head = list(map(lambda p: p[1] - self.config['fast_weight_lr'] * p[0], zip(grad, self.model.head.parameters())))

        ### loop through rest of the steps
        for grad_step in range(1, self.config['inner_gd_steps']):
                    
            ## TODO implement for bert weights
            ##self.bert_opt.zero_grad()
            encoded_data = self.model.encode(x=support_x, masks=support_masks)
            output = self.model.classify_encoded(encoded_data, support_labels, custom_params= fast_weights_head)
            logits = output['logits']
            loss = output['loss']
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad_norm'])
            grad = torch.autograd.grad(loss, fast_weights_head, create_graph = True)

            fast_weights_head = list(map(lambda p: p[1] - self.config['fast_weight_lr'] * p[0], zip(grad, fast_weights_head)))
            
        ### classifiy query set and get loss for meta update
        self.model.zero_grad()
        query_encoded = self.model.encode(x=query_x, masks=query_masks)
        output = self.model.classify_encoded(query_encoded, query_labels, fast_weights_head)
        loss = output['loss']
            

        results = {'accuracy': output['acc'], 'loss': loss}
        return results