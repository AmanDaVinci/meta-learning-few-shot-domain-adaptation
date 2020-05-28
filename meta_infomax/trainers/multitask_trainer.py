import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from typing import Dict

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

        # for now, we say that the training data, is the train split of every train domain
        # we could eventually also include the test split of the train_domain
        train_data = MultiTaskDataset(tokenizer=self.tokenizer, data_dir=config['data_dir'], split='train',
                                      keep_datasets=config['train_domains'],
                                      random_state=config['random_state'], validation_size=0)
        if self.config['test_same_domains']:
            val_data = MultiTaskDataset(tokenizer=self.tokenizer, data_dir=config['data_dir'], split='test',
                                        keep_datasets=config['train_domains'],
                                        random_state=config['random_state'], validation_size=0)
        else:
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
                                         collate_fn=val_data.collator, shuffle=False)
            self.test_loader = DataLoader(test_data, batch_size=config['batch_size'],
                                          collate_fn=test_data.collator, shuffle=False)
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
                                                              num_training_steps=len(self.train_loader) * config['epochs'])
        self.n_examples_seen = 0

    def train(self):
        """Main training loop."""
        assert self.config['collapse_domains'] == True, 'only implemented for collapse_domains=True'
        logging.info("***** Running training *****")
        logging.info("  Num Batches = %d", len(self.train_loader))
        logging.info("  Num Epochs = %d", self.config['epochs'])
        logging.info("  Batch size = %d", self.config['batch_size'])
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch

            for i, batch in enumerate(self.train_loader):
                self.n_examples_seen = self.current_iter * self.config['batch_size']
                # num_examples overrides n_epochs
                if ('num_examples' in self.config and self.config['num_examples'] > 0 and
                    self.n_examples_seen >= self.config['num_examples']):
                        # we have seen num_examples examples, stop training loop
                        return

                self.current_iter += 1
                results = self._batch_iteration(batch, training=True)

                self.writer.add_scalar('Accuracy/Train', results['accuracy'], self.current_iter)
                self.writer.add_scalar('Loss/Train', results['loss'], self.current_iter)
                self.writer.add_scalar('Accuracy/Train-Examples', results['accuracy'], self.n_examples_seen)
                self.writer.add_scalar('Loss/Train-Examples', results['loss'], self.n_examples_seen)

                # TODO: only every log_freq steps
                # TODO: also write to csv file every log_freq steps
                if self.current_iter % self.config['log_freq'] == 0:
                    logging.info(f"EPOCH:{epoch} STEP:{i}\t Accuracy: {results['accuracy']:.3f} Loss: {results['loss']:.3f}")

                if self.current_iter % self.config['valid_freq'] == 0:
                    self.validate()

    def validate(self):
        """ Main validation loop """
        losses = []
        accuracies = []

        logging.info("***** Running evaluation *****")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, leave=False)):
                results = self._batch_iteration(batch, training=False)
                losses.append(results['loss'])
                accuracies.append(results['accuracy'])

        mean_accuracy = np.mean(accuracies)
        mean_loss = np.mean(losses)
        if mean_accuracy > self.best_accuracy:
            self.best_accuracy = mean_accuracy
            self.save_checkpoint(self.BEST_MODEL_FNAME)
        self.writer.add_scalar('Accuracy/Valid', mean_accuracy, self.current_iter)
        self.writer.add_scalar('Loss/Valid', mean_loss, self.current_iter)
        self.writer.add_scalar('Accuracy/Valid-Examples', mean_accuracy, self.n_examples_seen)
        self.writer.add_scalar('Loss/Valid-Examples', mean_loss, self.n_examples_seen)

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
            output = self.model(x=x, masks=masks, labels=labels, domains=domains)  # domains is ignored for now
            logits = output['logits']
            loss = output['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad_norm'])
            self.bert_opt.step()
            self.bert_scheduler.step()
            self.ffn_opt.step()
        else:
            with torch.no_grad():
                output = self.model(x=x, masks=masks, labels=labels, domains=domains)  # domains is ignored for now
                logits = output['logits']
                loss = output['loss']

        results = {'accuracy': output['acc'], 'loss': loss.item()}
        return results
