import csv
import logging
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict

from meta_infomax.datasets.fudan_reviews import MultiTaskDataset
from meta_infomax.trainers.super_trainer import BaseTrainer


class EvaluationTrainer(BaseTrainer):
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
        self.load_checkpoint(config['eval_experiment'])
        self.model_state_dict = self.model.state_dict().copy()  # we have to re init at every evaluation
        self.bert_opt_dict = self.bert_opt.state_dict().copy()
        self.ffn_opt_dict = self.ffn_opt.state_dict().copy()

        # for now, we say that the training data, is the train split of every train domain
        # we could eventually also include the test split of the train_domain
        train_data = MultiTaskDataset(tokenizer=self.tokenizer, data_dir=config['data_dir'], split='train',
                                      keep_datasets=config['test_domains'],
                                      random_state=config['random_state'], validation_size=0.8)
        val_data = MultiTaskDataset(tokenizer=self.tokenizer, data_dir=config['data_dir'], split='val',
                                    keep_datasets=config['test_domains'],
                                    random_state=config['random_state'], validation_size=0.8)

        # we sample 1 batch of k samples, train on those samples for `epoch` steps,
        # and evaluate on the val set
        # we assume (for now) that k is small enough to fit in memory
        self.train_loader_positive = train_data.domain_dataloaders(label=1,
                                                                   batch_size=config['k_shot'],
                                                                   shuffle=True)
        self.train_loader_negative = train_data.domain_dataloaders(label=0,
                                                                   batch_size=config['k_shot'],
                                                                   shuffle=True)
        # make iterators for each dataset
        for domain in self.config['test_domains']:
            self.train_loader_positive[domain] = iter(self.train_loader_positive[domain])
            self.train_loader_negative[domain] = iter(self.train_loader_negative[domain])

        self.val_loader = val_data.domain_dataloaders(batch_size=config['batch_size'], shuffle=False)

    def run(self):
        """ Run the train-eval loop
        
        If the loop is interrupted manually, finalization will still be executed
        """
        try:
            for i in range(self.config['n_evaluations']):
                logging.info(f"Begin evaluation {i + 1}/{self.config['n_evaluations']}")
                self.evaluate()
        except KeyboardInterrupt:
            logging.info("Manual interruption registered. Please wait to finalize...")
            # self.save_checkpoint()

    def evaluate(self):
        """
        Train for a few steps on k samples for the 2 classes and evaluate on the test set.
        """
        for domain in self.config['test_domains']:
            self.model.load_state_dict(self.model_state_dict)  # we have to re init at every evaluation
            self.bert_opt.load_state_dict(self.bert_opt_dict)  # we have to re init at every evaluation
            self.ffn_opt.load_state_dict(self.ffn_opt_dict)    # we have to re init at every evaluation
            logging.info(f"Begin training on domain {domain} for {self.config['epochs']} epochs")
            self.train(domain)
            self.validate(domain)

    def train(self, domain):
        """Main training loop."""
        logging.info("***** Running training *****")
        logging.info("  Num Epochs = %d", self.config['epochs'])
        # sample k positive and negative samples, and train epoch steps on those
        positive_batch = next(self.train_loader_positive[domain])
        negative_batch = next(self.train_loader_negative[domain])

        samples = {}
        shuffle_idx = torch.randperm(len(positive_batch['x']))
        for k in positive_batch.keys():
            if isinstance(positive_batch[k], list):
                samples[k] = []
                samples[k].extend(positive_batch[k])
                samples[k].extend(negative_batch[k])
            else:
                samples[k] = torch.cat([positive_batch[k], negative_batch[k]])
                samples[k] = samples[k][shuffle_idx]

        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            self.current_iter += 1

            results = self._batch_iteration(samples, training=True)

            acc = results['accuracy']
            loss = results['loss']

            self.writer.add_scalar('Accuracy/Train', acc, self.current_iter)
            self.writer.add_scalar('Loss/Train', loss, self.current_iter)

            logging.info(f"EPOCH:{epoch+1}\t Accuracy: {acc:.3f} Loss: {loss:.3f}")

    def validate(self, domain):
        """ Main validation loop """
        losses = []
        accuracies = []

        val_loader = self.val_loader[domain]

        logging.info("***** Running evaluation *****")
        for i, batch in enumerate(tqdm(val_loader, leave=False)):
            results = self._batch_iteration(batch, training=False)
            losses.append(results['loss'])
            accuracies.append(results['accuracy'])

        mean_accuracy = np.mean(accuracies)
        mean_loss = np.mean(losses)

        report = ("[Validation]\t"
                  f"Domain {domain} "
                  f"Accuracy: {mean_accuracy:.3f} "
                  f"Total Loss: {mean_loss:.3f}")
        logging.info(report)

        # write result
        with open(Path(self.log_dir) / 'eval_result.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([self.config['k_shot'], self.config['epochs'], domain, mean_loss, mean_accuracy])

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
            self.ffn_opt.step()
        else:
            with torch.no_grad():
                output = self.model(x=x, masks=masks, labels=labels, domains=domains)  # domains is ignored for now
                logits = output['logits']
                loss = output['loss']

        results = {'accuracy': output['acc'], 'loss': loss.item()}
        return results
