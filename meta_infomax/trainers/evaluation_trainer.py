import copy
import csv
import logging
import numpy as np
import torch
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
import json
from typing import Dict

from meta_infomax.datasets.fudan_reviews import MultiTaskDataset
from meta_infomax.trainers.PMIScorer import PMIScorer
from meta_infomax.trainers.super_trainer import BaseTrainer, RESULTS, LOG_DIR


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
        config['log_dir'] = RESULTS / config['exp_name'] / 'evaluation' / LOG_DIR
        super().__init__(config)
        self.load_checkpoint(config['exp_name'])#
        eval_dir = f"kshot_{config['k_shot']}_lr_{config['lr']}_epochs_{config['epochs']}"
        self.eval_dir = self.exp_dir / 'evaluation' / eval_dir # we save results here
        if "pmi_scorer" in config and config["pmi_scorer"]:
            self.eval_dir /= 'pmi'
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        config['log_dir'] = config['log_dir'].as_posix() # Path not serializable
        json.dump(config, open(self.eval_dir / 'config.json', 'w')) # write used parameters to file
        self.model_state_dict = copy.deepcopy(self.model.state_dict())  # we have to re init at every evaluation

        # for now, we say that the training data, is the train split of every train domain
        # we could eventually also include the test split of the train_domain
        train_data = MultiTaskDataset(tokenizer=self.tokenizer, data_dir=config['data_dir'], split='train',
                                      keep_datasets=config['test_domains'],
                                      random_state=config['random_state'], validation_size=0.8, const_len=True)
        val_data = MultiTaskDataset(tokenizer=self.tokenizer, data_dir=config['data_dir'], split='val',
                                    keep_datasets=config['test_domains'],
                                    random_state=config['random_state'], validation_size=0.8, const_len=True)

        if "pmi_scorer" in config and config["pmi_scorer"]:
            logging.info("PMI_SCORING")
            scorer = PMIScorer(self.tokenizer, config['test_domains'])
            sorted_ds = scorer.sort_datasets()

            self.train_loader_positive = {ds_name.split("_")[0]:
                                              iter(DataLoader(ds, batch_size=config['k_shot'], shuffle=False, collate_fn=val_data.collator))
                                          for ds_name, ds in sorted_ds.items() if int(ds_name.split("_")[1])}

            self.train_loader_negative = {ds_name.split("_")[0]:
                                              iter(DataLoader(ds, batch_size=config['k_shot'], shuffle=False, collate_fn=val_data.collator))
                                          for ds_name, ds in sorted_ds.items() if not int(ds_name.split("_")[1])}
        else:
            logging.info("NORMAL TESTING")
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
        self.result_dict = {domain: {'acc': [], 'loss': []} for domain in self.config['test_domains']}

        # evaluate n times
        for i in range(self.config['n_evaluations']):
            self.evaluation_ix = i
            logging.info(f"Begin evaluation {i + 1}/{self.config['n_evaluations']}")
            self.evaluate()
        
        # compute statistics and write to file
        for domain, results in self.result_dict.items():
            self.result_dict[domain]['mean_acc'] = np.array(results['acc']).mean()
            self.result_dict[domain]['mean_loss'] = np.array(results['loss']).mean()
            self.result_dict[domain]['std_acc'] = np.array(results['acc']).std()
            self.result_dict[domain]['std_loss'] = np.array(results['loss']).std()

        json.dump(self.result_dict, open(self.eval_dir / 'eval_result.json', 'w'))


    def evaluate(self):
        """
        Train for a few steps on k samples for the 2 classes and evaluate on the test set.
        """
        for domain in self.config['test_domains']:
            # we have to re init at every evaluation
            self.current_iter = 0
            self.model.load_state_dict(copy.deepcopy(self.model_state_dict))
            self.ffn_opt = optim.Adam(self.model.head.parameters())
            self.bert_opt = AdamW(self.model.encoder.parameters(),
                                  lr=self.config['lr'],
                                  correct_bias=False,
                                  weight_decay=self.config['weight_decay'])

            logging.info(f"Begin training on domain {domain} for {self.config['epochs']} epochs")
            self.train(domain)
            acc, loss = self.validate(domain)
            self.result_dict[domain]['acc'].append(acc)
            self.result_dict[domain]['loss'].append(loss)

    def train(self, domain):
        """Main training loop."""
        logging.info("***** Running training *****")
        logging.info("  Num Epochs = %d", self.config['epochs'])
        # sample k positive and negative samples, and train epoch steps on those
        positive_batch = next(self.train_loader_positive[domain])
        negative_batch = next(self.train_loader_negative[domain])

        # combine the samples in one big batch and shuffle them
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

            self.writer.add_scalar(f'Evaluation-{self.evaluation_ix}/Accuracy/Train', acc, self.current_iter)
            self.writer.add_scalar(f'Evaluation-{self.evaluation_ix}/Loss/Train', loss, self.current_iter)

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

        return mean_accuracy, mean_loss

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
            loss = output['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip_grad_norm'])
            self.bert_opt.step()
            self.ffn_opt.step()
        else:
            with torch.no_grad():
                output = self.model(x=x, masks=masks, labels=labels, domains=domains)  # domains is ignored for now
                loss = output['loss']

        results = {'accuracy': output['acc'], 'loss': loss.item()}
        return results
