from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict


from meta_infomax.trainers.super_trainer import BaseTrainer


class MAMLTrainer(BaseTrainer):
    """Train to classify sentiment across different domains/tasks"""

    def __init__(self, config: Dict):
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

    def train(self):
        """ Main training loop """
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            print("checking domains")
            print(self.train_loader.keys())
            for i, batch in enumerate(self.train_loader):
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
            for i, batch in enumerate(self.valid_iter):
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
