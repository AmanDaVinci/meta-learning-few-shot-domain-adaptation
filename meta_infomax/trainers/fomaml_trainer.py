import torch
from copy import deepcopy

import logging
import torch
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup, AdamW
from typing import Dict

from meta_infomax.datasets.fudan_reviews import MultiTaskDataset
from meta_infomax.datasets.utils import sample_domains
from meta_infomax.trainers.super_trainer import BaseTrainer


class FOMAMLTrainer(BaseTrainer):
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
        self.train_loader = train_data.domain_dataloaders(batch_size=config['k_shot_num'],
                                                          shuffle=True)
        self.val_loader = val_data.domain_dataloaders(batch_size=config['k_shot_num'],
                                                      shuffle=False)
        self.test_loader = test_data.domain_dataloaders(batch_size=config['k_shot_num'],
                                                        shuffle=False)

        ## define iterators
        self.train_loader_iterator = {domain: iter(domain_loader) for domain, domain_loader in self.train_loader.items()}
        self.val_loader_iterator = {domain: iter(domain_loader) for domain, domain_loader in self.val_loader.items()}
        self.test_loader_iterator = {domain: iter(domain_loader) for domain, domain_loader in self.test_loader.items()}

        self.current_episode = 0

        self.ffn_opt = optim.Adam(self.model.head.parameters(), lr=self.config['meta_lr'])

        self.bert_opt = AdamW(self.model.encoder.parameters(), lr=config['meta_lr'], correct_bias=False,
                              weight_decay=config['weight_decay'])  # use transformers AdamW

        self.bert_scheduler = get_linear_schedule_with_warmup(self.bert_opt,
                                                              num_warmup_steps=config['warmup_steps'],
                                                              num_training_steps=len(self.train_loader) *
                                                                                 config['epochs'])

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
            episode_domains = sample_domains(self.train_loader, n_samples=self.config['n_domains'],
                                             strategy=self.config['domain_sampling_strategy'])
            results = self.outer_loop(episode_domains, mode='training')

            # TODO:  also write to csv file every log_freq steps
            self.writer.add_scalar('Query_Accuracy/Train', results['accuracy'], self.current_episode)
            self.writer.add_scalar('Meta_Loss/Train', results['loss'].item(), self.current_episode)
            # TODO: only every log_freq steps
            logging.info(f"EPSIODE:{episode} Query_Accuracy: {results['accuracy']:.3f} Meta_Loss: {results['loss'].item():.3f}")

            # TODO add validation
            if self.current_episode % self.config['valid_freq'] == 0:
                self.fine_tune(mode = 'validate')

    def fine_tune(self, mode):
        """ Main validation loop """

        acc_total = 0
        loss_total = 0

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
            acc_total += results['accuracy']
            loss_total += results['loss']

        mean_accuracy = acc_total / (episode + 1)
        mean_loss = results['loss'] / (episode + 1)
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
            loader = self.train_loader_iterator
        elif mode == 'validate':
            loader = self.val_loader_iterator
        elif mode == 'test':
            loader = self.test_loader_iterator

        domain_grads_head = []
        domain_grads_bert = []
        
        for domain in domains:
            batch_iterator = loader[domain]

            grads_head, grads_bert, results = self.inner_loop(batch_iterator)
            domain_grads_head.append(grads_head)
            domain_grads_bert.append(grads_bert)

            meta_loss += results["loss"]
            meta_acc += results["accuracy"]

        ### updating main parameters - head
        
        if mode == "training":
        ## summing domain grads

            sum_grads_head = domain_grads_head[0]
            for grad_ind in range(1, len(domain_grads_head)):
                for layer_ind in range(len(domain_grads_head[grad_ind])):
                    sum_grads_head[layer_ind] += domain_grads_head[grad_ind][layer_ind]

            bert_grad_keys = domain_grads_bert[0].keys()
            sum_grads_bert = domain_grads_bert[0]
            for grad_ind in range(1, len(domain_grads_bert)):
                for layer_key in bert_grad_keys:
                    sum_grads_bert[layer_key] += domain_grads_bert[grad_ind][layer_key]

            ### putting grads into the parameters
            self.model.update_head_grads(sum_grads_head)
            self.model.update_bert_grads(sum_grads_bert)

            ## calling the update
            self.ffn_opt.step()
            self.bert_opt.step()
            self.bert_scheduler.step()

            self.ffn_opt.zero_grad()
            self.bert_opt.zero_grad()

        meta_results = {"loss": meta_loss, "accuracy": meta_acc / len(domains)}
        return meta_results

    def inner_loop(self, batch_iterator, mode = 'training'):
        # send tensors to model device
        
        support_batch = next(batch_iterator)
        support_x, support_masks, support_labels, support_domains = support_batch['x'], support_batch['masks'], support_batch[
            'labels'], support_batch['domains']
        support_x = support_x.to(self.config['device'])
        support_masks = support_masks.to(self.config['device'])
        support_labels = support_labels.to(self.config['device'])

        query_batch = next(batch_iterator)
        query_x, query_masks, query_labels, query_domains = query_batch['x'], query_batch['masks'], query_batch['labels'], \
                                                            query_batch['domains']
        query_x = query_x.to(self.config['device'])
        query_masks = query_masks.to(self.config['device'])
        query_labels = query_labels.to(self.config['device'])

        ### initial update based on the net's weights
        ## TODO implement for bert weights
        ##self.bert_opt.zero_grad()
        fast_weight_net = deepcopy(self.model)
        self.ffn_opt_inner = optim.Adam(fast_weight_net.parameters(), lr=self.config['fast_weight_lr'])



        for grad_step in range(0, self.config['inner_gd_steps'] - 1):


            output = fast_weight_net(x=support_x, masks=support_masks, labels=support_labels, domains=support_domains)
            logits = output['logits']
            loss = output['loss']
            loss.backward()

            torch.nn.utils.clip_grad_norm_(fast_weight_net.parameters(), self.config['clip_grad_norm'])
            self.ffn_opt_inner.step()
            self.ffn_opt_inner.zero_grad()

        ### FOMAML - we'll use last step's gradients for update
        output = fast_weight_net(x=query_x, masks=query_masks, labels=query_labels,
                                 domains=query_domains)  # domains is ignored for now
        logits = output['logits']
        loss = output['loss']

        if mode == "training":

            loss.backward()

            torch.nn.utils.clip_grad_norm_(fast_weight_net.parameters(), self.config['clip_grad_norm'])

            grad_head = fast_weight_net.get_head_grads()
            grad_bert = fast_weight_net.get_bert_grads()

        elif mode == "validate" or mode == "test":
            grad_head, grad_bert = None

        results = {'accuracy': output['acc'], 'loss': loss}
        return grad_head, grad_bert, results
