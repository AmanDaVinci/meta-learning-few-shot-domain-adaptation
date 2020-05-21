import torch
from copy import deepcopy

import logging
import torch
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup, AdamW
from typing import Dict

from meta_infomax.datasets.fudan_reviews import MultiTaskDataset
from meta_infomax.trainers.super_trainer import BaseTrainer
from meta_infomax.datasets.utils import sample_domains

from random import shuffle, choice

class FOMAMLTrainer(BaseTrainer):
    """Train to classify sentiment across different domains/tasks"""

    def __init__(self, config: Dict):
        """Initialize the trainer with data, models and optimizers

        Parameters
        ---
        config:
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
                                      keep_datasets=config['train_domains'], random_state=config['random_state'],
                                      validation_size=0, const_len=True)
        val_data = MultiTaskDataset(tokenizer=self.tokenizer, data_dir=config['data_dir'], split='train',
                                    keep_datasets=config['val_domains'], random_state=config['random_state'],
                                    validation_size=0, const_len=True)
        test_data = MultiTaskDataset(tokenizer=self.tokenizer, data_dir=config['data_dir'], split='train',
                                     keep_datasets=config['test_domains'], random_state=config['random_state'],
                                     validation_size=0, const_len=True)

        # loaders are now dicts mapping from domains to individual loaders
        ### k-shot is defined per class (pos/negative), so here we multiply by 2, as we just sample the whole data
        self.train_loader = train_data.domain_dataloaders(batch_size=config['k_shot_num']*2,
                                                          shuffle=True)
        self.val_loader = val_data.domain_dataloaders(batch_size=config['k_shot_num']*2,
                                                      shuffle=False)
        self.test_loader = test_data.domain_dataloaders(batch_size=config['k_shot_num']*2,
                                                        shuffle=False)

        ## define iterators
        self.train_loader_iterator = {domain: iter(domain_loader) for domain, domain_loader in self.train_loader.items()}
        self.val_loader_iterator = {domain: list(iter(domain_loader)) for domain, domain_loader in self.val_loader.items()}
        self.test_loader_iterator = {domain: list(iter(domain_loader)) for domain, domain_loader in self.test_loader.items()}

        self.train_examples_per_episode = config['k_shot_num']*4 *  config['n_domains']

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
        logging.info("***** Running training - FoMAML *****")
        logging.info("  Num examples = %d", len(self.train_loader))
        logging.info("  Num Episodes = %d", self.config['episodes'])
        logging.info("  K-shot = %d", self.config['k_shot_num'])
        logging.info("  N-way = %d", self.config['n_domains'])

        for episode in range(self.current_episode, self.config['episodes']):
            self.current_episode = episode

            ### break if too amny iterators exhausted
            if len(self.train_loader.keys()) < self.config['n_domains']:
                logging.info("Breaking training: Not enough training data remaining")
                break

            episode_domains = sample_domains(self.train_loader, n_samples=self.config['n_domains'],
                                             strategy=self.config['domain_sampling_strategy'])
            results = self.outer_loop(episode_domains, mode='training')
            
            ### none returned if there is no more data
            if not results:
                break

            self.writer.add_scalar('Query_Accuracy/Train', results['accuracy'], self.current_episode)
            self.writer.add_scalar('Meta_Loss/Train', results['loss'], self.current_episode)
            logging.info(f"EPSIODE:{episode} Query_Accuracy: {results['accuracy']:.3f} Meta_Loss: {results['loss']:.3f}")

            if self.current_episode % self.config['valid_freq'] == 0:
                self.fine_tune(mode = 'validate')
            
            ## break if number of examples exceed the threshold
            if (self.config['num_examples'] != 'all' and episode * self.train_examples_per_episode  > self.config['num_examples']):
                logging.info("Breaking training: num examples threshold exceeded")
                break

    def test(self):
        self.fine_tune(mode = 'test')

    def fine_tune(self, mode):
        """ Main validation loop """
        if mode == 'validate':
            logging.info("***** Running evaluation *****")
            domains = self.config['val_domains']
        elif mode == 'test':
            logging.info("***** Running test *****")
            domains = self.config['test_domains']

        acc_across_domains = 0
        loss_across_domains = 0
        total_episodes = 0
        for fine_tune_domain in domains:
            acc_total = 0
            loss_total = 0
            if mode == 'validate':
                episodes = range(len(self.val_loader_iterator[fine_tune_domain]))
            elif mode == 'test':
                episodes = range(10)

            logging.info("Fine tuning on domain: " + str(fine_tune_domain) + " num episodes: " + str(episodes))

            for episode in episodes:
                results = self.outer_loop([fine_tune_domain], mode=mode, episode=episode)
                acc_total += results['accuracy']
                loss_total += results['loss']

            mean_accuracy = acc_total / (episode + 1)
            mean_loss = loss_total / (episode + 1)

            report = (f"[Validation]\t"
                    f"Query_Accuracy: {mean_accuracy:.3f} "
                    f"Total Meta_Loss: {mean_loss:.3f}")
            logging.info("Domain " + fine_tune_domain  + " performance")
            logging.info(report)

            acc_across_domains += acc_total
            loss_across_domains += loss_total
            total_episodes += episode+1

        ### averaging over fine tune domains
        acc_across_domains /= total_episodes
        loss_across_domains /= total_episodes
        if acc_across_domains > self.best_accuracy and mode == 'validate':
            self.best_accuracy = acc_across_domains
            self.save_checkpoint("unfrozen_bert:"+ str(self.config['unfreeze_layers']) + "_num_examples:" + str(self.config['num_examples']) + "_" + self.BEST_MODEL_FNAME)
        self.writer.add_scalar('Avg_FineTune_Accuracy/' + mode, acc_across_domains, self.current_episode)
        self.writer.add_scalar('Avg_FineTune_Loss/' + mode, loss_across_domains, self.current_episode)

        report = (f"[Validation]\t"
                f"Query_Accuracy: {acc_across_domains:.3f} "
                f"Total Meta_Loss: {loss_across_domains:.3f}")
        logging.info("Average fine tune performance across domains")
        logging.info(report)

    def outer_loop(self, domains, mode: str, episode = None):
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

            grads_head, grads_bert, results = self.inner_loop(batch_iterator, mode = mode, episode=episode)

            ## return if the train iterator is exhausted
            if results == 'exhausted':
                ### remove domain from selectables
                del loader[domain]
                del self.train_loader[domain]
                ### select random replacement from remaining ones
                remaining_domians = list(set(loader.keys()) - set(domains))
                if len(remaining_domians) == 0:
                    logging.info("No more populated domains remain, breaking train")
                    return
                logging.info("domain " + domain + " exhausted, appending new one")
                new_dom = choice(remaining_domians)
                domains.append(new_dom)
                continue


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

    def inner_loop(self, batch_iterator, mode = 'training', episode = None):
        # episode number of only used in val/test, to select batches one by one
        
        if mode == 'training':
            ### checking if the iterator is exhausted
            try:
                support_batch = next(batch_iterator)
                query_batch = [next(batch_iterator)]
                query_chunks = 1
            except StopIteration:
                print("dataset was exhausted, returning")
                return None, None, 'exhausted'
        else:
            ### for test/valid, we draw a batch in each episode and test on all the rest
            support_batch = batch_iterator[episode]
            query_chunks = 30
            query_batch = self.concatenate_remaining_batches_and_chunk(batch_iterator,episode, query_chunks)
            ##rewriting with actual number of chunks
            query_chunks = len(query_batch)


        support_x, support_masks, support_labels, support_domains = support_batch['x'], support_batch['masks'], support_batch[
            'labels'], support_batch['domains']
        support_x = support_x.to(self.config['device'])
        support_masks = support_masks.to(self.config['device'])
        support_labels = support_labels.to(self.config['device'])
        
        ### initial update based on the net's weights
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
        ###looping through the query chunks
        loss = 0
        query_acc = 0
        for chunkInd in range(query_chunks):
            ##no grad calc if validation/test
            if mode != 'training':
                torch.set_grad_enabled(False)
            else:
                torch.set_grad_enabled(True)

            query_x, query_masks, query_labels, query_domains = query_batch[chunkInd]['x'], query_batch[chunkInd]['masks'], query_batch[chunkInd]['labels'], \
                                                                query_batch[chunkInd]['domains']
            query_x = query_x.to(self.config['device'])
            query_masks = query_masks.to(self.config['device'])
            query_labels = query_labels.to(self.config['device'])
            output = fast_weight_net(x=query_x, masks=query_masks, labels=query_labels,
                                    domains=query_domains)  # domains is ignored for now
            logits = output['logits']

            if mode == "training":
                loss += output['loss']
            else:
                #### not saving comp graph if not training
                loss += output['loss'].item()
            query_acc += output['acc']

        query_acc /= query_chunks

        if mode == "training":

            loss.backward()

            torch.nn.utils.clip_grad_norm_(fast_weight_net.parameters(), self.config['clip_grad_norm'])

            grad_head = fast_weight_net.get_head_grads()
            grad_bert = fast_weight_net.get_bert_grads()
            loss = loss.item()

        elif mode == "validate" or mode == "test":
            grad_head, grad_bert = None, None

        results = {'accuracy': query_acc, 'loss': loss}
        return grad_head, grad_bert, results

    def concatenate_remaining_batches_and_chunk(self, iterator, index, num_chunks):

        x = torch.chunk(torch.cat([batch["x"] for batch in iterator[0:index] + iterator[index+1:]]), num_chunks)
        masks = torch.chunk(torch.cat([batch["masks"] for batch in iterator[0:index] + iterator[index+1:]]), num_chunks)
        labels = torch.chunk(torch.cat([batch["labels"] for batch in iterator[0:index]+ iterator[index+1:]]), num_chunks)
        ### domains not used in the script
        domains = None

        batches = []
        for ind in range(len(x)):
            batches.append({"x":x[ind], "masks":masks[ind], "labels":labels[ind], "domains": None})

        return batches
