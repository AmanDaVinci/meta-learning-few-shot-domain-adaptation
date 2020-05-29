import logging
from .base_config import base_config, get_config

new_config = {
    'trainer': 'evaluation',
    'exp_name': 'multitask_10-11-3500-sports-dvd',
    'weight_decay': 0,
    'lr': 1e-5,
    'epochs': 1,
    'device': 'cuda',
    'batch_size': 32,
    'n_evaluations': 1, # run the experiment with this many different training batches
    'warmup_steps': 0, # number of steps to warm up for learning rate scheduler
    'k_shot': 5, # train on this many samples from test domain
    'test_domains': ['MR'],
    'pmi_scorer': True
}

config = get_config(base_config, new_config)
