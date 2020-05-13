import logging
from .base_config import base_config, get_config

new_config = {
    'trainer': 'evaluation',
    'exp_name': 'multitask_test3',
    'weight_decay': 0,
    'lr': 1e-5,
    'epochs': 3,
    'device': 'cuda',
    'batch_size': 32,
    'n_evaluations': 3, # run the experiment with this many different training batches
    'warmup_steps': 0, # number of steps to warm up for learning rate scheduler
    'k_shot': 4, # train on this many samples from test domain
}

config = get_config(base_config, new_config)
