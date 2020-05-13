import logging
from .base_config import base_config, get_config

new_config = {
    'trainer': 'evaluation',
    'exp_name': 'maml_evaluation',
    'epochs': 1,
    'batch_size': 32,
    'n_evaluations': 4, # run the experiment with this many different training batches
    'warmup_steps': 0, # number of steps to warm up for learning rate scheduler
    'k_shot': 4, # train on this many samples from test domain
    'eval_experiment': 'maml_test',
    'device': 'cpu'
}

config = get_config(base_config, new_config)