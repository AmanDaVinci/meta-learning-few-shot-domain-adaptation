import logging
from .base_config import base_config, get_config

new_config = {
    'trainer' : 'maml',
    'k_shot_num' : 5,
    'num_batches_for_query' : 4,        ## concatenate that many bacthes of size k-shot for validation/test
    'n_domains' : 5,
    'fast_weight_lr' : 5e-5,
    'meta_lr' : 5e-5, 
    'collapse_domains' : False,
    'exp_name': 'maml_train',
    'episodes': 500,
    'test_episodes':40,
    'val_episodes' : 5,
    'test_episodes' : 5,
    'domain_sampling_strategy' : "uniform",
    'inner_gd_steps': 3,
    'num_examples' : 'all',
}

config = get_config(base_config, new_config)