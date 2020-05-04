import logging
from .base_config import base_config, get_config

new_config = {
    'trainer' : 'maml',
    'k_shot_num' : 5,
    'n_domains' : 5,
    'fast_weight_lr' : 1e-3,
    'meta_lr' : 1e-3, 
    'collapse_domains' : False,
    'exp_name': 'multitask_test',
    'episodes': 100,
    'domain_sampling_strategy' : "uniform",
    'inner_gd_steps': 3
}

config = get_config(base_config, new_config)