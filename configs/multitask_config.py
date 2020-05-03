import logging
from .base_config import base_config, get_config

new_config = {
    'exp_name': 'multitask_test2',
    'batch_size': 32,
    'device': 'cuda'
}

config = get_config(base_config, new_config)
