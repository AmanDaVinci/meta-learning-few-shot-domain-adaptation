

import logging
from .base_config import base_config, get_config

new_config = {
    'exp_name': 'multitask_6-12',
    'epochs': 1,
    'unfreeze_layers': (7,8,9,10,11), # layers of bert to unfreeze
    'batch_size': 16,
    'device': 'cuda'
}

config = get_config(base_config, new_config)
