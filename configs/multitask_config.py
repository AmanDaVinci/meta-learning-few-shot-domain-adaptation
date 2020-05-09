

import logging
from .base_config import base_config, get_config

new_config = {
    'exp_name': 'multitask_test',
    'epochs': 1,
    'batch_size': 4,
}

config = get_config(base_config, new_config)