

import logging
from .base_config import base_config, get_config

new_config = {
    'exp_name': 'multitask_6-11-3500-sports-dvd',
    'epochs': 1,
    'unfreeze_layers': (6,7,8,9,10,11), # layers of bert to unfreeze
    'batch_size': 16,
    'num_examples': 3500,
    # 'train_domains':  ['apparel', 'baby', 'books', 'camera_photo', 'electronics',
    #             'health_personal_care', 'kitchen_housewares', 'magazines',
    #             'music', 'software', 'sports_outdoors', 'toys_games', 'video', 'dvd'],
    # 'val_domains': ['imdb'],
    # 'test_domains': ['MR'],
    'device': 'cuda'
}

config = get_config(base_config, new_config)
