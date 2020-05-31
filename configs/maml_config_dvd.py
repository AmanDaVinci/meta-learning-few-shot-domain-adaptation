import logging
from .base_config import base_config, get_config

new_config = {
    'trainer' : 'maml',
    'n_domains' : 5,
    'valid_freq': 5,
    'valid_chunks':4,       #### validation/test data will be chunked into this many batches to fit in memory
    'fast_weight_lr' : 5e-5,
    'meta_lr' : 5e-5, 
    'collapse_domains' : False,
    'exp_name': 'maml_train',
    'episodes': 500,
    'val_episodes':5,
    'test_episodes':1,
    'domain_sampling_strategy' : "uniform",
    'inner_gd_steps': 3,
    'unfreeze_layers': (10, 11),
    'num_examples' : 3500,
    'seed':40,
    'train_domains':  ['apparel', 'baby', 'books', 'camera_photo', 'electronics',
                'health_personal_care', 'kitchen_housewares', 'magazines',
                'music', 'software', 'toys_games', 'video', 'MR', 'imdb'],
    'val_domains': ['sports_outdoors'],
    'test_domains': ['dvd'],
    'sort_test_by_pmi' : True
}

config = get_config(base_config, new_config)