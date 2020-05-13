from .base_config import base_config, get_config

new_config = {
    'exp_name': "protonet_traindomain_small",
    'trainer': 'prototypical',
    'n_episodes': 10,
    'val_episodes': 10,
    'n_support': 10,
    'n_query': 20,
    'train_domains': ['books', 'imdb'],
    'val_domains': ['software', 'electronics'], 
}

config = get_config(base_config, new_config)