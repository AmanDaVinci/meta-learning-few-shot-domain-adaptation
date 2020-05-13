from .base_config import base_config, get_config

new_config = {
    'exp_name': "protonet_support5",
    'trainer': 'prototypical',
    'n_episodes': 10,
    'val_episodes': 10,
    'n_support': 5,
    'n_query': 20,
}

config = get_config(base_config, new_config)