from .base_config import base_config, get_config

new_config = {
    'exp_name': "protonet_default",
    'trainer': 'prototypical',
    'num_training_examples': 14000,
    'val_episodes': 10,
    'n_support': 10,
    'n_query': 20,
    'freeze_until_layer': 10,
}

config = get_config(base_config, new_config)