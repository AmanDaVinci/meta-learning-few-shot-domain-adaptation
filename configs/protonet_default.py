from .base_config import base_config, get_config

new_config = {
    'exp_name': "protonet_default",
    'trainer': 'prototypical',
    'num_training_examples': 14000,
    'n_support': 5,
    'n_query': 20,
    'n_test_query': 100,
    'freeze_until_layer': 10,
}

config = get_config(base_config, new_config)