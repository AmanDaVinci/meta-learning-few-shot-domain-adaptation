from .base_config import base_config, get_config

new_config = {
    'exp_name': "protonet_lowshift_seed42",
    'seed': 41,
    'trainer': 'prototypical',
    'num_training_examples': 16000,
    'n_support': 5,
    'n_query': 20,
    'n_test_query': 100,
    'freeze_until_layer': 10,
    'train_domains':  ['apparel', 'baby', 'books', 'camera_photo', 'electronics',
                       'health_personal_care', 'kitchen_housewares', 'magazines',
                       'music', 'software', 'imdb', 'toys_games', 'video', 'MR'],
    'val_domains': ['sports_outdoors'],
    'test_domains': ['dvd'],
    'save_at': 3500,
}

config = get_config(base_config, new_config)