from .base_config import base_config, get_config

new_config = {
    'exp_name': "protonet_highshift_seed42",
    'seed': 41,
    'trainer': 'prototypical',
    'num_training_examples': 16000,
    'n_support': 5,
    'n_query': 20,
    'n_test_query': 100,
    'freeze_until_layer': 10,
    'train_domains':  ['apparel', 'baby', 'books', 'camera_photo', 'electronics',
                       'health_personal_care', 'kitchen_housewares', 'magazines',
                       'music', 'software', 'sports_outdoors', 'toys_games', 'video', 'dvd'],
    'val_domains': ['imdb'],
    'test_domains': ['MR'],
    'save_at': 3500,
}

config = get_config(base_config, new_config)