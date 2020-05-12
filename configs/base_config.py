import logging

base_config = {
    'trainer': 'multitask',
    'lr': 5e-5, # only for bert for now
    'weight_decay': 0.01,
    'warmup_steps': 100, # number of steps to warm up for learning rate scheduler
    'collapse_domains': True, # whether to load all datasets together or individually
    'epochs': 3,
    'batch_size': 16,
    'valid_freq': 200,
    'save_freq': 100,
    'unfreeze_layers': (10, 11), # layers of bert to unfreeze
    'clip_grad_norm': 1,
    'validation_size': 0, # percentage of train data used for validation data (over split, not over domain)
    'random_state': 42, # random state for reproducibility, currently only used for splitting train and val
    'device': 'cpu',
    'data_dir': 'data/mtl-dataset/',
    'transformer_name': 'bert-base-uncased',
    'domains': ['apparel', 'baby', 'books', 'camera_photo', 'electronics',
                'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines',
                'music', 'software', 'sports_outdoors', 'toys_games', 'video'],
    'train_domains': ['apparel', 'baby', 'books', 'camera_photo', 'health_personal_care',
                        'imdb', 'kitchen_housewares', 'magazines', 'sports_outdoors', 'toys_games'],
    'val_domains': ['software', 'electronics'],
    'test_domains': ['music', 'video'],
    'log_level': logging.INFO,
    'log_freq': 100
}

def get_config(base_config, new_config):
    config = dict(base_config)
    for key, value in new_config.items():
        config[key] = value
    return config