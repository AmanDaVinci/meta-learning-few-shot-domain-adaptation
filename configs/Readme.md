## Defining configurations
The model runs take a config argument as input to define the hyperparameters of the training. Config files extend (and overwrite if colliding) base_config.py.


In the following we list all the arguments that are used by the models, with example values. Shared configs are used by all models, and base_config.py provides a default value for them (that can be overwritten in the specific config files).
### Shared configs

* ***'trainer': 'multitask'*** : name of trainer (used in main to load the proper class, so strictly multitask|prototypical|maml)
* ***'lr': 5e-5*** : learning rate used
* ***'weight_decay': 0.01*** : L2 regulatization value
* ***'exp_name': "multitask_train"*** : the name of the experiment, used to create checkpoint/resutls folders
* ***'warmup_steps': 100*** : number of steps to warm up for learning rate scheduler
* ***'collapse_domains': True*** : whether to load all datasets together or individually
* ***'epochs': 3*** : number of epochs (has no effect on MAML as that stops after certain number of examples)
* ***'valid_freq': 10*** : evaluation frequency
* ***'save_freq': 100*** : checkpoint saving frequency
* ***'unfreeze_layers': (10, 11)*** : layers of bert to unfreeze
* ***'clip_grad_norm': 1*** : gradient clipping max value
* ***'validation_size': 0*** : percentage of train data used for validation data (over split, not over domain)
* ***'random_state': 42*** : random state for reproducibility, currently only used for splitting train and val
* ***'device': 'cuda'*** : device to run the model on
* ***'data_dir': 'data/mtl-dataset/'*** : the directory for the data
* ***'transformer_name': 'bert-base-uncased'*** :
* ***'domains': ['apparel', 'baby', 'books', 'camera_photo', 'electronics',
            'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines',
            'music', 'software', 'sports_outdoors', 'toys_games', 'video', 'MR', 'dvd']*** : all the domains
* ***'train_domains':  ['apparel', 'baby', 'books', 'camera_photo''electronics',
            'health_personal_care', 'kitchen_housewares', 'magazines',
            'music', 'software', 'toys_games', 'video', 'MR', 'imdb']*** : the domains used for training
* ***'val_domains': ['sports_outdoors']*** : the domains used for validation
* ***'test_domains': ['dvd']*** : the domains used for testing
* ***'log_level': logging.INFO*** : the level of the logging messeges
* ***'log_freq': 100*** : how often to print statistics
* ***'k_shot': 5*** : fine-tune on this many samples per sentiment class (so examples per domain here would be 2*5). In multitask it only affects testing, but in meta learning used to define the episode size for training, validation and test.

Besides shared parameters, certain parameters are only used by specific models
### Multitask specific configs
* ***'batch_size': 16*** : batch size used for training
* ***'pmi_scorer': True*** : whether to use PMI scorer to select support set when testing

### MAML specific configs
* ***'n_domains' : 5*** : the number of domains to be sampled in an episode
* ***'valid_chunks': 8*** : validation/test data will be chunked into this many batches to fit in memory
* ***'fast_weight_lr' : 5e-5*** : learning rate for inner loop
* ***'meta_lr' : 5e-5*** : learning rate for meta-udaptes
* ***'episodes': 500*** : max number of episodes to  be done
* ***'val_episodes':5*** : how many episodes to be used for validation
* ***'test_episodes':5*** : how many episodes to be used for testing
* ***'domain_sampling_strategy' : "uniform"*** : how to sample domains in an episode: uniform|domain_size (weigh the probabilty with the domain size in the latter)
* ***'inner_gd_steps': 3*** : how many inner gradient steps to execute
* ***'num_examples' : 3500*** : how much data to show to the model (pass 'all' to show all data once)
* ***'sort_test_by_pmi' : True*** : whether to test based on PMI scoring

### Protonet specific oonfigs
* ***'num_training_examples': 14000*** : how much data to show to the model
* ***'n_support': 5*** : how many samples from  each class for prototypes
* ***'n_query': 20*** : how many query samples for updates
* ***'n_test_query': 100*** : how many query  samples when testing