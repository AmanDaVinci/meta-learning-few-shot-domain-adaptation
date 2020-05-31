# Meta-learning for Domain Adaptation in Sentiment Analysis

This repository contains the code for comparing the following three approaches on sentiment analysis with domain shift: 
* Hard-shared multitask learning
* [Prototypical network](https://arxiv.org/abs/1703.05175)
* [Model-agnostic meta-learning (MAML)](https://arxiv.org/abs/1703.03400)
We used the [Fudan review dataset](https://github.com/FrankWork/fudan_mtl_reviews) with sentiment labels on 16 domains.
All models use an architecture based on [BERT](https://arxiv.org/abs/1810.04805) with a feedworward head.

## Usage
The used packages can be installed with
```
pip install -r requirements.txt
```
The general command for running a model is as follows
```
python <MAIN_SCRIPT> --trainer=<TRAINER_NAME> --config=configs.<CONFIG_FILE> --<MODE>
```
Where the MAIN_SCRIPT can be one of the following two:
* main.py to run the model once
* main_loop.py to run the model 3 times with different random seeds (otherwise they are the same)
<!-- end of the list -->
The TRAINER_NAME can be either:
* multitask
* prototypical
* maml
<!-- end of the list -->
The CONFIG_FILE is a file that contains all the hyperparameters for training, see the examples for training in the configs folder and see the full list of config arguments [here](google.com).\
\
MODE flag can be
* train
* test
<!-- end of the list -->
Both flags can be used at the same time.\
\
An example of the usage for training maml with 3 different seeds on the dvd test dataset would be:
```
python main_loop.py --trainer=maml --config=configs.maml_config_dvd --train
```
For more examples on the usage and analysis of the models, see the notebooks folder.
## File structure

    .
    ├── configs                         # config files with hyperparameters
        |──base_config.py               # contains default parameters
        |──..._config.py                # run specific parameters (extend and overwrite base)
    ├── meta_infomax                    # contains most parts of the implementation of the project
        ├──datasets                     # scripts for processing and representing the data
        ├──losses                       # contains the specific loss function for ProtoNet
        ├──models                       # contains the files for the shared model architecture (BERT, ffn head etc.)
        ├──trainers                     # The spedific trainer classes for the different approaches
            ├──evaluation_trainer.py    # Class for evaluating trained models, currently compatible with ProtoNet
            ├──fomaml_trainer.py        # Class for first-order MAML approximation
            ├──maml_trainer.py          # Class for original MAML (not used for the project)
            ├──multitask_trainer.py     # Class for hard-shared multitask model
            ├──PMIScorer.py             # Class for calculating PMI scores and sort domains based on that
            ├──protonet_trainer.py      # Class for Prototypical Network implementation
            |──super_trainer.py         # Superclass of the above classes
    ├── notebooks                       # contains usage and analyses example notebooks
    ├── checkpoints*                    # Stored models from training
        |──...                          # model specific folders, name extracted from config
    ├── data/mtl_dataset                # Processed data files per domain and split
    ├── results*
        |──.../logs                     # model specific folders, name extracted from config
                |──....0                # tensorboard logs for the model
                |──logs.log             # logging output
    ├── .gitignore                  
    ├── LICENSE
    ├── README.md
    ├── main.py                         # Run model once
    ├── main_loop.py                    # Run model 3 times with different seeds
    └── requirements.txt                # Install for used packages

    Note: folder with * and their subfolders are created automatically during run