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
python <MAIN_SCRIPT> --trainer=<TRAINER_NAME> --config=<CONFIG_FILE> --<MODE>
```
Where the MAIN_SCRIPT can be one of the following two:
* main.py to run the model once
* main_loop.py to run the model 3 times with different random seeds (otherwise they are the same)
The TRAINER_NAME can be either:
* multitask
* prototypical
* maml
The CONFIG_FILE is a file that contains all the hyperparameters for training, see the examples for training in the configs folder and see the full list of config arguments [here](google.com)
MODE flag can be
* train
* test
Both flags can be used at the same time.
An example of the usage for training maml with 3 different seeds on the dvd test dataset would be:
```
python main_loop.py --trainer=maml --config=configs.maml_config_dvd --train
```
## File structure

### On the repository

### Created by the script
