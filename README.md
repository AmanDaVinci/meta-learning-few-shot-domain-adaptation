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

## File structure

### On the repository

### Created by the script
