# DeepLearningMiniProject1
## Replicating the result
### Installing environment
Install the conda environment included, and use `conda activate dl2022` to activate it.
### Data augmentation search
Run `autoalbument-search --config-dir ./data_augmentation/` to do a search of data augmentation based on the dataset. This will create the augmentation policy in `./data_augmentation/outputs/<current_date>/<current_time>/`.
### Run hyperparameter search
Run `python optuna_script.py` to do a hyperparameter search. You can repeat this step for as many times as you want, this will give incrementally better results (possibly diminishing) as long as you keep the record file `optuna_records.db`.
### Load model and test performance of the model
Run `ResNet.py` with `final_weight.ckpt` in the same directory.

## Summary of methods
### Hyperparameter optimization
We use [Optuna](https://optuna.org/), a hyperparameter tuning package to find optimal sets of hyperparameters. 
The optimization of hyperparameter in this project can be formulated in different ways. We mainly tried two different formulations of this problem.
#### Attempt 1: [CMAES](https://arxiv.org/pdf/1604.00772.pdf)
We first use `CMAES` evolution algorithm to find the best combinations of hyperparameters. We directly use the final validation accuracy as the search criterion to search for optimal networks.

However, since the model also have to satisfy the constraint of <5 million parameters, we need a way to encode whether a network is small enough inside the search criterion. What we used for this stage
is we don't train networks with sizes bigger than 5 million and directly assign a fitness score of `-n_params/1e6` to the network. We've found that if we simply assign 0, the algorithm struggles to generate networks smaller than 5M, which is a somewhat intuitive outcome.
This way the search algorithm knows to look for smaller networks when it produces too big of a network.

However, this approach might be confusing to the search algorithm since really two types of information are being encoded into the same criterion function. We attempt a remedy by using Multi-Objective genetic algorithms.

#### Attempt 2: [NGAII](https://ieeexplore.ieee.org/document/996017)
NGAII is a fast and high performance multi-objective genetic algorithm. Instead of searching wrt one criterion, it tries to search for models on the furthest pareto front of multiple objective functions. In this attempt we consider the previous search problem as a multi-objective optimization problem. The first objective is to 
create networks that satisfy the size constraint. The objective function follows from the previous attempt: `min(-n_params/1e6, 5)`. The purpose of this objective is to select against networks larger than 5M but is neutral towards networks smaller than 5M.

The second objective is the final validation accuracy. For networks with sizes larger than 5M, we don't train them and assign a criterion of 0. This, admittedly, is a less than perfect way to handle the bigger sized models, but is necessary since we are computationally constrained.

### Data augmentation
We use [AutoAlbument](https://github.com/albumentations-team/autoalbument).
TODO

### Dropblock
We use [Dropblock](https://arxiv.org/abs/1810.12890).
TODO

