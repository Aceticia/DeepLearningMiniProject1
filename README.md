# DeepLearningMiniProject1
## Replicating the result
### Installing environment
Install the conda environment included, and use `conda activate dl2022` to activate it.
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

However, since the model also have to satisfy the constraint of <5 million parameters, we need a way to encode whether a network is small enough inside the search criterion. This way the search algorithm knows to look for smaller networks when it produces ones that are too big. We do not evaluate the network's performance on those and impose a dynamic penalty of `-n_params/1e6`.

However, this approach might be confusing to the search algorithm since really two types of information are being encoded into the same criterion function. We attempt a remedy by using Multi-Objective genetic algorithms.

#### Attempt 2: [NGAII](https://ieeexplore.ieee.org/document/996017)
NGAII is a fast and high performance multi-objective genetic algorithm. Instead of searching wrt one criterion, it tries to search for models on the furthest pareto front of multiple objective functions. In this attempt we consider the previous search problem as a multi-objective optimization problem. The first objective is to create networks that satisfy the size constraint. The objective function follows from the previous attempt: `min(-n_params/1e6, 5)`. The purpose of this objective is to select against networks larger than 5M but is neutral towards all networks smaller than 5M.

The second objective is the final validation accuracy. For networks with sizes larger than 5M, we don't train them and assign a criterion of 0. This, admittedly, is a less than perfect way to handle the bigger sized models, but is necessary since we are computationally constrained.

### Regularization
#### Data augmentation
We use a random selection of RandAugment and AutoAugment that are trained on CIFAR10. For each batch we sample a Bernouli distribution and if it's 1, we use RandAugment; if it's 0, we use AutoAugment. Near the end of the search, we find that despite AutoAugment being tailored for CIFAR10 and RandAugment being completely random, RandAugment has a significantly stronger presence compared to AutoAugment.

We also let the network choose randomly between cutmix and mixup. We find that our network tends to use cutmix significantly more than mixup.

#### Drop block
Instead of using standard dropouts for the regularization of activations, we use dropblock. Drop block is similar to drop out but the difference is that it drops an entire block around some randomly selected point in the feature map, instead of independently along all dimensions. We allow the network to search for the sizes being dropped for each layer.

#### Stochastic weight averaging
Stochastic weight averaging, or SWA for short, is a method that attempts to reduce overfitting by averaging steps in optimization. We observe it helping with the performance quite a lot.

#### Label smoothing
Label smoothing is a simple yet effective way to reduce overfitting. Instead of using 1-hot target when training, we use a linear combination of that and a uniform target. This penalizes the network for being over-confident and therefore reduces overfitting. 


