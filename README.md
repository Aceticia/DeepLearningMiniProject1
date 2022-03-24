# DeepLearningMiniProject1
## Replicating the result
### Installing environment
Install the conda environment included, and use `conda activate dl2022` to activate it.

In case this environment can't be installed, these are the essential packages you need to get a model:
1. `pytorch`, `torchvision`
2. `pytorch-lightning`
3. `dropblock` (from pip)

These are the packages you need to run the search again:
1. All of the above;
2. `optuna`

### Run hyperparameter search
Run `python optuna_script.py` to do a hyperparameter search. You can repeat this step for as many times as you want, this will give incrementally better results (possibly diminishing) as long as you keep the record file `storage/multi.db`.

### Load model and test performance of the model
First import the ResNet from `ResNet.py`. Then do `ResNet.load_from_checkpoint("./storage/best_checkpoint.ckpt")`.
