# DeepLearningMiniProject1
## TODOs
1. OPTUNA script for optimization (Xujin)
    a. Input: Serch space
    b. Output: A set of hyperparameter
2. Script to train network: 
    1). take in hyperparameter (Boyu)
        a. Input: Set of hyperparameter, in addition to:
            i. Dropout, blocked dropout, etc.
        b. Output: A model
    2). output test performance (Yufeng)
        a. Input: Subset of hyperparameter, The model
            i. Learning rate scheduler
            ii. Choice of optimizer
            iii. Weight decay
        b. Output: Test accuracy & trained model
3. How to contrain search space

## Recreating a result
### Installing environment
Install the conda environment included, and use `conda activate dl2022` to activate it.
### Data augmentation search
Run `autoalbument-search --config-dir ./data_augmentation/` to do a search of data augmentation based on the dataset. This will create the augmentation policy in `./data_augmentation/outputs/<current_date>/<current_time>/`.
### Run hyperparameter search
Run `python optuna_script.py` to do a hyperparameter search. You can repeat this step for as many times as you want, this will give incrementally better results (possibly diminishing) as long as you keep the record file `optuna_records.db`.
