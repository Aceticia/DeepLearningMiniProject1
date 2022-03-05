import optuna

"""
TODOs:
1. Add early stopping
2. Add importance sampling
3. Add Learning rate decay
"""

# Generic hyperparameters
min_layers = 3     # Minimum layers
max_layers = 7     # Maximum layers
min_b = 2          # Minimum blocks per layer
max_b = 10         # Maximum blocks per layer
min_h = 32         # Minimum channels per block
max_h = 512        # Maximum channels per block
min_w = 1          # Minimum kernel size (1+2x)
max_w = 9          # Maximum kernel size (1+2x)
min_w_s = 1        # Minimum skip connection kernel size (1+2x)
max_w_s = 9        # Maximum skip connection kernel size (1+2x)
min_avg_w = 2      # Minimum average pooling kernel size (1+2x)
max_avg_w = 9      # Minimum average pooling kernel size (1+2x)

# Regularizer
max_weight_decay = 0
min_weight_decay = 1e-2

# Learning related
max_lr = 1e-2
min_lr = 1e-6
max_beta_1 = 1-1e-10
min_beta_1 = 0.5
max_beta_2 = 1-1e-10
min_beta_2 = 0.5
opt = ['AdamW', 'Adam']

# Others
reps = 5

# Objective func
def objective(trial):
    d = []

    # Global hyperparams
    d['n_layers'] = trial.suggest_int('n_layers', min_layers, max_layers)
    d['lr'] = trial.suggest_float('lr', min_lr, max_lr, log=True)
    d['beta_1'] = trial.suggest_float('beta_1', min_beta_1, max_beta_1, log=True)
    d['beta_2'] = trial.suggest_float('beta_2', min_beta_2, max_beta_2, log=True)
    d['optimizer'] = trial.suggest_categorical('optimizer', opt)
    d['weight_decay'] = \
        trial.suggest_float('weight_decay', min_weight_decay, max_weight_decay, log=True)
    d['average_pool_kernel_size'] = \
        trial.suggest_int('avg_kernel_size', min_avg_w, max_avg_w, step=2)

    # Local hyperparams
    d['blocks'] = []
    d['n_channels'] = []
    d['kernel_sizes'] = []
    d['skip_kernel_sizes'] = []
    for layer_idx in range(d['n_layers']):
        d['blocks'].append(
            trial.suggest_int(f'layer{layer_idx}_nblocks', min_b, max_b))
        d['n_channels'].append(
            trial.suggest_int(f'layer{layer_idx}_nhidden', min_h, max_h))
        d['kernel_sizes'].append(
            trial.suggest_int(f'layer{layer_idx}_kernel_size', min_w, max_w, step=2))
        d['skip_kernel_sizes'].append(
            trial.suggest_int(f'layer{layer_idx}_skip_kernel_size', min_w, max_w, step=2))
   
    # Evaluate model performance
    test_accs = []
    for _ in range(reps):
        # Instantiate model
        model = ResNet(d)

        # If model parameter count > 5M, return a bad value
        if sum(p.numel() for p in model.parameters() if p.requires_grad) > 5e6:
            return -1

        # Run train and test
        test_accs.append(train_and_test(model, d))

    return sum(test_accs)/len(test_accs)

# Create study
study = optuna.create_study(
    direction='maximize',
    study_name='DL2022',
    storage='sqlite:///database.db',
    load_if_exists=True,
    pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=1000, timeout=60000)