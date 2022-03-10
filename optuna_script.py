import wandb
import optuna

import torchvision.datasets as D
import torchvision.transforms as T

from pytorch_lightning_spells.callbacks import CutMixCallback, MixUpCallback, RandomAugmentationChoiceCallback

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from torch.utils.data import random_split, DataLoader
from ResNet import ResNet

# Select which approach to take, one of revolution, multi, and nsga
search_approach = 'multi'

# Generic hyperparameters
min_b = 2          # Minimum blocks per layer
max_b = 10         # Maximum blocks per layer
min_h = 32         # Minimum channels per block
max_h = 128        # Maximum channels per block
min_w = 1          # Minimum kernel size (1+2x)
max_w = 9          # Maximum kernel size (1+2x)
min_w_s = 1        # Minimum skip connection kernel size (1+2x)
max_w_s = 9        # Maximum skip connection kernel size (1+2x)

# Regularizer
max_weight_decay = 1e-2
min_weight_decay = 1e-6

# Augmix, randaug, and more
type_aug = {
    "AutoAugment": (
        T.AutoAugment,
        lambda t: T.AutoAugmentPolicy.CIFAR10),
    "RandAugment": (
        T.RandAugment,
        lambda t: t.suggest_int("randaug_num_ops", 2, 10)),
    "TrivialAugment": T.TrivialAugmentWide
}

# Whether to use mixup, cutmix and label smoothing
# https://pytorch-lightning-spells.readthedocs.io/en/latest/#augmentation
# https://pytorch-lightning-spells.readthedocs.io/en/latest/pytorch_lightning_spells.losses.html#pytorch_lightning_spells.losses.MixupSoftmaxLoss
mix_type = {
    'cutmix': CutMixCallback,
    'mixup': MixUpCallback,
}
label_smoothing_min = 0
label_smoothing_max = 0.5
alpha_min = 0
alpha_max = .8

# Dropblock
min_prob = 0.05
max_prob = 0.2
min_size = 1
max_size = 7

# Learning related
max_lr = 1e-2
min_lr = 1e-6


def objective(trial):
    d = {}

    # Global hyperparams
    d['label_smooth'] = trial.suggest_float(
        'label_smoothing', label_smoothing_min, label_smoothing_max)
    d['n_layers'] = 4
    d['lr'] = trial.suggest_float('lr', min_lr, max_lr, log=True)
    d['optimizer'] = 'AdamW'
    d['weight_decay'] = \
        trial.suggest_float('weight_decay', min_weight_decay,
                            max_weight_decay, log=True)
    # Local hyperparams
    d['blocks'] = []
    d['kernel_sizes'] = []
    d['skip_kernel_sizes'] = []
    d['dropblock'] = []
    d['n_channels'] = trial.suggest_int('nhidden', min_h, max_h)
    for layer_idx in range(d['n_layers']):
        d['blocks'].append(
            trial.suggest_int(f'layer{layer_idx}_nblocks', min_b, max_b))
        d['kernel_sizes'].append(
            trial.suggest_int(f'layer{layer_idx}_kernel_size',
                              min_w, max_w, step=2))
        d['skip_kernel_sizes'].append(
            trial.suggest_int(f'layer{layer_idx}_skip_kernel_size',
                              min_w, max_w, step=2))
        d['dropblock'].append(
            (trial.suggest_float(f'layer{layer_idx}_drop_prob', min_prob, max_prob),
             trial.suggest_int(f'layer{layer_idx}_drop_size', min_size, max_size))
        )

    # Sample aug
    aug_to_use = trial.suggest_categorical(
        "augment_type", list(type_aug.keys()))
    d["aug_to_use"] = aug_to_use

    # Sample mixup, snapmix and cutmix
    d['mixup_alpha'] = trial.suggest_float('mixup_alpha', alpha_min, alpha_max)
    d['cutmix_alpha'] = trial.suggest_float(
        'cutmix_alpha', alpha_min, alpha_max)
    d['mixup_p'] = trial.suggest_float('mixup_p', 0, .1)
    d['cutmix_p'] = 1-d['mixup_p']

    model = ResNet(**d)

    # If model parameter count > 5M, return a bad value
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if param_count > 5e6:
        return -param_count/1e6, 0
    else:
        params_penalty = -5

    # Sample augmentations
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(
            (0.4914, 0.4821, 0.4465), (0.2469, 0.2430, 0.2610)),
    ])

    # Generate augmentations
    aug = type_aug[aug_to_use]
    if isinstance(aug, tuple):
        aug = aug[0](aug[1](trial))
    else:
        aug = aug()
    train_transform = T.Compose([
        T.ToTensor(),
        aug,
        T.Normalize(
            (0.4914, 0.4821, 0.4465), (0.2469, 0.2430, 0.2610)),
    ])

    # Create dataset
    train_dataset = D.CIFAR10(root='~/data/CIFAR10',
                              transform=train_transform, train=True)
    test_dataset = D.CIFAR10(root='~/data/CIFAR10',
                             transform=test_transform, train=False)

    # Split train dataset into train and test. Use 0.2 as ratio.
    val_len = len(train_dataset) // 5
    train_dataset, val_dataset = random_split(
        train_dataset, [len(train_dataset)-val_len, val_len])

    # Val data should also use train transform
    val_dataset.transform = test_transform
    train_loader = DataLoader(
        train_dataset,
        batch_size=1024, num_workers=32, shuffle=True, pin_memory=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1024, num_workers=8, shuffle=False, pin_memory=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1024, num_workers=8, shuffle=False, pin_memory=True)

    # Run train and val
    trial_id = trial.number
    wbl = WandbLogger(
        project=f"debug_DLProject1_{search_approach}_training", name=str(trial_id))
    trainer = pl.Trainer(
        logger=wbl,
        max_epochs=100,
        gpus=1,
        callbacks=[
            ModelCheckpoint(filepath=f"./outputs/{search_approach}/checkpoints/{trial_id}.pt",
                            monitor="val_loss"),
            EarlyStopping(monitor="val_loss", patience=20),
            RandomAugmentationChoiceCallback([
                CutMixCallback(d['cutmix_alpha']),
                MixUpCallback(d['mixup_alpha'])],
                [d['cutmix_p'], d['mixup_p']]
            )

        ],
    )
    trainer.fit(model, train_dataloader=train_loader,
                val_dataloaders=val_loader)
    trainer.test(model, test_dataloaders=test_loader)
    wandb.finish()

    return params_penalty, trainer.callback_metrics["val_acc"].item()


if __name__ == "__main__":
    # Create sampler
    sampler = {
        "multi": optuna.samplers.NSGAIISampler,
        "revolution": optuna.samplers.CmaEsSampler
    }

    sampler_args = {
        "multi": {},
        "revolution": {
            "n_startup_trials": 200,
            "restart_strategy": 'ipop',
            "inc_popsize": 2
        }
    }

    # Optional params depending on strategy
    options = {
        "multi": {"directions": ['maximize', 'maximize']},
        "evolution": {"direction": 'maximize'}
    }

    # Create study
    study = optuna.create_study(
        **options[search_approach],  # Options specifically for the strategy
        study_name=f'DL2022_{search_approach}',
        storage=optuna.storages.RDBStorage(
            url=f"sqlite:///records/debug_{search_approach}.db",
            engine_kwargs={"connect_args": {"timeout": 500}}),
        sampler=sampler[search_approach](**sampler_args[search_approach]),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100000000000,
                   timeout=60000)
