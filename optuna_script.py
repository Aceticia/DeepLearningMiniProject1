import wandb

import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import random_split, DataLoader

import albumentations as A
import albumentations.pytorch as P
from data_augmentation.dataset import Cifar10SearchDataset

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from project1_model import ResNet


# Generic hyperparameters
min_layers = 4     # Minimum layers
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
max_avg_w = 3      # Minimum average pooling kernel size (1+2x)

# Regularizer
max_weight_decay = 1e-2
min_weight_decay = 1e-6

# Data augmentations
augs = [None]+list(range(0, 40, 10))

# Learning related
max_lr = 1e-2
min_lr = 1e-6


def objective(trial):
    d = {}

    # Global hyperparams
    d['n_layers'] = trial.suggest_int('n_layers', min_layers, max_layers)
    d['lr'] = trial.suggest_float('lr', min_lr, max_lr, log=True)
    d['optimizer'] = 'AdamW'
    d['weight_decay'] = \
        trial.suggest_float('weight_decay', min_weight_decay,
                            max_weight_decay, log=True)
    d['average_pool_kernel_size'] = \
        trial.suggest_int('avg_kernel_size', min_avg_w, max_avg_w)

    # Local hyperparams
    d['blocks'] = []
    d['n_channels'] = []
    d['kernel_sizes'] = []
    d['skip_kernel_sizes'] = []
    for layer_idx in range(d['n_layers']):
        d['blocks'].append(
            trial.suggest_int(f'layer{layer_idx}_nblocks', min_b, max_b))
        d['n_channels'].append(
            trial.suggest_int(f'layer{layer_idx}_nhidden',
                              min_h, max_h, log=True))
        d['kernel_sizes'].append(
            trial.suggest_int(f'layer{layer_idx}_kernel_size',
                              min_w, max_w, step=2))
        d['skip_kernel_sizes'].append(
            trial.suggest_int(f'layer{layer_idx}_skip_kernel_size',
                              min_w, max_w, step=2))

    # Instantiate model
    model = ResNet(**d)

    # If model parameter count > 5M, return a bad value
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if param_count > 5e6:
        return -param_count/1e6

    # Load augmentation policy
    type_train_aug = trial.suggest_categorical('type_aug', augs)
    if type_train_aug is None:
        train_transform = A.Compose([
            A.augmentations.transforms.Normalize(
                (0.4914, 0.4821, 0.4465), (0.2469, 0.2430, 0.2610)),
            P.transforms.ToTensorV2()])
    else:
        train_transform = A.load(
            f"./data_augmentation/outputs/2022-03-07/20-57-53/policy/epoch_{type_train_aug}.json")
    test_transform = A.Compose([
        A.augmentations.transforms.Normalize(
            (0.4914, 0.4821, 0.4465), (0.2469, 0.2430, 0.2610)),
        A.pytorch.transforms.ToTensorV2()])

    # Create dataset
    train_dataset = Cifar10SearchDataset(root='~/data/CIFAR10',
                                         transform=train_transform, train=True)
    test_dataset = Cifar10SearchDataset(root='~/data/CIFAR10',
                                        transform=test_transform, train=False)

    # Split train dataset into train and test. Use 0.2 as ratio.
    val_len = len(train_dataset) // 5
    train_dataset, val_dataset = random_split(
        train_dataset, [len(train_dataset)-val_len, val_len])
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
    wandb_logger = WandbLogger(project="DLProject1", name=str(trial_id))
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=50,
        gpus=1,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val_acc"),
            ModelCheckpoint(filepath=f"./outputs/checkpoints/{trial_id}.pt",
                            monitor="val_loss")
        ],
    )
    trainer.fit(model, train_dataloader=train_loader,
                val_dataloaders=val_loader)
    trainer.test(model, test_dataloaders=test_loader)
    wandb.finish()

    return trainer.callback_metrics["val_acc"].item()


if __name__ == "__main__":
    # Create study
    study = optuna.create_study(
        direction='maximize',
        study_name='DL2022',
        storage='sqlite:///optuna_record.db',
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner())
    study.optimize(objective, n_trials=100000000000, timeout=60000)
