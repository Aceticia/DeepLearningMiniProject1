import pytorch_lightning as pl
from torchmetrics.functional import accuracy

import torch
import torch.nn as nn
import torch.nn.functional as F

from dropblock import DropBlock2D, LinearScheduler
from pytorch_lightning_spells.losses import MixupSoftmaxLoss


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        stride=1,
        kernel_size=3,
        skip_kernel_size=1,
        drop_prob=0.2,
        block_size=3,
    ):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=skip_kernel_size,
                    padding=skip_kernel_size // 2,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(pl.LightningModule):
    def __init__(self, **d):
        super(ResNet, self).__init__()

        self.save_hyperparameters()

        block = BasicBlock
        num_classes = 10
        self.d = d
        self.in_planes = d["n_channels"]

        self.drops = nn.ModuleList()
        for drop_prob, block_size in d["dropblock"]:
            self.drops.append(
                LinearScheduler(
                    DropBlock2D(drop_prob=drop_prob, block_size=block_size),
                    start_value=0.0,
                    stop_value=drop_prob,
                    nr_steps=1000,
                )
            )

        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_planes)

        self.layers = nn.ModuleList()
        for layer in range(d["n_layers"]):
            self.layers.append(
                self._make_layer(
                    block,
                    d["n_channels"] * 2 ** layer,
                    d["blocks"][layer],
                    kernel_size=d["kernel_sizes"][layer],
                    skip_kernel_size=d["skip_kernel_sizes"][layer],
                    stride=1 if layer == 1 else 2,
                )
            )

        self.linear = nn.Linear(d["n_channels"] * 2 ** layer, num_classes)
        self.loss = MixupSoftmaxLoss(label_smooth_eps=d["label_smooth"])

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.d["lr"], weight_decay=self.d["weight_decay"]
        )

        return optim

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, **kwargs))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer, drop in zip(self.layers, self.drops):
            drop.step()
            out = drop(layer(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.linear(out)

    # Training step, eval step, and val step from
    # https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/cifar10-baseline.html
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        # Use label smoothing mixup loss
        loss = self.loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = F.log_softmax(self(x), dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")


if __name__ == "__main__":
    checkpoint_path = ""
    loaded_model = ResNet.load_from_checkpoint(checkpoint_path)
    print(loaded_model)
