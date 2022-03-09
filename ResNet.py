import torch.nn as nn
import torch.nn.functional as F


class MiniProj1Model(nn.Module):
    """
    The MiniProj1Model class is made specific for the mini project 1. This model
    is a residual network with customizable residual layers to be optimized on
    CIFAR-10 benchmark.

    The input layer contains a single convolutional layer with f1 x f1 sized
    kernel and c1 output channels, and followed by batch normalization and ReLU.
    If dropout is enabled, a dropout layer is added at the back.

    The input layer is followed by customised residual layer(s). A P x P average
    pooling layer is followed the residual layer(s). The following parameters
    are used to customise the residual layer(s)

    N - Number of Residual Layers
    B_i -  Number of Residual blocks in Residual Layer i
    C_1 - Number of channels in Residual Layer 1
    F_i - Convolving kernel size in the residual layer i
    K_i - Skip connection kernel size in the residual layer i
    P - Average pool kernel size

    Each layer uses 2 stride when covolving. The output channels of each layer
    doubles every layer

    The output layers contains a fully connected layer with ReLu and softmax.
    """
    IMAGE_CHANNEL = 3
    IMAGE_SIZE = 32
    NUM_CLASSES = 10

    def __init__(self, n, b, c1, f, k, p, dropout_prob=0.0):
        """
        :param n(int): N - Number of Residual Layers
        :param b(list[int]): B_i -  Number of Residual blocks in Residual Layer i
        :param c1(int): C-1 - Number of channels in Residual Layer 1
        :param f(list[int]): F_i - Convolving kernel size in the residual layer i
        :param k(list[int]): K_i - Skip connection kernel size in the residual layer i
        :param p(int): P - Average pool kernel size
        :param dropout_prob(float): The dropout probability for an element to be zero-ed. Default: 0.0
        """
        super(MiniProj1Model, self).__init__()
        # verifying input parameters
        assert n <= 5, "Too many residual layers"
        assert len(b) == n, "List length doesn't match with the number of residual layers"
        assert len(f) == n, "List length doesn't match with the number of residual layers"
        assert len(k) == n, "List length doesn't match with the number of residual layers"
        assert self.IMAGE_SIZE // 2 ** n >= p, "Pooling kernel size too small"
        assert dropout_prob <= 0 and dropout_prob <= 1, "The dropout probability shout be in [0, 1]"

        # constructing the first convolutional layer
        self.conv1 = nn.Conv2d(self.IMAGE_CHANNEL, c1, kernel_size=f[0], stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.dropout1 = nn.Dropout2d(p=dropout_prob)

        # constructing the residual layers
        layer_config = []
        in_channel = c1
        out_channel = c1 * 2
        for i in range(n):
            config = {"in_channels": in_channel, "out_channels": out_channel, "num_blocks": b[i], "stride": 2,
                      "kernel_size": f[i], "shortcut_kernel": k[i]}

            layer_config.append(config)
            in_channel = out_channel
            out_channel *= 2
        self.res_net_layers = ResNet(layer_config, batch_norm=True, dropout_prob=dropout_prob)

        # constructing the fully connected layer
        self.pooling_size = p
        fc1_size = (c1 * 2**n) * (self.IMAGE_SIZE // 2**n // p) ** 2
        self.fc1 = nn.Linear(fc1_size, self.NUM_CLASSES)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.res_net_layers(out)
        out = F.avg_pool2d(out, self.pooling_size)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.softmax(out)
        return out


class ResNet(nn.Module):
    """
    The ResNet class represent a neural network with only residual layer(s).
    Each residual layer can have one or more residual blocks. The kernel
    size, dropout probability, and batch normalization is the same across all
    blocks. Only the first block inherent the stride; the following blocks
    will have stride = 1.
    """
    def __init__(self, layers_config, batch_norm=False, dropout_prob=0.0):
        """
        The class constructor takes a list of dictionaries, each dictionary
        contains the parameters for constructing a residual layer. Refer to
        _make_layer() function documentation for the details of the parameters.
        :param layers_config(List(dict)): List of dictionaries. Each dictionary contains the parameter for each layer
        :param batch_norm(boolean): If true a batch normalization layer is added after each convolutional layer.
        Default: False
        :param dropout_prob(float): The dropout probability for an element to be zero-ed. Default: 0.0
        """
        super(ResNet, self).__init__()
        layers = []
        for config in layers_config:
            layers.append(self._make_layer(**config, batch_norm=batch_norm, dropout_prob=dropout_prob))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def _make_layer(in_channels, out_channels, num_blocks, kernel_size=3, stride=2, shortcut_kernel=None,
                    batch_norm=False, dropout_prob=0.0):
        """
        Construct a residual layer base on the parameters.
        :param in_channels(int): Number of channels in the input
        :param out_channels(int): Number of channels in the output
        :param num_blocks(int): Number of residual block(s) in this layer
        :param kernel_size(int): The kernel size of the convolving kernel in the convolutional layer. Default: 3
        :param stride(int): Stride of the convolution. Default: 2
        :param shortcut_kernel(int): The kernel size of the convoling kernel in the shortcut when projecting shortcut is
        used. Default: None
        :param batch_norm(boolean): If true a batch normalization layer is added after each convolutional layer.
        Default: False
        :param dropout_prob(float): The dropout probability for an element to be zero-ed. Default: 0.0
        :return(nn.Module): A neural network with only residual layer(s).
        """
        residual_layer = [
            _ResidualBlock(in_channels, out_channels, kernel_size, stride, shortcut_kernel, batch_norm, dropout_prob)]
        for i in range(1, num_blocks):
            residual_layer.append((_ResidualBlock(out_channels, out_channels, kernel_size, 1, shortcut_kernel,
                                                  batch_norm, dropout_prob)))

        return nn.Sequential(*residual_layer)


class _ResidualBlock(nn.Module):
    """
    The ResidualBlock class represent a single residual block in the ResNet.
    The structure follows the equation F(x) + x. The block contains two
    convolutional layers, which is the residual part of the equation
    and one shortcut connection, which is the identy part of the equation.

    The class constructor automatically adjust the shortcut connection if the
    dimension of the input and the output are different. On default, zero-padded
    is used in the shortcut connection to adjust increase the channels size. If
    shortcut_kernel is given, projecting shortcut is used. If the dimension of
    the input and output are the same (in_channels = out_channels and
    stride == 1), shortcut_kernel is ignored.

    If batch normalization is enabled, there will be a normalization layer added
    after each convolutional layer. If projecting shortcut is used, a batch
    normalization layer is also added after the convolutional layer in the
    shortcut connection.

    The class assume all kernels are square kernels. Therefore, all kernel size
    should be in integers, tuple is not allowed.
    All activation functions are using ReLU.
    Bias term is assumed to be turned off for all convolutional layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 shortcut_kernel=None, batch_norm=False, dropout_prob=0.0):
        """
        :param in_channels(int): Number of channels in the input
        :param out_channels(int): Number of channels in the output
        :param kernel_size(int): The kernel size of the convolving kernel in the convolutional layer. Default: 3
        :param stride(int): Stride of the convolution. Default: 1
        :param shortcut_kernel(int): The kernel size of the convoling kernel in the shortcut when projecting shortcut is
        used. Default: None
        :param batch_norm(boolean): If true a batch normalization layer is added after each convolutional layer.
        Default: False
        :param dropout_prob(float): The dropout probability for an element to be zero-ed. Default: 0.0
        """
        super(_ResidualBlock, self).__init__()

        padding = kernel_size // 2
        conv_layer1 = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]
        conv_layer2 = [nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)]
        shortcut_link = []

        # adjusting the skip link output dimension
        if (in_channels != out_channels) or (stride != 1):
            if shortcut_kernel is None:
                # zero-padded shortcut
                raise NotImplemented("Zero padding method is not implemented")
            else:
                # projecting shortcut
                shortcut_padding = shortcut_kernel // 2
                shortcut_link.append(nn.Conv2d(in_channels, out_channels, kernel_size=shortcut_kernel, stride=stride,
                                               padding=shortcut_padding, bias=False))
                if batch_norm:
                    shortcut_link.append(nn.BatchNorm2d(out_channels))

        if batch_norm:
            conv_layer1.append(nn.BatchNorm2d(out_channels))
            conv_layer2.append(nn.BatchNorm2d(out_channels))

        self.conv1 = nn.Sequential(*conv_layer1)
        self.conv2 = nn.Sequential(*conv_layer2)
        self.shortcut = nn.Sequential(*shortcut_link)
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        # F(x)
        fx = F.relu(self.conv1(x))
        fx = self.dropout(fx)
        fx = self.conv2(fx)
        # x
        identy = self.shortcut(x)
        identy = self.dropout(identy)
        # F(x) + x
        out = fx + identy
        out = F.relu(out)
        out = self.dropout(out)
        return out
