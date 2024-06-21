import torch
import torch.nn as nn


def act(act_fun='LeakyReLU'):
    """Easy selection of activation function by passing string or
    module (e.g. nn.ReLU)

    Parameters
    ----------
    act_fun :obj:`str`
        Activation function

    """
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        elif act_fun == 'ReLU':
            return nn.ReLU()
        elif act_fun == 'Tanh':
            return nn.Tanh()
        else:
            assert False
    else:
        return act_fun()


def downsample(stride=2, downsample_mode='max'):
    """Downsampling

    Parameters
    ----------
    stride :obj:`int`
        Stride
    downsample_mode :obj:`str`
        Downsampling mode (`avg`: average pooling, `max`:max pooling)

    Returns
    -------
    downsampler : :obj:`nn.Module`
        Downsampling layer

    """
    if downsample_mode == 'avg':
        downsampler = nn.AvgPool2d(stride, stride)
    elif downsample_mode == 'max':
        downsampler = nn.MaxPool2d(stride, stride)
    else:
        assert False
    return downsampler


def Conv2d_Block(in_f, out_f, kernel_size, stride=1, bias=True, bnorm=True,
                 act_fun='LeakyReLU', dropout=None):
    """2d Convolutional Block (conv, dropout, batchnorm, activation)

    Parameters
    ----------
    in_f : :obj:`int`
        Input channels
    out_f : :obj:`int`
        Output channels
    kernel_size : :obj:`int`
        Kernel size
    stride : :obj:`int`, optional
        Convolution stride
    bias : :obj:`int`, optional
        Convolution bias
    bnorm : :obj:`str`, optional
        Batch normalization
    act_fun : :obj:`str`, optional
        Activation function
    dropout : :obj:`float`, optional
        Dropout ratio

    Returns
    -------
    block : :obj:`nn.Sequential`
        Combined sequential network

    """
    to_pad = int((kernel_size - 1) / 2)  # to mantain input size
    block = [nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias), ]
    if dropout is not None:
        block.append(nn.Dropout2d(dropout))
    if bnorm:
        block = block + [nn.BatchNorm2d(out_f), ]
    if act_fun is not None:
        block = block + [act(act_fun), ]
    block = nn.Sequential(*block)
    return block


class ResNetBlock(nn.Module):
    """Residual Block (See https://towardsdatascience.com/residual-network-implementing-resnet-a7da63c7b278)

    Parameters
    ----------
    in_f : :obj:`int`
        Input channels
    out_f : :obj:`int`
        Output channels
    kernel_size : :obj:`int`
        Kernel size
    stride : :obj:`int`, optional
        Convolution stride
    act_fun : :obj:`str`, optional
        Activation function
    expansion : :obj:`int`, optional
        Expansion of channel number
    dropout : :obj:`float`, optional
        Dropout ratio
    use_bn : :obj:`bool`, optional
        Batch normalization

    """
    def __init__(self, in_f, out_f, kernel_size, stride=1, act_fun='LeakyReLU', expansion=1,
                 dropout=None, use_bn=True):

        super(ResNetBlock, self).__init__()
        self.blocks = nn.Sequential(
            Conv2d_Block(in_f, out_f, kernel_size, stride=stride,
                         bias=not use_bn, bnorm=use_bn, act_fun=act_fun),
            Conv2d_Block(out_f, out_f * expansion, kernel_size,
                         bias=not use_bn, bnorm=use_bn, act_fun=None),
        )
        self.shortcut = Conv2d_Block(in_f, out_f * expansion, kernel_size=1,
                                     stride=stride, bias=False, bnorm=use_bn,
                                     act_fun=None)
        self.accfun = act(act_fun)
        self.dropout = nn.Dropout2d(dropout) if dropout is not None else None
        self.bnorm = nn.BatchNorm2d(out_f * expansion) if use_bn is not None else None

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.accfun(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.bnorm is not None:
            x = self.bnorm(x)
        return x


class ContractingBlock_Res(nn.Module):
    """Contracting block

    Single block in contracting path composed of one ResNet block preceded by a max pool operation.
    We allow also to optionally include a batch normalization and dropout step.

    Parameters
    ----------
    input_channels : :obj:`int`
        Number of input channels
    output_channels : :obj:`int`
        Number of output channels
    use_dropout : :obj:`bool`, optional
        Add dropout
    use_bn : :obj:`bool`, optional
        Add batch normalization
    downstride : :obj:`int`, optional
        Downsampling stride

    """
    def __init__(self, input_channels, output_channels, use_dropout=None,
                 use_bn=True, downstride=2):
        super(ContractingBlock_Res, self).__init__()
        # Res block
        self.lay = ResNetBlock(input_channels,
                               output_channels,
                               kernel_size=3,
                               stride=1,
                               act_fun='LeakyReLU',
                               expansion=1,
                               dropout=use_dropout,
                               use_bn=use_bn)
        # Downsampling block
        self.downsample = downsample(stride=downstride, downsample_mode='max')

    def forward(self, x):
        x = self.downsample(x)
        x = self.lay(x)
        return x


class ExpandingBlock_Res(nn.Module):
    """Expanding block

    Single block in expanding path composed of one ResNet block preceded by an upsampling operation.
    We allow also to optionally include a batch normalization and dropout step.

    Parameters
    ----------
    input_channels : :obj:`int`
        Number of input channels
    output_channels : :obj:`int`
        Number of output channels
    skip_channels : :obj:`int`
        Number of skip connection channels
    use_dropout : :obj:`bool`, optional
        Add dropout
    use_bn : :obj:`bool`, optional
        Add batch normalization
    upstride : :obj:`int`, optional
        Upsampling stride

    """
    def __init__(self, input_channels, output_channels, skip_channels=0,
                 use_dropout=None, use_bn=True, upstride=2):
        super(ExpandingBlock_Res, self).__init__()
        # Upsampling block
        self.upsample = nn.Upsample(scale_factor=upstride, mode='bilinear')
        # Res block
        self.lay = ResNetBlock(input_channels + skip_channels,
                                output_channels,
                                kernel_size=3,
                                stride=1,
                                act_fun='LeakyReLU',
                                expansion=1,
                                dropout=use_dropout,
                                use_bn=use_bn)

    def forward(self, x, skip_con_x):
        x = self.upsample(x)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.lay(x)
        return x


class FeatureMapBlock(nn.Module):
    """Feature Map block

    Final layer of U-Net which restores for the output channel dimensions to those of the input (or any other size)
    using a 1x1 convolution.

    Parameters
    ----------
    input_channels : :obj:`int`
        Input channels
    output_channels : :obj:`int`
        Output channels

    """

    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResUNet(nn.Module):
    """UNet architecture with Residual blocks

    UNet architecture composed of a series of contracting blocks followed by expanding blocks.

    Parameters
    ----------
    input_channels : :obj:`int`, optional
        Number of input channels
    output_channels : :obj:`int`, optional
        Number of output channels
    hidden_channels : :obj:`int`, optional
        Number of hidden channels
    levels : :obj:`int`, optional
        Number of levels in contracting and expanding paths
    subsampling : :obj:`list`, optional
        Submsampling in different dimensions
    use_bn : :obj:`bool`, optional
        Add batch normalization
    use_dropout : :obj:`bool`, optional
        Add dropout

    """
    def __init__(self, input_channels=1, output_channels=1, hidden_channels=64, levels=2,
                 subsampling=[2, 2], use_bn=True, use_dropout=None):
        super(ResUNet, self).__init__()
        self.levels = levels
        self.subsampling1 = [subsampling[0], ] * self.levels if isinstance(subsampling[0], int) else subsampling[0]
        self.subsampling2 = [subsampling[1], ] * self.levels if isinstance(subsampling[1], int) else subsampling[1]

        # First layer
        self.first = ResNetBlock(input_channels, hidden_channels, 3,
                                 stride=1, act_fun='LeakyReLU', expansion=1)

        # Downsampling layers
        self.contract = []
        for level, sub1, sub2 in zip(range(levels), self.subsampling1, self.subsampling2):
            self.contract.append(ContractingBlock_Res(hidden_channels * (2 ** level),
                                                      hidden_channels * (2 ** (level + 1)),
                                                      use_bn=use_bn, use_dropout=use_dropout,
                                                      downstride=(sub1, sub2)))
        self.contracts = nn.Sequential(*self.contract)

        # Upsampling layers
        self.expand = []
        for level, sub1, sub2 in zip(range(levels), self.subsampling1[::-1], self.subsampling2[::-1]):
            self.expand.append(ExpandingBlock_Res(hidden_channels * (2 ** (levels - level)),
                                                  hidden_channels * (2 ** (levels - level - 1)),
                                                  skip_channels=hidden_channels * (2 ** (levels - level - 1)),
                                                  use_bn=use_bn, use_dropout=use_dropout,
                                                  upstride=(sub1, sub2)))
        self.expands = nn.Sequential(*self.expand)

        # Final layers
        self.final = FeatureMapBlock(hidden_channels, output_channels)

    def forward(self, x):
        x = self.first(x)
        xenc = [x, ]
        for level in range(self.levels):
            x = self.contract[level](x)
            if level < self.levels - 1:
                xenc.append(x)
        for level in range(self.levels):
            x = self.expand[level](x, xenc[self.levels - level - 1])
        x = self.final(x)
        return x

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)