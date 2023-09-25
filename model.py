# References:
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, relu=True):
        super().__init__()

        self.relu = relu

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="reflect",
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.relu:
            x = torch.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels=256, padding=1):
        super().__init__()

        # self.conv1 = nn.Conv2d(
        #     channels, channels, kernel_size=3, stride=1, padding=padding, padding_mode="reflect",
        # )
        # self.conv2 = nn.Conv2d(
        #     channels, channels, kernel_size=3, stride=1, padding=padding, padding_mode="reflect",
        # )
        self.conv1 = ConvNormRelu(channels, channels, kernel_size=3, stride=1, padding=padding, relu=True)
        self.conv2 = ConvNormRelu(channels, channels, kernel_size=3, stride=1, padding=padding, relu=True)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class TransConvNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, padding, output_padding):
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=padding,
            output_padding=output_padding,
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = torch.relu(x)
        return x


# "Weights are initialized from a Gaussian distribution $N(0, 0.02).$"
def _init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.InstanceNorm2d)):
            m.weight.data.normal_(0, 0.02)


# "This network contains three convolutions, several residual blocks, two fractionally-strided
# convolutions with stride $\frac{1}{2}$, and one convolution that maps features to RGB. We
# use 6 blocks for 128 × 128 images and 9 blocks for 256 × 256 and higher-resolution training images."
# "Let 'c7s1-k' denote a 7 × 7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1.
# 'dk' denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2. Reflection
# padding was used to reduce artifacts. 'Rk' denotes a residual block that contains two 3 × 3
# convolutional layers with the same number of filters on both layer. 'uk' denotes a 3 × 3
# fractional-strided-Convolution-InstanceNorm-ReLU layer with k filters and stride $\frac{1}{2}$."
# "The network with 6 residual blocks consists of: 'c7s1-64, d128, d256, R256, R256, R256, R256, R256,
# R256, u128, u64, c7s1-3'."
# "The network with 9 residual blocks consists of: 'c7s1-64, d128, d256, R256, R256, R256, R256, R256,
# R256, R256, R256, R256, u128, u64, c7s1-3'."
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block1 = ConvNormRelu(3, 64, kernel_size=7, stride=1, padding=3, relu=True) # "'c7s1-64'"
        self.conv_block2 = ConvNormRelu(64, 128, kernel_size=3, stride=2, padding=1, relu=True) # "'d128'"
        self.conv_block3 = ConvNormRelu(128, 256, kernel_size=3, stride=2, padding=1, relu=True) # "'d256'"
        self.resid_blocks = nn.Sequential(*[ResidualBlock() for _ in range(1, 9 + 1)]) # "'R256'"
        self.trans_conv_block1 = TransConvNormRelu(
            256, 128, padding=1, output_padding=1,
        ) # "'u128'"
        self.trans_conv_block2 = TransConvNormRelu(
            128, 64, padding=1, output_padding=1,
        ) # "'u64'"
        self.conv_block4 = ConvNormRelu(64, 3, kernel_size=7, stride=1, padding=3, relu=False) # "'c7s1-3'"

        _init_weights(self)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.resid_blocks(x)
        x = self.trans_conv_block1(x)
        x = self.trans_conv_block2(x)
        x = self.conv_block4(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        instance_norm=True,
    ):
        super().__init__()

        self.instance_norm = instance_norm

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False,
        )
        if instance_norm:
            self.norm = nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=False)

    def forward(self, x):
        x = self.conv(x)
        if self.instance_norm:
            x = self.norm(x)
        # "We use leaky ReLUs with a slope of 0.2."
        x = F.leaky_relu(x, 0.2)
        return x


# "We use 70 × 70 PatchGAN. Let 'Ck' denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer
# with k filters and stride 2. The discriminator architecture is: 'C64-C128-C256-C512'"
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # "We do not use InstanceNorm for the first 'C64' layer."
        self.conv_block1 = ConvBlock(3, 64, instance_norm=False) # "'C64'"
        self.conv_block2 = ConvBlock(64, 128, instance_norm=True) # "'C128'"
        self.conv_block3 = ConvBlock(128, 256, instance_norm=True) # "'C256'"
        self.conv_block4 = ConvBlock(256, 512, instance_norm=True) # "'C512'"
        # "After the last layer, we apply a convolution to produce a 1-dimensional output."
        self.conv_block5 = nn.Conv2d(512, 1, kernel_size=1)

        _init_weights(self)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = x.mean(dim=(2, 3))
        return x


if __name__ == "__main__":
    img_size = 256
    x = torch.randn(2, 3, img_size, img_size)
    gen = Generator()
    disc = Discriminator()
    out = disc(x)
    out.shape
    
    out = gen(x)
    out.shape
