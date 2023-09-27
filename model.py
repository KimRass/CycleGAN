# References:
    # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

# 논문과 공식 repository가 서로 다른 부분 또는 알기 어려운 부분이 많은데, 공식 repository를 기준으로 구현했습니다.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, padding_mode, activ):
        super().__init__()

        self.activ = activ

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode=padding_mode,
            bias=False,
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activ == "relu":
            x = torch.relu(x)
        elif self.activ == "tanh":
            x = torch.tanh(x)
        elif self.activ == "none":
            pass
        return x


# "'Rk' denotes a residual block that contains two 3 × 3 convolutional layers with the same number of filters
# on both layer."
class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = ConvNormRelu(256, 256, kernel_size=3, padding=1, padding_mode="reflect", activ="relu")
        self.conv2 = ConvNormRelu(256, 256, kernel_size=3, padding=1, padding_mode="reflect", activ="none")

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class TransConvNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, padding):
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=padding,
            output_padding=1,
            bias=False,
        )
        self.norm = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = torch.relu(x)
        return x


# "Weights are initialized from a Gaussian distribution $N(0, 0.02).$"
def _init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            m.weight.data.normal_(0, 0.02)


# "This network contains three convolutions, several residual blocks, two fractionally-strided
# convolutions with stride $\frac{1}{2}$, and one convolution that maps features to RGB. We
# use 6 blocks for 128 × 128 images and 9 blocks for 256 × 256 and higher-resolution training images."
# "The network with 6 residual blocks consists of: 'c7s1-64, d128, d256, R256, R256, R256, R256, R256,
# R256, u128, u64, c7s1-3'."
# "The network with 9 residual blocks consists of: 'c7s1-64, d128, d256, R256, R256, R256, R256, R256,
# R256, R256, R256, R256, u128, u64, c7s1-3'."
class Generator(nn.Module):
    def __init__(self, n_resid_blocks=9):
        super().__init__()

        # "Let 'c7s1-k' denote a 7 × 7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1."
        self.conv_block1 = ConvNormRelu(3, 64, kernel_size=7, padding=3, padding_mode="reflect", activ="relu") # "'c7s1-64'"
        # "'dk' denotes a 3 × 3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2. Reflection
        # padding was used to reduce artifacts."
        self.conv_block2 = ConvNormRelu(
            64, 128, kernel_size=3, stride=2, padding=1, padding_mode="zeros", activ="relu",
        ) # "'d128'"
        self.conv_block3 = ConvNormRelu(
            128, 256, kernel_size=3, stride=2, padding=1, padding_mode="zeros", activ="relu",
        ) # "'d256'"
        self.resid_blocks = nn.Sequential(
            *[ResidualBlock() for _ in range(n_resid_blocks)]
        ) # "'R256'"
        # "'uk' denotes a 3 × 3 fractional-strided-Convolution-InstanceNorm-ReLU layer with k filters and
        # stride $\frac{1}{2}$."
        self.trans_conv_block1 = TransConvNormRelu(256, 128, padding=1) # "'u128'"
        self.trans_conv_block2 = TransConvNormRelu(128, 64, padding=1) # "'u64'"
        # 논문에는 나와있지 않지만, $[-1, 1]$의 tensor를 이미지로 변환할 것이므로 activation function으로 tanh를
        # 사용하겠습니다.
        self.conv_block4 = nn.Conv2d(64, 3, kernel_size=7, padding=3, padding_mode="relection") # "'c7s1-3'"

        _init_weights(self)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.resid_blocks(x)
        x = self.trans_conv_block1(x)
        x = self.trans_conv_block2(x)
        x = self.conv_block4(x)
        x = torch.tanh(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm=True):
        super().__init__()

        self.norm = norm

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False,
        )
        if norm:
            self.norm = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        # "We use leaky ReLUs with a slope of 0.2."
        x = F.leaky_relu(x, negative_slope=0.2)
        return x


def get_receptive_field(out_channels, kernel_size, stride):
    return (out_channels - 1) * stride + kernel_size


# "We use 70 × 70 PatchGAN. Let 'Ck' denote a 4 × 4 Convolution-InstanceNorm-LeakyReLU layer
# with k filters and stride 2. The discriminator architecture is: 'C64-C128-C256-C512'"
# Pix2Pix의 discriminator와 architecture가 거의 동일합니다.
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # "We do not use InstanceNorm for the first 'C64' layer."
        self.conv_block1 = ConvBlock(3, 64, stride=2, norm=False) # "'C64'"
        self.conv_block2 = ConvBlock(64, 128, stride=2, norm=True) # "'C128'"
        self.conv_block3 = ConvBlock(128, 256, stride=2, norm=True) # "'C256'"
        self.conv_block4 = ConvBlock(256, 512, stride=1, norm=True) # "'C512'"
        # "After the last layer, we apply a convolution to produce a 1-dimensional output."
        self.conv = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

        _init_weights(self)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv(x)
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

    # x = torch.randn(1, 1, 3, 4)
    # pad = nn.ReflectionPad2d(2)
    # conv1 = nn.Conv2d(1, 1, kernel_size=2)
    # conv2 = nn.Conv2d(1, 1, kernel_size=2, padding=2, padding_mode="reflect")
    # conv2.weight.data = conv1.weight.data
    # conv2.bias.data = conv1.bias.data

    # torch.equal(conv1(pad(x)), conv2(x))

    x = torch.randn(1, 4, 3, 4)
    conv1 = nn.Conv2d(4, 4, kernel_size=2, bias=True)
    conv2 = nn.Conv2d(4, 4, kernel_size=2, bias=False)
    conv2.weight.data = conv1.weight.data
    # conv2.bias.data = conv1.bias.data
    norm = nn.InstanceNorm2d(4)

    torch.equal(norm(conv1(x)), norm(conv2(x)))
    norm(conv1(x))
    norm(conv2(x))