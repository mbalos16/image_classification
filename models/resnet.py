import torch
import torchvision.transforms as transforms


class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels=64, out_channels=256, kernel_size=3, stride=2):
        super(ResNetBlock, self).__init__()
        self.conv_1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
        )  # stride = 1 => Shape (b, 64, w, h) | # stride = 2 => Shape (b, 64, w/2, h/2)
        self.conv_2 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )  # stride = 2 => (b, 256, w/2, h/2) | stride = 1 =>    (b, 256, w/2, h/2)
        self.activation = torch.nn.ReLU()
        self.batch_norm_1 = torch.nn.BatchNorm2d(in_channels)
        self.batch_norm_2 = torch.nn.BatchNorm2d(in_channels)
        self.residual_conv = None
        if in_channels != out_channels or stride > 1:
            self.residual_conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
            )  # Shape: b, 256, w/2, h/2

    def forward(self, x):
        h = self.batch_norm_1(x)  # Shape: (batch_size, channels, 30, 30)
        h = self.conv_1(h)  # Shape: (batch_size, channels, 15, 15)
        h = self.activation(h)

        h = self.batch_norm_2(h)  # Shape: (batch_size, channels, 15, 15)
        h = self.conv_2(h)  # Shape: (batch_size, channels, 15, 15)
        h = self.activation(h)
        if self.residual_conv is not None:
            x = self.residual_conv(x)
        return x + h


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv_1 = torch.nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=2
        )
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.block_1 = ResNetBlock(
            in_channels=64, out_channels=64, kernel_size=3, stride=2
        )
        self.block_2 = ResNetBlock(
            in_channels=64, out_channels=128, kernel_size=3, stride=2
        )
        self.block_3 = ResNetBlock(
            in_channels=128, out_channels=256, kernel_size=3, stride=2
        )
        self.fully_connected = torch.nn.Linear(in_features=256, out_features=6)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        # Input size: (batch_size, 3, 128, 128)
        x = self.conv_1(x)  # Shape: (batch_size, 64, 61, 61)
        x = self.activation(x)
        x = self.max_pool(x)  # Shape: (batch_size, 64, 30, 30)

        x = self.block_1(x)  # Shape: (batch_size, 64, 15, 15)
        x = self.block_2(x)  # Shape: (batch_size, 128, 8, 8)
        x = self.block_3(x)  # Shape: (batch_size, 64, 4, 4)

        x = self.adaptive_avg_pool(x)  # Shape: (batch_size, 256, 1, 1)
        x = torch.squeeze(x)  # Shape: (batch_size, 256)
        x = self.fully_connected(x)  # Shape: (batch_size, 84)
        return x


# From notebook
class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels=64, out_channels=256, kernel_size=3, stride=2):
        super(ResNetBlock, self).__init__()
        self.conv_1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
        )  # stride = 1 => Shape (b, 64, w, h) | # stride = 2 => Shape (b, 64, w/2, h/2)
        self.conv_2 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )  # stride = 2 => (b, 256, w/2, h/2) | stride = 1 =>    (b, 256, w/2, h/2)
        self.activation = torch.nn.ReLU()
        self.batch_norm_1 = torch.nn.BatchNorm2d(in_channels)
        self.batch_norm_2 = torch.nn.BatchNorm2d(in_channels)
        self.residual_conv = None
        if in_channels != out_channels or stride > 1:
            self.residual_conv = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
            )  # Shape: b, 256, w/2, h/2

    def forward(self, x):
        h = self.batch_norm_1(x)  # Shape: (batch_size, channels, 30, 30)
        h = self.conv_1(h)  # Shape: (batch_size, channels, 15, 15)
        h = self.activation(h)

        h = self.batch_norm_2(h)  # Shape: (batch_size, channels, 15, 15)
        h = self.conv_2(h)  # Shape: (batch_size, channels, 15, 15)
        h = self.activation(h)
        if self.residual_conv is not None:
            x = self.residual_conv(x)
        return x + h


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv_1 = torch.nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=2
        )
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.block_1 = ResNetBlock(
            in_channels=64, out_channels=64, kernel_size=3, stride=2
        )
        self.block_2 = ResNetBlock(
            in_channels=64, out_channels=128, kernel_size=3, stride=2
        )
        self.block_3 = ResNetBlock(
            in_channels=128, out_channels=256, kernel_size=3, stride=2
        )
        self.fully_connected = torch.nn.Linear(in_features=256, out_features=6)
        self.activation = torch.nn.ReLU()

    def forward(self, x):
        # Input size: (batch_size, 3, 128, 128)
        x = self.conv_1(x)  # Shape: (batch_size, 64, 61, 61)
        x = self.activation(x)
        x = self.max_pool(x)  # Shape: (batch_size, 64, 30, 30)

        x = self.block_1(x)  # Shape: (batch_size, 64, 15, 15)
        x = self.block_2(x)  # Shape: (batch_size, 128, 8, 8)
        x = self.block_3(x)  # Shape: (batch_size, 64, 4, 4)

        x = self.adaptive_avg_pool(x)  # Shape: (batch_size, 256, 1, 1)
        x = torch.squeeze(x)  # Shape: (batch_size, 256)
        x = self.fully_connected(x)  # Shape: (batch_size, 84)
        return x
