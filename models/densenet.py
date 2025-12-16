import torch
import torchvision.transforms as transforms


class DenseNetLayer(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm2d(in_channels)
        self.activation = torch.nn.ReLU()
        self.conv_3x3 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding="same",
            stride=1,
        )
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.conv_3x3(x)
        x = self.dropout(x)
        return x


class DenseBlock(torch.nn.Module):
    def __init__(
        self, num_layers: int, in_channels: int, growth_size: int, dropout: float = 0.2
    ):
        """The class used to define all the Dense Blocks

        Args:
            num_layers (int): Number of convolutional layers in the dense block.
            in_channels (int): Initial in_channels. These will be updated for each iteration.
            growth_size (int): The number of output channels is calculated as (in_channels + num_layers * growth_size)
            dropout (float, optional): The percentage of drop out. Defaults to 0.2.
        """
        super().__init__()
        self.dense_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            dense_layer = DenseNetLayer(
                in_channels=in_channels, out_channels=growth_size, dropout=dropout
            )
            self.dense_layers.append(dense_layer)
            in_channels += growth_size

    def forward(self, x: torch.Tensor):
        hiddens = [x]
        for layer in self.dense_layers:
            x = torch.concat(hiddens, dim=1)
            hidden = layer(x)
            hiddens.append(hidden)
        return x


class TransitionBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.2,
        max_pool_stride: float = 2,
    ):
        super().__init__()
        self.batch_norm = torch.nn.BatchNorm2d(in_channels)
        self.activation = torch.nn.ReLU()
        self.conv_1x1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding="same",
            stride=1,
        )
        self.drop_out = torch.nn.Dropout(p=dropout)
        self.max_pool = torch.nn.MaxPool2d(
            kernel_size=2, stride=max_pool_stride, padding=1
        )

    def forward(self, x: torch.Tensor):
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.conv_1x1(x)
        x = self.drop_out(x)
        x = self.max_pool(x)
        return x


class DenseNet(torch.nn.Module):
    def __init__(self, in_channels=3, growth_size=12):
        super().__init__()
        # Input size: (batch_size, 3, 128, 128)
        self.conv_7x7 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=7, stride=2
        )  # (batch_size, 64, 61, 61)
        self.max_pool = torch.nn.MaxPool2d(
            kernel_size=3, stride=2
        )  # (batch_size, 64, 30, 30)

        self.dense_block = DenseBlock(
            num_layers=6, in_channels=64, growth_size=growth_size
        )  # (batch_size, 124, 30, 30)
        self.transition = TransitionBlock(
            in_channels=124, out_channels=124
        )  # (batch_size, 124, 16, 16)

        self.dense_block_2 = DenseBlock(
            num_layers=12, in_channels=124, growth_size=growth_size
        )  # (batch_size, 256, 16, 16)
        self.transition_2 = TransitionBlock(
            in_channels=256, out_channels=256
        )  # (batch_size, 256, 9, 9)

        self.dense_block_3 = DenseBlock(
            num_layers=24, in_channels=256, growth_size=growth_size
        )  # (batch_size, 532, 9, 9)

        self.max_pool_7x7 = torch.nn.MaxPool2d(
            kernel_size=7, stride=1
        )  # batch_size, 532, 3,3)
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(
            output_size=1
        )  # batch_size, 532, 1, 1)
        self.output = torch.nn.Linear(
            in_features=532, out_features=6
        )  # (batch_size, 6)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.max_pool(x)
        x = self.dense_block(x)
        x = self.transition(x)
        x = self.dense_block_2(x)
        x = self.transition_2(x)
        x = self.dense_block_3(x)
        x = self.max_pool_7x7(x)
        x = self.adaptive_avg_pool(x)
        x = torch.squeeze(x)
        x = self.output(x)
        return x
