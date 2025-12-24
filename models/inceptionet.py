import torch
import torchvision.transforms as transforms


class InceptionBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels=192,
        out_ch_h1=64,
        out_ch_h2=96,
        out_ch_h2_2=128,
        out_ch_h3=16,
        out_ch_h3_3=32,
        out_ch_h4=32,
        kernel_size=1,
    ):
        super(InceptionBlock, self).__init__()
        self.conv_1 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_ch_h1, kernel_size=kernel_size
        )  # Col 1

        self.conv_2 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_ch_h2, kernel_size=kernel_size
        )  # Col 2
        self.conv_3 = torch.nn.Conv2d(
            in_channels=out_ch_h2, out_channels=out_ch_h2_2, kernel_size=3, padding=1
        )

        self.conv_4 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_ch_h3, kernel_size=kernel_size
        )  # Col 3
        self.conv_5 = torch.nn.Conv2d(
            in_channels=out_ch_h3, out_channels=out_ch_h3_3, kernel_size=5, padding=2
        )

        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # Col 4
        self.conv_6 = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=out_ch_h4, kernel_size=kernel_size
        )

        self.activation = torch.nn.ReLU()

    def forward(self, x):
        # Initial shape: (8, 3, 128, 128)
        h1 = self.conv_1(x)  # COL 1
        h1 = self.activation(h1)

        h2 = self.conv_2(x)  # COL 2
        h2 = self.activation(h2)
        h2 = self.conv_3(h2)
        h2 = self.activation(h2)  # Shape: 8, 128, 128, 128

        h3 = self.conv_4(x)  # COL 3
        h3 = self.activation(h3)
        h3 = self.conv_5(h3)
        h3 = self.activation(h3)  # Shape: 8, 32, 128, 128

        h4 = self.max_pool(x)  # COL 4
        h4 = self.conv_6(h4)
        h4 = self.activation(h4)  # Shape: 8, 32, 128, 128

        return torch.cat((h1, h2, h3, h4), dim=1)


class GoogLeNet(torch.nn.Module):
    def __init__(self, dropout=0.4):
        super(GoogLeNet, self).__init__()
        # Stage One
        # 7X7 Conv
        self.conv_7x7 = torch.nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3
        )
        # 3X3MaxPool
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage Two
        # 1x1 Conv
        self.conv_1x1 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        # 3x3 Conv
        self.conv_3x3 = torch.nn.Conv2d(
            in_channels=64, out_channels=192, kernel_size=3, padding=1
        )
        # 3x3 MaxPool same as stage 1

        # Stage Three | Inception BlocK One
        self.block_one = InceptionBlock(
            in_channels=192,
            out_ch_h1=64,
            out_ch_h2=96,
            out_ch_h2_2=128,
            out_ch_h3=16,
            out_ch_h3_3=32,
            out_ch_h4=32,
            kernel_size=1,
        )

        # Stage Three | Inception BlocK One
        self.block_two = InceptionBlock(
            in_channels=256,
            out_ch_h1=128,
            out_ch_h2=128,
            out_ch_h2_2=192,
            out_ch_h3=32,
            out_ch_h3_3=96,
            out_ch_h4=64,
            kernel_size=1,
        )

        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Out
        # Global AVg Pool
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.fully_connected = torch.nn.Linear(in_features=480, out_features=84)
        self.output = torch.nn.Linear(in_features=84, out_features=6)

        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        # Input size: (batch_size, 3, 128, 128)
        # Stage One
        # 7X7 Conv:
        x = self.conv_7x7(x)  # Shape: (batch_size, 64, 128, 128)
        x = self.activation(x)
        # 3X3MaxPool:
        x = self.max_pool(x)  # Shape: (batch_size, 64, 64, 64)

        # Stage Two
        # 1x1 Conv:
        x = self.conv_1x1(x)  # Shape: (batch_size, 64, 64, 64)
        x = self.activation(x)
        # 3x3 Conv:
        x = self.conv_3x3(x)  # Shape (batch_size, 192, 64, 64)
        x = self.activation(x)
        # 3x3 MaxPool:
        x = self.max_pool(x)  # Shape: (batch_size, 192, 32, 32)

        # Stage Three | Inception Block One
        x = self.block_one(x)

        # Stage Three | Inception Block Two
        x = self.block_two(x)

        # 3X3MaxPool
        x = self.max_pool2(x)  # Shape: (batch_size, 608, 32, 32)

        # Out
        # Flatten the red. Shape: (batch_size, 32*32*608)
        x = self.adaptive_avg_pool(x)
        x = torch.squeeze(x)  # (batch_size, 608)

        x = self.fully_connected(x)
        x = self.activation(x)

        x = self.dropout(x)
        x = self.output(x)

        return x
