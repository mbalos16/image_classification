import torch
import torchvision.transforms as transforms


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
        # Col 1
        self.ins_conv_1x1_64 = torch.nn.Conv2d(
            in_channels=192, out_channels=64, kernel_size=1
        )
        # Col 2
        self.ins_conv_1x1_96 = torch.nn.Conv2d(
            in_channels=192, out_channels=96, kernel_size=1
        )
        self.ins_conv_3x3_128 = torch.nn.Conv2d(
            in_channels=96, out_channels=128, kernel_size=3, padding=1
        )
        # Col 3
        self.ins_conv_1x1_16 = torch.nn.Conv2d(
            in_channels=192, out_channels=16, kernel_size=1
        )
        self.ins_conv_5x5_32 = torch.nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, padding=2
        )
        # Col 4
        self.ins_max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.ins_conv_1x1_32 = torch.nn.Conv2d(
            in_channels=192, out_channels=32, kernel_size=1
        )

        # Stage Three | Inception BlocK Two
        # Col 1
        self.ins2_conv_1x1_128 = torch.nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=1
        )
        # Col 2
        # same: ins2_conv_1x1_128
        self.ins2_conv_3x3_192 = torch.nn.Conv2d(
            in_channels=128, out_channels=192, kernel_size=3, padding=1
        )
        # Col 3
        self.ins2_conv_1x1_32 = torch.nn.Conv2d(
            in_channels=256, out_channels=32, kernel_size=1
        )
        self.ins2_conv_5x5_96 = torch.nn.Conv2d(
            in_channels=32, out_channels=96, kernel_size=5, padding=2
        )
        # Col 4
        self.ins2_max_pool = torch.nn.MaxPool2d(kernel_size=3)
        self.ins2_conv_1x1_64 = torch.nn.Conv2d(
            in_channels=256, out_channels=64, kernel_size=1
        )
        # # TODO CONCATENATION

        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Out
        # Global AVg Pool
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.fully_connected = torch.nn.Linear(in_features=608, out_features=84)
        self.output = torch.nn.Linear(in_features=84, out_features=6)

        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.softmax = torch.nn.Softmax()

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
        #  Conv 1x1, (64)
        x_i_1 = self.ins_conv_1x1_64(x)  # Shape: (batch_size, 64, 32, 32)
        x_i_1 = self.activation(x_i_1)

        # Conv 1x1, (96) and Conv 3x3, (128)
        x_i_2 = self.ins_conv_1x1_96(x)  # Shape: (batch_size, 96, 32, 32)
        x_i_2 = self.activation(x_i_2)
        x_i_2 = self.ins_conv_3x3_128(x_i_2)  # Shape: (batch_size, 128, 32, 32)
        x_i_2 = self.activation(x_i_2)

        # Conv 1x1, (16) and Conv 5x5, (32)
        x_i_3 = self.ins_conv_1x1_16(x)  # Shape: (batch_size, 16, 32, 32)
        x_i_3 = self.activation(x_i_3)
        x_i_3 = self.ins_conv_5x5_32(x_i_3)  # Shape: (batch_size, 32, 32, 32)
        x_i_3 = self.activation(x_i_3)

        # MaxPool 3x3 and Conv 1x1 (32)
        x_i_4 = self.ins_max_pool(
            x
        )  # Shape: (batch_size, 192, 30, 30) # Why in debbug its (8, 192, 10, 10)
        x_i_4 = self.ins_conv_1x1_32(x_i_4)  # Shape: (batch_size, 32, 30, 30)
        x_i_4 = self.activation(x_i_4)

        # Concatenation
        x = torch.cat(
            (x_i_1, x_i_2, x_i_3, x_i_4), dim=1
        )  # Shape: (batch_size, 256, 32, 32)

        # Stage Three | Inception Block Two
        # Conv 1x1, (128)
        x_i2_1 = self.ins2_conv_1x1_128(x)  # Shape: (batch_size, 128, 32, 32) # WRONG
        x_i2_1 = self.activation(x)

        # Conv 1x1, (128) and Conv 3x3, (192)
        x_i2_2 = self.ins2_conv_1x1_128(x)  # Shape: (batch_size, 128, 32, 32)
        x_i2_2 = self.activation(x_i2_2)
        x_i2_2 = self.ins2_conv_3x3_192(x_i2_2)  # Shape: (batch_size, 192, 32, 32)
        x_i2_2 = self.activation(x_i2_2)

        # Conv 1x1, (32) and Conv 5x5, (96)
        x_i2_3 = self.ins2_conv_1x1_32(x)  # Shape: (batch_size, 32, 32, 32)
        x_i2_3 = self.activation(x_i2_3)
        x_i2_3 = self.ins2_conv_5x5_96(x_i2_3)  # Shape: (batch_size, 96, 32, 32)
        x_i2_3 = self.activation(x_i2_3)

        # MaxPool 3x3 and Conv 1x1 (64)
        x_i2_4 = self.ins_max_pool(x)  # Shape: (batch_size, 256, 32, 32)
        x_i2_4 = self.ins2_conv_1x1_64(
            x_i2_4
        )  # Shape: (batch_size, 64, 32, 32) # Falla sin razon.
        x_i2_4 = self.activation(x_i2_4)

        # Concatenation
        x = torch.cat(
            (x_i2_1, x_i2_2, x_i2_3, x_i2_4), dim=1
        )  # Shape: (batch_size, 608, 32, 32)

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

        # Softmax
        x = self.softmax(x)
        return x
