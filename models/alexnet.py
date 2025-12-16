import torch
import torchvision.transforms as transforms


class AlexNet(torch.nn.Module):
    def __init__(self, dropout=0.5):
        super(AlexNet, self).__init__()
        self.conv_1 = torch.nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=5
        )

        self.conv_2 = torch.nn.Conv2d(
            in_channels=96, out_channels=120, kernel_size=5, padding="same"
        )  # out original = 256 pad = 2

        self.conv_3 = torch.nn.Conv2d(
            in_channels=120, out_channels=120, kernel_size=3, padding="same"
        )  # out original 384

        self.conv_4 = torch.nn.Conv2d(
            in_channels=120, out_channels=256, kernel_size=3, padding="same"
        )  # out original 384

        # self.conv_5 = torch.nn.Conv2d(
        #     in_channels= 120, out_channels = 256, kernel_size = 3, padding = 1) # out original  = 256

        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.activation = torch.nn.ReLU()

        self.fully_connect_1 = torch.nn.Linear(
            in_features=4 * 4 * 256, out_features=4096
        )
        self.fully_connect_2 = torch.nn.Linear(in_features=4096, out_features=4096)
        self.fully_connect_3 = torch.nn.Linear(in_features=4096, out_features=6)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        # Input size: (batch_size, 3, 128, 128)
        x = self.conv_1(x)  # Shape (batch_size, 96, 32, 32)
        x = self.activation(x)  # Shape (batch_size, 96, 32, 32)
        x = self.max_pool(x)  # Shape (batch_size, 96, 16, 16)

        x = self.conv_2(x)  # Shape (batch_size, 120, 16, 16)
        x = self.activation(x)  # Shape (batch_size, 120, 16, 16)
        x = self.max_pool(x)  # Shape (batch_size, 120, 8, 8)

        x = self.conv_3(x)  # Shape(batch_size, 144, 8, 8)
        x = self.activation(x)  # Shape (batch_size, 144, 8, 8)

        x = self.conv_4(x)  # Shape(batch_size, 168, 8, 8)
        x = self.activation(x)  # Shape (batch_size, 168, 8, 8)
        x = self.max_pool(x)  # Shape (batch_size, 256, 4, 4)

        # Flatten
        x = torch.reshape(x, shape=(-1, 4 * 4 * 256))

        # Fully_connected
        x = self.fully_connect_1(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fully_connect_2(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.fully_connect_3(x)
        return x
