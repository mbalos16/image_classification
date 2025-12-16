import torch
import torchvision.transforms as transforms


class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_1 = torch.nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=(5, 5)
        )  # C1
        self.conv_2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=(5, 5)
        )  # C3
        self.conv_3 = torch.nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=(5, 5)
        )  # C5
        self.activation = torch.nn.Tanh()
        self.av_pool = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.fully_connect = torch.nn.Linear(
            in_features=25 * 25 * 120, out_features=84
        )  # F6
        self.output = torch.nn.Linear(in_features=84, out_features=6)

    def forward(self, x):
        # Input shape: (batch_size, 3, 128, 128)
        x = self.conv_1(x)  # Shape: (batch_size, 14, 124, 124) C1
        x = self.activation(x)
        x = self.av_pool(x)  # Shape: (batch_size, 14, 62, 62) S2

        x = self.conv_2(x)  # Shape: (bach_size, 16, 58, 58) C3
        x = self.activation(x)
        x = self.av_pool(x)  # Shape: (batch_size, 16, 29, 29) S4

        x = self.conv_3(x)  # Shape: (batch_size, 120, 25, 25) C5
        x = self.activation(x)

        # Flatten the red. Shape: (batch_size, 25*25*120)
        x = torch.reshape(
            x, shape=(-1, 25 * 25 * 120)
        )  # Use -1 for the first dim when you want the batch size to be calculatted automatically.

        x = self.fully_connect(x)  # Shape: (batch_size, 84)
        x = self.activation(x)

        x = self.output(x)

        return x


class LeNetMod(torch.nn.Module):
    """
    Sustitute the reshape layer in the model definition with a globalAveragePooling (AdaptativeAvgPool2d with output_size of 1) layer.
    After this layer a squeeze is needed so the dimensions with a value of 1 are removed. This is a smaller model, with less paramethers
    so it is bettter for the small dataset we have.
    """

    def __init__(self):
        super(LeNetMod, self).__init__()
        self.conv_1 = torch.nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=(5, 5)
        )  # C1
        self.conv_2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=(5, 5)
        )  # C3
        self.conv_3 = torch.nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=(5, 5)
        )  # C5
        self.activation = torch.nn.Tanh()
        self.av_pool = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.fully_connect = torch.nn.Linear(in_features=120, out_features=84)  # F6
        self.output = torch.nn.Linear(in_features=84, out_features=6)

    def forward(self, x):
        # Input shape: (batch_size, 3, 128, 128)
        x = self.conv_1(x)  # Shape: (batch_size, 14, 124, 124) C1
        x = self.activation(x)
        x = self.av_pool(x)  # Shape: (batch_size, 14, 62, 62) S2

        x = self.conv_2(x)  # Shape: (bach_size, 16, 58, 58) C3
        x = self.activation(x)
        x = self.av_pool(x)  # Shape: (batch_size, 16, 29, 29) S4

        x = self.conv_3(x)  # Shape: (batch_size, 120, 25, 25) C5
        x = self.activation(x)

        # Flatten the red. Shape: (batch_size, 25*25*120)
        x = self.adaptive_avg_pool(x)
        x = torch.squeeze(x)  # (batch_size, 120)

        x = self.fully_connect(x)  # Shape: (batch_size, 84)
        x = self.activation(x)  # delete dimensions with 1s

        x = self.output(x)

        return x


class LeNetModNorm(torch.nn.Module):
    """
    # LeNet - Modified + Batch normalization of 1 conv
    """

    def __init__(self):
        super(LeNetModNorm, self).__init__()
        self.conv_1 = torch.nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=(5, 5)
        )  # C1
        self.conv_2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=(5, 5)
        )  # C3
        self.conv_3 = torch.nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=(5, 5)
        )  # C5
        self.activation = torch.nn.Tanh()
        self.av_pool = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.fully_connect = torch.nn.Linear(in_features=120, out_features=84)  # F6
        self.output = torch.nn.Linear(in_features=84, out_features=6)
        self.batch_norm = torch.nn.BatchNorm2d(num_features=6)

    def forward(self, x):
        # Input shape: (batch_size, 3, 128, 128)
        x = self.conv_1(x)  # Shape: (batch_size, 14, 124, 124) C1
        x = self.batch_norm(x)  # batch normalization on channels.

        x = self.activation(x)
        x = self.av_pool(x)  # Shape: (batch_size, 14, 62, 62) S2

        x = self.conv_2(x)  # Shape: (bach_size, 16, 58, 58) C3
        x = self.activation(x)
        x = self.av_pool(x)  # Shape: (batch_size, 16, 29, 29) S4

        x = self.conv_3(x)  # Shape: (batch_size, 120, 25, 25) C5
        x = self.activation(x)

        # Flatten the red. Shape: (batch_size, 25*25*120)
        x = self.adaptive_avg_pool(x)
        x = torch.squeeze(x)  # (batch_size, 120)

        x = self.fully_connect(x)  # Shape: (batch_size, 84)
        x = self.activation(x)  # delete dimensions with 1s

        x = self.output(x)

        return x


class LeNetModNorm2(torch.nn.Module):
    """
    LeNet - Modified + Batch normalization of all conv
    """

    def __init__(self):
        super(LeNetModNorm2, self).__init__()
        self.conv_1 = torch.nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=(5, 5)
        )  # C1
        self.conv_2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=(5, 5)
        )  # C3
        self.conv_3 = torch.nn.Conv2d(
            in_channels=16, out_channels=120, kernel_size=(5, 5)
        )  # C5
        self.activation = torch.nn.Tanh()
        self.av_pool = torch.nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.adaptive_avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.fully_connect = torch.nn.Linear(in_features=120, out_features=84)  # F6
        self.output = torch.nn.Linear(in_features=84, out_features=6)
        self.batch_norm1 = torch.nn.BatchNorm2d(6)
        self.batch_norm2 = torch.nn.BatchNorm2d(16)
        self.batch_norm3 = torch.nn.BatchNorm2d(120)

    def forward(self, x):
        # Input shape: (batch_size, 3, 128, 128)
        x = self.conv_1(x)  # Shape: (batch_size, 14, 124, 124) C1
        x = self.batch_norm1(x)  # batch normalization on channels.

        x = self.activation(x)
        x = self.av_pool(x)  # Shape: (batch_size, 14, 62, 62) S2

        x = self.conv_2(x)  # Shape: (bach_size, 16, 58, 58) C3
        x = self.batch_norm2(x)
        x = self.activation(x)
        x = self.av_pool(x)  # Shape: (batch_size, 16, 29, 29) S4

        x = self.conv_3(x)  # Shape: (batch_size, 120, 25, 25) C5
        x = self.batch_norm3(x)
        x = self.activation(x)

        # Flatten the red. Shape: (batch_size, 25*25*120)
        x = self.adaptive_avg_pool(x)
        x = torch.squeeze(x)  # (batch_size, 120)

        x = self.fully_connect(x)  # Shape: (batch_size, 84)
        x = self.activation(x)  # delete dimensions with 1s

        x = self.output(x)

        return x
