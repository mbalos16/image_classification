import torch
import torchvision.transforms as transforms


class BaselineCNN(torch.nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        # Input shape: (batch_size, 3, 128, 128)
        self.conv_1 = torch.nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3), padding="same"
        )  # Shape: (batch_size, 32, 128, 128)
        self.activation = torch.nn.ReLU()
        self.conv_2 = torch.nn.Conv2d(
            in_channels=32, out_channels=6, kernel_size=(3, 3), padding="same"
        )  # Shape: (batch_size, 6, 128, 128)
        self.av_pool = torch.nn.AvgPool2d(kernel_size=(128, 128))

    def forward(self, x):
        x = self.conv_1(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = self.av_pool(x)
        x = torch.squeeze(input=x, dim=[2, 3])  # Remove the last two dimensions
        return x
