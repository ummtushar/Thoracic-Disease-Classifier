import torch
import torch.nn as nn
import torchvision.models as models


# ORIGINAL ORGINAL NET (from template)
class Net_BINARY(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(Net_BINARY, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, 32, kernel_size=4, stride=1),
            nn.PReLU(),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            nn.AvgPool2d(kernel_size=3),
            torch.nn.Dropout(p=0.5, inplace=True),
            # Defining another 2D convolution layer
            nn.Conv2d(32, 64, kernel_size=4, stride=1),
            nn.PReLU(),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
            nn.AvgPool2d(kernel_size=3),
            torch.nn.Dropout(p=0.25, inplace=True),
            # Defining another 2D convolution layer
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=3),
            torch.nn.Dropout(p=0.125, inplace=True),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(1152, 312),
            nn.Linear(312, n_classes)
        )

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        # After our convolutional layers which are 2D, we need to flatten our
        # input to be 1 dimensional, as the linear layers require this.
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
    
