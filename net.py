import torch
import torch.nn as nn
import torchvision.models as models

# Improved Net
class Net(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2), 
            nn.Dropout(p=0.5),

            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),  
            nn.Dropout(p=0.25),
            
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(p=0.125),

            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(p=0.1),

            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.05), 

            # New layer
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.05), 
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(16, 256),
            nn.Linear(256, n_classes)
        )

    # Defining the forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_layers(x)
        # After our convolutional layers which are 2D, we need to flatten our
        # input to be 1 dimensional, as the linear layers require this.
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
        

# Implementing Pre-Trained ResNet as a class
class ResNetModel(nn.Module):
    def __init__(self, n_classes: int, pretrained: bool = True): 
        # Loading a pre-trained ResNet model 
        super(ResNetModel, self).__init__()
        self.resnet = models.resnet34(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, n_classes)


    def forward(self, x):
        # Forward pass through the ResNet model
        return self.resnet(x)
    
# Implementing the Efficient Net Model with efficientnet_b0
class EfficientNetModel(nn.Module):
    def __init__(self, n_classes: int, version: str = 'b0', pretrained: bool = True):
        super(EfficientNetModel, self).__init__()
        # Loading a pretrained EfficientNet model
        self.efficientnet = models.efficientnet_b0(pretrained=pretrained) if version == 'b0' else models.__dict__[f'efficientnet_{version}'](pretrained=pretrained)
        
        # Adjusting the classifier to match the number of classes
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_ftrs, n_classes)

    def forward(self, x):
        # Forward pass through the EfficientNet model
        # Replicating the grayscale channel to have 3 channels
        x = x.repeat(1, 3, 1, 1)
        return self.efficientnet(x)
    
# Implementing the Efficient Net Model with efficientnet_b7
# NOTE: This model takes a lot of time to run
class EfficientNetModel_b7(nn.Module):
    def __init__(self, n_classes: int, version: str = 'b0', pretrained: bool = True):
        super(EfficientNetModel_b7, self).__init__()
        # Loading a pretrained EfficientNet model
        self.efficientnet = models.efficientnet_b7(pretrained=pretrained) if version == 'b7' else models.__dict__[f'efficientnet_{version}'](pretrained=pretrained)
        
        # Adjusting the classifier to match the number of classes
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(num_ftrs, n_classes)

    def forward(self, x):
        # Forward pass through the EfficientNet model
        # Replicating the grayscale channel to have 3 channels
        x = x.repeat(1, 3, 1, 1)
        return self.efficientnet(x)
    