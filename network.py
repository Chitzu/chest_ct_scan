import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, n_channels=3):
        super(Network, self).__init__()

        self.n_features = 2 * n_channels

        self.conv1 = nn.Sequential(
            nn.Conv2d(n_channels, self.n_features, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(self.n_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1)
        )
    
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.n_features, 2 * self.n_features, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(2 * self.n_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * self.n_features, 4 * self.n_features, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(4 * self.n_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(4 * self.n_features, 8 * self.n_features, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(8 * self.n_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1),
            nn.Dropout()
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(8 * self.n_features, 8 * self.n_features, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(8 * self.n_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(8 * self.n_features, 8 * self.n_features, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(8 * self.n_features),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout()
        )
        
        self.classification_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3888, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = self.classification_layer(x)

        return x