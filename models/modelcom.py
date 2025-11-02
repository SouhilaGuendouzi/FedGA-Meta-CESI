import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, channels, dataset=None):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Max-pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        """
        if dataset == 'svhn':
            input_size = 128 * 8 * 8  # Adjusted input size for SVHN dataset
        else:
            input_size = 128 * 7 * 7  # Default input size
        """
        self.resize_layer = nn.AdaptiveAvgPool2d((7, 7))  # Réduction à 7x7 pour tous
        #input_size = 128 * 7 * 7
        input_size=64* 7 * 7
        self.classification = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 10), #128
            #nn.ReLU(),
            #nn.Linear(128, 10), 
            #nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Convolutional layers with ReLU and max-pooling
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.pool(F.relu(self.conv3(x)))
        # Redimensionner à une taille fixe (7x7) pour garder le même input_size
        x = self.resize_layer(x)
        # Flatten the tensor
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.classification(x)
        
        return x