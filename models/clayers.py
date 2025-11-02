import torch
from torch import nn

class Feature_extractor_Layers(nn.Module):  ## The same as global model 
  def __init__(self):
    super().__init__()

    # define layers
    self.features = nn.Sequential(

            nn.Conv2d(1, 10, kernel_size=5,padding=2),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(10, 24, kernel_size=5,padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(24, 12, kernel_size=5,padding=1),  
            nn.ReLU(),
            nn.Dropout2d()


    )
  def forward(self, t):

    t = self.features(t)
  
    return t