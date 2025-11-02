from torch import  nn
from .ImageClassificationBase import ImageClassificationBase



class Model_cloud(ImageClassificationBase):

    
     def __init__(self):
        super(Model_cloud, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 12, kernel_size=5, padding=1),
            #nn.BatchNorm2d(12),
            nn.Dropout2d(),
            nn.ReLU()

        )
        
        self.classification=nn.Sequential(

            nn.Flatten(),  #500*192
            nn.Linear(12*4*4, 10),
            nn.ReLU(),
             #nn.Linear(50, 100),
            #nn.ReLU(),
            #nn.Dropout(),
            #nn.Linear(50, 10)
  )
      
     def forward(self, x):
         x=self.features(x)
         x=self.classification(x)
         return x

