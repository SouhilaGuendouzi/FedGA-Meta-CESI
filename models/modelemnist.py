from torch import  nn
from .ImageClassificationBase import ImageClassificationBase

class Model_emnist(ImageClassificationBase):

    
     def __init__(self):
        super(Model_emnist, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5,padding=2),  #1
            #nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #2
            nn.Conv2d(10, 24, kernel_size=5,padding=1),  #3
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #4
            nn.Conv2d(24, 12, kernel_size=5,padding=1),  #5
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2), #6
            nn.Dropout2d()

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
