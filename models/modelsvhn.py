from torch import  nn
from .ImageClassificationBase import ImageClassificationBase

class Model_svhn(ImageClassificationBase):

    
     def __init__(self):
        super(Model_svhn, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=5,padding=2),  #1
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #2
            nn.Conv2d(100, 240, kernel_size=5,padding=1),  #3
            nn.BatchNorm2d(240),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #4
            nn.Conv2d(240, 256, kernel_size=5,padding=1),  #5
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #6
            nn.Dropout2d(),
            nn.Conv2d(256, 12, kernel_size=2,padding=4),  #7
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), #8
        #nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
        #nn.ReLU(),
        #nn.MaxPool2d(kernel_size=2, stride=2),
        #nn.Conv2d(in_channels=6, out_channels=24, kernel_size=5),
        #nn.ReLU(),
        #nn.MaxPool2d(kernel_size=2, stride=2),
        #nn.Conv2d(in_channels=24, out_channels=12, kernel_size=1, padding=2),
        #nn.ReLU(),
        #nn.MaxPool2d(kernel_size=3, stride=2),

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


'''

class Model_svhn_new(nn.Module):

    
     def __init__(self):
        super(Model_svhn_new, self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5,padding=2),  #1
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2), #2
            nn.Conv2d(256, 240, kernel_size=5,padding=1),  #3
            nn.BatchNorm2d(240),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), #4
            nn.Dropout(0.3),
            nn.Conv2d(240, 256, kernel_size=5,padding=1),  #5
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(kernel_size=2, stride=2), #6
            nn.Dropout2d(),
            nn.Conv2d(256, 12, kernel_size=2,padding=4),  #7
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), #8
        )
        
        self.classification=nn.Sequential(
            nn.Flatten(),  #500*192
            nn.Linear(12*4*4, 50),
            nn.ReLU(),
            nn.Linear(50, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 10),
            nn.Softmax(dim=1)
  )
      
     def forward(self, x): 
         
         x=self.features(x)
        
         x=self.classification(x)
         return x




class Model_svhn_newN(nn.Module):

    
     def __init__(self):
        super(Model_svhn_newN, self).__init__()
        self.features=nn.Sequential(
             nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4
        )
        
        self.classification=nn.Sequential(
            nn.Flatten(),  #500*192
            nn.Linear(12*4*4, 50),
            nn.ReLU(),
            nn.Linear(50, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 10),
            nn.Softmax(dim=1)
  )
      
     def forward(self, x): 
         
         x=self.features(x)
        
         x=self.classification(x)
         return x



class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    
    def validation_step(self, batch):
        images, labels = batch 
        images, labels = images.cuda(), labels.cuda()
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))




class SVHNCnnModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
        
    def forward(self, xb):
        return self.network(xb)



class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return 
'''