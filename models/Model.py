from datetime import datetime
import torch
from torch import  nn
import random
import torch.nn.functional as F
from Enumeration import ModelScope, MachineLearning
import os



class Model(object):
    def __init__(self,
                  id: int, 
        scope: ModelScope, 
        type: MachineLearning,  
        model:nn.Module,
        
        device
        ):
        
        self.id = id
        self.scope = scope
        self.type = type
        
        self.cid=None
        self.model=model.to(device)  ## Non Initialized Model

        self.hyperparameters = {"lr":0.01, "batch_size": 32, "epochs": 10}
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.hyperparameters["lr"])
        self.test_metrics = {"accuracy": 0, "loss": 0,"F1-score":0}
  

        self.training_duration=None,  # TimeSpan can be represented by a float in seconds
        self.model_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.weights =self.get_weights_biases()[0]
        self.bias = self.get_weights_biases()[1]
        self.creation_date = datetime.now()
        self.training_history = None  ##JsonFile
        self.last_trained = None

    
    def get_weights_biases(self):
        weights = []
        biases = []
        for layer in self.model.modules(): 
            if hasattr(layer, 'weight') and isinstance(layer.weight, nn.Parameter):
                weights.append(layer.weight.data)
                biases.append(layer.bias.data if hasattr(layer, 'bias') else None)
        return weights, biases
       

    def model_size(self):
       self.model_size=sum(p.numel() for p in self.model.parameters() if p.requires_grad)
       return self.model_size
    

    def save_model(self, path):
        """
        print("Saving model to path: ", path, os.path.exists(path))
        print("Peut-on écrire dans Storage/EdgeModels/ ?", os.access("Storage/EdgeModels/", os.W_OK))
        print("Peut-on écrire dans le fichier ?", os.access(path, os.W_OK))
        print("Vérification du modèle :", isinstance(self.model, torch.nn.Module))
        print("Nombre de paramètres du modèle :", sum(p.numel() for p in self.model.parameters()))
        print(path)"""
        torch.save(self.model.state_dict(), path)
        return path
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        return self.model
    

    def print_last_layer(self):

        last_layer = list(self.model.children())[-1]

        # Vérifiez si c'est une couche linéaire ou convolutionnelle
        if isinstance(last_layer, torch.nn.Linear) or isinstance(last_layer, torch.nn.Conv2d):
           # Affichez les poids de la dernière couche
           print(last_layer.weight)
        else:
           # Si la dernière couche n'est pas directement nn.Linear ou nn.Conv2d, essayez de trouver la dernière couche linéaire/convolutionnelle
           for module in reversed(list(self.model.modules())):
              if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
               print(module.weight)
               break
    
    def simulate_attack_on_model(self,percentage=70):
       # Convert the percentage to a fraction
       fraction_to_modify = percentage / 100.0

       # Get the model's state_dict (a dictionary of all the model's parameters)
       state_dict = self.model.state_dict()

       # Iterate over all the parameters in the model
       for name, param in state_dict.items():
           if param.requires_grad:
               # Flatten the parameter tensor to a 1D array for easier manipulation
               param_flat = param.view(-1)
               num_params = param_flat.size(0)

               # Determine the number of parameters to modify
               num_to_modify = int(num_params * fraction_to_modify)

               # Randomly select the indices of the parameters to modify
               indices_to_modify = random.sample(range(num_params), num_to_modify)

               # Apply the modifications (here we add random noise)
               for idx in indices_to_modify:
                   param_flat[idx] += torch.randn(1) * 0.1  # Adding small random noise

            # Reshape the parameter tensor back to its original shape
               state_dict[name].copy_(param_flat.view(param.size()))

    # Load the modified state_dict back into the model
       self.model.load_state_dict(state_dict)

       return self.model
    
    
    

