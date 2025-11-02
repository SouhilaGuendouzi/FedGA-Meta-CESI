import torch
import copy
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
class Edge:
    """
    Classe pour représenter un edge dans le cadre de fedprox.
    Chaque edge exécute localement des mises à jour.
    """
    def __init__(self, edge_id, model, data, optimizer, loss_fn, args):
        self.id = edge_id
        self.model = copy.deepcopy(model)
        self.train_data = data[0]
        self.test_data = data[1]
        #print(f"Edge {self.id}: Train {len(self.train_data)} Test {len(self.test_data)}")
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        try:
           dataset=self.test_data.dataset
           split_size=int(0.8 * len(dataset))
           indices = list(range(len(dataset)))
           support_indices = indices[:split_size]
           query_indices = indices[split_size:]
           # Créer des sous-datasets
           support_set = Subset(dataset, support_indices)
           query_set = Subset(dataset, query_indices)
           # Créer les DataLoaders correspondants
           self.support_set = DataLoader(support_set, batch_size=self.fedga_dataset.batch_size, shuffle=True)
           self.query_set= DataLoader(query_set, batch_size=self.fedga_dataset.batch_size, shuffle=True)

        except:
            pass

    def local_update_fedprox(self, global_model, mu=0.01):
          self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
          """Effectue une mise à jour locale sur l'edge avec FedProx."""
          self.model.train()
          
          for epoch in range(self.args.local_epochs):
              for batch_idx, (x, y) in enumerate(self.train_data):
                    x, y = x.to(self.device), y.to(self.device)
                    x, y = Variable(x), Variable(y)
                    if (self.args.dataset=='usps'):
                        vector=[]
                        for i in range(len(y)):
                          vector.append(y[i])
                        vector=torch.tensor(vector).cuda()
                        y=vector
                        x = torch.reshape(x, (len(x), 1, 28, 28))
                    self.optimizer.zero_grad()
                    output = self.model(x)
                    try: 
                      loss = self.loss_fn(output, y)
                    except:
                       y=y.squeeze(1)
                       loss = self.loss_fn(output, y)
                    
                    prox_penalty = 0.0
                    global_weights = global_model.state_dict()
                    for name, param in self.model.named_parameters():
                       if param.requires_grad:
                         prox_penalty += torch.sum((param -global_weights[name].to(self.device)) ** 2)
                         prox_penalty = (mu / 2) * prox_penalty
                    loss_total = loss + prox_penalty
                    loss_total.backward()
                    self.optimizer.step()

          return self.model.state_dict(), loss_total.item()
    
    def expected_calibration_error(self,probs, labels, num_bins=15):
       """Calcule l'Expected Calibration Error (ECE)"""
       bin_boundaries = torch.linspace(0, 1, num_bins + 1)
       bin_lowers = bin_boundaries[:-1]
       bin_uppers = bin_boundaries[1:]

       confidences, predictions = torch.max(probs, 1)
       accuracies = predictions.eq(labels)

       ece = torch.tensor(0.0, device=probs.device)
       for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
          in_bin = (confidences >= bin_lower) & (confidences < bin_upper)
          prop_in_bin = in_bin.float().mean()

          if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

       return ece.item()
    
    def local_update(self):
        """Effectue une mise à jour locale sur l'edge."""
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.model.train()
        for epoch in range(self.args.local_epochs):
            for batch_idx, (x, y) in enumerate(self.train_data):

                x, y = x.to(self.device), y.to(self.device)
                x, y = Variable(x), Variable(y)
                if (self.args.dataset=='usps'):
                   vector=[]
                   for i in range(len(y)):
                      vector.append(y[i])
                   vector=torch.tensor(vector).cuda()
                   y=vector
                   x = torch.reshape(x, (len(x), 1, 28, 28))

                self.optimizer.zero_grad()

               
                output = self.model(x)
                try: 
                  loss = self.loss_fn(output, y)
                except:
                    y=y.squeeze(1)
                    loss = self.loss_fn(output, y)
                loss.backward()
                self.optimizer.step()
                
        return self.model.state_dict(), loss.item()
    
    """
    def evaluate(self, train=True):
      
       
        if train:
            data = self.train_data
        else :
            data = self.test_data
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in data:
                x, y = x.to(self.device), y.to(self.device)     
                x, y = Variable(x), Variable(y)      
                if (self.args.dataset=='usps'):
                   vector=[]
                   for i in range(len(y)):
                      vector.append(y[i])
                   vector=torch.tensor(vector).cuda()
                   y=vector
                   x = torch.reshape(x, (len(x), 1, 28, 28))
                output = self.model(x)
                try: 
                  loss = self.loss_fn(output, y)
                except:
                    y=y.squeeze(1)
                    loss = self.loss_fn(output, y)
                total_loss += loss.item()
                _, predicted = torch.max(output, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        accuracy = 100 * correct / total
        return total_loss / len(self.test_data), accuracy, all_preds, all_labels
"""

    def evaluate(self, train=True):
       """Évalue le modèle local sur les données de test."""
       data = self.train_data if train else self.test_data
       self.model.eval()
    
       correct = 0
       total = 0
       total_loss = 0.0
       all_preds = []
       all_labels = []
       all_probs = []

       #loss_fn = torch.nn.CrossEntropyLoss()  #reduction='mean', Utilisé pour la NLL

       with torch.no_grad():
        for x, y in data:
            x, y = x.to(self.device), y.to(self.device)
            x, y = torch.autograd.Variable(x), torch.autograd.Variable(y)

            if self.args.dataset == 'usps':
                y = torch.tensor([yi.item() for yi in y], dtype=torch.long).to(self.device)
                x = torch.reshape(x, (len(x), 1, 28, 28))

            output = self.model(x)
            probs=F.softmax(output, dim=1) #Calcul des probabilités pour le calcul de l'ECE
            
            try: 
                  loss = self.loss_fn(output, y)
            except:
                    y=y.squeeze(1)
                    loss = self.loss_fn(output, y)

            total_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

       accuracy = 100 * correct / total
       ece = self.expected_calibration_error(torch.tensor(all_probs), torch.tensor(all_labels))
       avg_loss = total_loss / len(data)

       return avg_loss, accuracy, ece, all_preds, all_labels


    def inner_update(self,global_classification_weights=None, round_num=None):
        self.model.classification.load_state_dict(global_classification_weights)
        _, loss_update= self.local_update()
        train_loss, train_accuracy, train_ece, train_preds, train_labels = self.evaluate(train=True)
        test_loss, test_accuracy, test_ece, test_preds, test_labels = self.evaluate(train=False)
        if (round_num!=self.args.epochs):
            print(f"Edge {self.id}: Starting inner update")
            classification_otpimizer = torch.optim.SGD(self.model.classification.parameters(), lr=self.args.lr)
            for batch__idx, (data, target) in enumerate(self.train_data):
                data, target = data.to(self.device), target.to(self.device)
                if (self.args.dataset=='usps'):
                        vector=[]
                        for i in range(len(target)):
                          vector.append(target[i])
                        vector=torch.tensor(vector).cuda()
                        target=vector
                        data = torch.reshape(data, (len(data), 1, 28, 28))
                classification_otpimizer.zero_grad()
                output = self.model(data)
                try:  
                  loss = self.loss_fn(output, target)
                except:
                  target = target.squeeze(1)
                  loss = self.loss_fn(output, target)

                loss.backward()
                classification_otpimizer.step()
        
                self.model.eval()

                classification_grads = {}
                 # Désactiver la mise à jour des gradients des couches convolutionnelles
                for param in self.model.parameters():
                   param.requires_grad = False
                # Réactiver les gradients uniquement pour la classification layer
                for param in self.model.classification.parameters():
                    param.requires_grad = True

                with torch.enable_grad():  # Activer la computation des gradients pour la classification
                  for data, target in self.test_data:
                      data, target = data.to(self.device), target.to(self.device)
                      if (self.args.dataset=='usps'):
                        vector=[]
                        for i in range(len(target)):
                          vector.append(target[i])
                        vector=torch.tensor(vector).cuda()
                        target=vector
                        data = torch.reshape(data, (len(data), 1, 28, 28))

                      output = self.model(data)
                      try:
                         loss = self.loss_fn(output, target)
                      except:
                         target = target.squeeze(1)
                         loss = self.loss_fn(output, target)

                      loss.backward()
                    # Stocker les gradients des couches fully connected (classification layer)
                      for name, param in self.model.classification.named_parameters():
                        if param.grad is not None:
                          classification_grads[name] = param.grad.clone().detach()

                      break  # Arrêter après le premier batch
                       # Réactiver tous les gradients pour l'entraînement normal après l'évaluation
                for param in self.model.parameters():
                   param.requires_grad = True




            return classification_grads, loss_update, train_accuracy, train_ece, train_preds, train_labels, test_loss, test_accuracy, test_ece, test_preds, test_labels 