import torch
import copy
import json
import os
from Entities.edge import Edge
from Aggregation.FedGA import FedGA
from torch.utils.data import DataLoader, Subset
import math
#from FedGA_test import FedGA
class Fog:
    """
    Classe pour représenter un Fog dans le cadre de fedprox.
    Chaque Fog gère plusieurs edges et exécute fedprox.
    """
    def __init__(self, fog_id, model, data, optimizer, loss_fn, args,method, fedga_dataset=None):
        self.id = fog_id
        self.edges = []
        self.global_model = copy.deepcopy(model)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)
        self.data = data
        self.method = method
        try:
           self.fedga_dataset = fedga_dataset
           dataset=self.fedga_dataset.dataset
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
      


        # Créer les edges pour ce Fog
        for i in range(args.num_users):
            edge_data = data[i]  # Chaque edge reçoit une partie des données
            
            edge = Edge(i, model, edge_data, optimizer, loss_fn, args)
            self.edges.append(edge)

        # Créer le répertoire pour sauvegarder les résultats
        self.results_dir = f"results_algos/{self.method}_{self.args.num_users}_{self.args.local_epochs}/fog{self.id}/"
        os.makedirs(self.results_dir, exist_ok=True)

    def federated_training(self):
        """Exécute plusieurs tours de fedprox au niveau du Fog."""
        for round_num in range(self.args.epochs+1):
            print(f"Fog {self.id}: Starting round {round_num + 1}/{self.args.epochs}")

            # Collecter les poids de chaque edge
            local_weights = []
            round_train_loss = []
            round_train_accuracy = []
            round_train_f1 = []
            round_test_loss = []
            round_test_accuracy = []
            round_test_f1 = []
            round_train_ece = []
            round_test_ece = []
            client_accuracies = []  # Pour le calcul de la moyenne des précisions des clients
            round_edges = {}

            for edge in self.edges:
                global_weights = self.global_model.state_dict()
                edge.model.load_state_dict(copy.deepcopy(global_weights))
                if self.method == 'Fedprox' or self.method =='Fed_prox':
                    local_weight, loss = edge.local_update_fedprox(self.global_model)
                else: 
                    local_weight, loss = edge.local_update()
                round_train_loss.append(loss)
                local_weights.append(local_weight)

                # Évaluation locale sur les données d'entraînement et de test
                #train_loss, train_accuracy, train_preds, train_labels = edge.evaluate(train=True)
                #test_loss, test_accuracy, test_preds, test_labels = edge.evaluate(train=False)
                 # Évaluation locale
                train_loss, train_accuracy, train_ece, train_preds, train_labels = edge.evaluate(train=True)
                test_loss, test_accuracy, test_ece, test_preds, test_labels = edge.evaluate(train=False)
                client_accuracies.append(test_accuracy)


                #round_train_loss.append(train_loss)
                round_train_accuracy.append(train_accuracy)
                round_train_f1.append(self.compute_f1(train_preds, train_labels))
                round_test_loss.append(test_loss)
                round_test_accuracy.append(test_accuracy)
                round_test_f1.append(self.compute_f1(test_preds, test_labels))
                round_train_ece.append(train_ece)
                round_test_ece.append(test_ece)

                round_edges[edge.id] = {
                    "Train Loss": loss if not math.isnan(loss) else 0,
                    "Train Accuracy": train_accuracy,
                    "Train F1": self.compute_f1(train_preds, train_labels),
                    "Train ECE": train_ece if not math.isnan(train_ece) else 0,
                    "Test Loss": test_loss if not math.isnan(test_loss) else 0,
                    "Test Accuracy": test_accuracy,
                    "Test F1": self.compute_f1(test_preds, test_labels),
                    "Test ECE": test_ece if not math.isnan(test_ece) else 0,
                }

            # Agréger les poids avec fedprox
            if (round_num != self.args.epochs):
              if self.method != "FedGA" and self.method != "Fed_GA" and  self.method!= "Fed_GA2" :
                 global_weights = self.fedavg_aggregate(local_weights)
              elif self.method == "FedGA" or self.method == "Fed_GA" or self.method == "Fed_GA2":
                  global_weights=self.fedga_aggregate(local_weights)
            
              self.global_model.load_state_dict(global_weights)
     
            # Calcul des moyennes pour ce round
            def safe_mean(data):
               return sum(data) / len(data) if len(data) > 0 and not math.isnan(sum(data)) else 0
            

            if (len(self.edges)>0):
                num_worst_clients = max(1, len(self.edges) // 10)  # Prend au moins un client
                worst_10_percent_acc = safe_mean(sorted(client_accuracies)[:num_worst_clients])
            else :
                worst_10_percent_acc = 0

            results = {
                "Round": round_num,
                "Train Loss": sum(round_train_loss) / len(round_train_loss) if not math.isnan(sum(round_train_loss) / len(round_train_loss)) else 0,
                "Train Accuracy": sum(round_train_accuracy) / len(round_train_accuracy),
                "Train F1": sum(round_train_f1) / len(round_train_f1),
                "Train ECE": sum(round_train_ece) / len(round_train_ece) if not math.isnan(sum(round_train_ece) / len(round_train_ece)) else 0,
                "Test Loss": sum(round_test_loss) / len(round_test_loss) if not math.isnan(sum(round_test_loss) / len(round_test_loss)) else 0,
                "Test Accuracy": sum(round_test_accuracy) / len(round_test_accuracy),
                "Test F1": sum(round_test_f1) / len(round_test_f1) ,
                "Test ECE": sum(round_test_ece) / len(round_test_ece) if not math.isnan(sum(round_test_ece) / len(round_test_ece)) else 0,
                "Worst 10% Accuracy": worst_10_percent_acc,
                "Edges": round_edges,
                
            }

            # Sauvegarder les résultats dans un fichier JSON
            with open(os.path.join(self.results_dir, f"round_{round_num + 1}.json"), "w") as f:
                json.dump(results, f, indent=4)

            print(f"Fog {self.id}: Round {round_num} - Results saved.")
            
               

            
    def fedga_meta_inner(self,global_classification_weights=None, round_num=None):
            """Exécute plusieurs tours de fedprox au niveau du Fog."""
            print(f"Fog {self.id}: Starting round {round_num + 1}/{self.args.epochs}")
            # Collecter les poids de chaque edge
            local_weights = []
            round_train_loss = []
            round_train_accuracy = []
            round_train_f1 = []
            round_test_loss = []
            round_test_accuracy = []
            round_test_f1 = []
            round_train_ece = []
            round_test_ece = []
            client_accuracies=[]
            round_edges = {}
            if global_classification_weights is not None:
                self.global_model.classification.load_state_dict(global_classification_weights)
            for edge in self.edges:
                global_weights = self.global_model.classification.state_dict()
                edge.model.classification.load_state_dict(copy.deepcopy(global_weights))
                local_weight, loss = edge.local_update()
                #local_weight, loss = edge.local_update_fedprox(self.global_model)
                round_train_loss.append(loss)
                #local_weights.append(local_weight)
                local_weights.append(copy.deepcopy(edge.model.classification.state_dict()))

                # Évaluation locale sur les données d'entraînement et de test
                train_loss, train_accuracy, train_ece, train_preds, train_labels = edge.evaluate(train=True)
                test_loss, test_accuracy, test_ece, test_preds, test_labels = edge.evaluate(train=False)
                client_accuracies.append(test_accuracy)

                #round_train_loss.append(train_loss)
                round_train_accuracy.append(train_accuracy)
                round_train_f1.append(self.compute_f1(train_preds, train_labels))
                round_test_loss.append(test_loss)
                round_test_accuracy.append(test_accuracy)
                round_test_f1.append(self.compute_f1(test_preds, test_labels))
                round_train_ece.append(train_ece)
                round_test_ece.append(test_ece)

                round_edges[edge.id] = {
                    "Train Loss": loss if not math.isnan(loss) else 0,
                    "Train Accuracy": train_accuracy,
                    "Train F1": self.compute_f1(train_preds, train_labels),
                    "Train ECE": train_ece if not math.isnan(train_ece) else 0,
                    "Test Loss": test_loss if not math.isnan(test_loss) else 0,
                    "Test Accuracy": test_accuracy,
                    "Test F1": self.compute_f1(test_preds, test_labels),
                    "Test ECE": test_ece if not math.isnan(test_ece) else 0,
                }
            
                # Calcul des moyennes pour ce round
                def safe_mean(data):
                   return sum(data) / len(data) if len(data) > 0 and not math.isnan(sum(data)) else 0
            

            
                if (len(self.edges)>0):
                   num_worst_clients = max(1, len(self.edges) // 10)  # Prend au moins un client
                   worst_10_percent_acc = safe_mean(sorted(client_accuracies)[:num_worst_clients])
                else :
                   worst_10_percent_acc = 0

                results = {
                   "Round": round_num,
                   "Train Loss": sum(round_train_loss) / len(round_train_loss) if not math.isnan(sum(round_train_loss) / len(round_train_loss)) else 0,
                   "Train Accuracy": sum(round_train_accuracy) / len(round_train_accuracy),
                    "Train F1": sum(round_train_f1) / len(round_train_f1),
                   "Train ECE": sum(round_train_ece) / len(round_train_ece) if not math.isnan(sum(round_train_ece) / len(round_train_ece)) else 0,
                   "Test Loss": sum(round_test_loss) / len(round_test_loss) if not math.isnan(sum(round_test_loss) / len(round_test_loss)) else 0,
                   "Test Accuracy": sum(round_test_accuracy) / len(round_test_accuracy),
                   "Test F1": sum(round_test_f1) / len(round_test_f1),
                   "Test ECE": sum(round_test_ece) / len(round_test_ece) if not math.isnan(sum(round_test_ece) / len(round_test_ece)) else 0,
                   "Worst 10% Accuracy": worst_10_percent_acc,
                   "Edges": round_edges,
                
            }

            # Sauvegarder les résultats dans un fichier JSON
            with open(os.path.join(self.results_dir, f"round_{round_num + 1}.json"), "w") as f:
                json.dump(results, f, indent=4)

            #print(f"Fog {self.id}: Round {round_num + 1} - Results saved.")
            if (round_num!=self.args.epochs):
              global_weights = FedGA(local_weights, self.global_model, self.support_set, self.args.dataset, round_num)
              self.global_model.classification.load_state_dict(global_weights)
              self.inner_update_maml()
              classification_grads = self.evaluate_on_query_set() 
              return classification_grads
            else : return None


    
    def fedga_aggregate(self, local_weights):
      
        global_weights = FedGA(local_weights, self.global_model, self.fedga_dataset, self.args.dataset)
        
        return global_weights

    def fedavg_aggregate(self, local_weights):
        """Agréger les poids locaux avec FedAvg."""
        global_weights = copy.deepcopy(local_weights[0])
        for key in global_weights.keys():
            for i in range(1, len(local_weights)):
                global_weights[key] += local_weights[i][key]
            global_weights[key] = torch.div(global_weights[key], len(local_weights))
        return global_weights
    
    def get_classification_layers(self, global_weights=None, round_num=None):
        """
        Récupère les couches de classification de tous les modèles locaux des edges.
        :return: Liste des dictionnaires de poids des couches classification.
        """
        print(f"Fog {self.id}: Getting classification layers for round {round_num + 1}")
         # Collecter les poids de chaque edge
        local_weights = []
        round_train_loss = []
        round_train_accuracy = []
        round_train_f1 = []
        round_test_loss = []
        round_test_accuracy = []
        round_test_f1 = []
        round_train_ece = []
        round_test_ece = []
        client_accuracies = []  # Pour le calcul de la moyenne des précisions des clients
        round_edges = {}

        classification_layers = []
        for edge in self.edges:
            if global_weights is not None:
                edge.model.classification.load_state_dict(copy.deepcopy(global_weights)) #strict 
           
            if self.method == 'Fedprox' or self.method =='Fed_prox':
                    _, loss = edge.local_update_fedprox(self.global_model)
            else: 
                    _, loss = edge.local_update()
            classification_layers.append(edge.model.classification.state_dict())
            round_train_loss.append(loss)
            # Évaluation locale sur les données d'entraînement et de test
            train_loss, train_accuracy, train_ece, train_preds, train_labels = edge.evaluate(train=True)
            test_loss, test_accuracy, test_ece, test_preds, test_labels = edge.evaluate(train=False)
            client_accuracies.append(test_accuracy)

            #round_train_loss.append(train_loss)
            round_train_accuracy.append(train_accuracy)
            round_train_f1.append(self.compute_f1(train_preds, train_labels))
            round_test_loss.append(test_loss)
            round_test_accuracy.append(test_accuracy)
            round_test_f1.append(self.compute_f1(test_preds, test_labels))
            round_train_ece.append(train_ece)
            round_test_ece.append(test_ece)

            round_edges[edge.id] = {
                    "Train Loss": loss if not math.isnan(loss) else 0,
                    "Train Accuracy": train_accuracy,
                    "Train F1": self.compute_f1(train_preds, train_labels),
                    "Train ECE": train_ece if not math.isnan(train_ece) else 0,
                    "Test Loss": test_loss if not math.isnan(test_loss) else 0,
                    "Test Accuracy": test_accuracy,
                    "Test F1": self.compute_f1(test_preds, test_labels),
                    "Test ECE": test_ece if not math.isnan(test_ece) else 0,
                }
            
             # Calcul des moyennes pour ce round
            def safe_mean(data):
               return sum(data) / len(data) if len(data) > 0 and not math.isnan(sum(data)) else 0
            

            if (len(self.edges)>0):
                num_worst_clients = max(1, len(self.edges) // 10)  # Prend au moins un client
                worst_10_percent_acc = safe_mean(sorted(client_accuracies)[:num_worst_clients])
            else :
                worst_10_percent_acc = 0

            results = {
                "Round": round_num,
                "Train Loss": sum(round_train_loss) / len(round_train_loss) if not math.isnan(sum(round_train_loss) / len(round_train_loss)) else 0,
                "Train Accuracy": sum(round_train_accuracy) / len(round_train_accuracy),
                "Train F1": sum(round_train_f1) / len(round_train_f1),
                "Train ECE": sum(round_train_ece) / len(round_train_ece) if not math.isnan(sum(round_train_ece) / len(round_train_ece)) else 0,
                "Test Loss": sum(round_test_loss) / len(round_test_loss) if not math.isnan(sum(round_test_loss) / len(round_test_loss)) else 0,
                "Test Accuracy": sum(round_test_accuracy) / len(round_test_accuracy),
                "Test F1": sum(round_test_f1) / len(round_test_f1),
                "Test ECE": sum(round_test_ece) / len(round_test_ece) if not math.isnan(sum(round_test_ece) / len(round_test_ece)) else 0,
                "Worst 10% Accuracy": worst_10_percent_acc,
                "Edges": round_edges,
                
            }

        # Sauvegarder les résultats dans un fichier JSON
        with open(os.path.join(self.results_dir, f"round_{round_num + 1}.json"), "w") as f:
            json.dump(results, f, indent=4)
        
        return classification_layers

    def inner_update_maml(self):
        """Effectue une mise à jour MAML sur la classification layer."""
        print(f"Fog {self.id}: Starting inner update")
        classification_otpimizer = torch.optim.SGD(self.global_model.classification.parameters(), lr=self.args.lr)
        for batch__idx, (data, target) in enumerate(self.support_set):
            data, target = data.to(self.device), target.to(self.device)
            if (self.args.dataset=='usps'):
                        vector=[]
                        for i in range(len(target)):
                          vector.append(target[i])
                        vector=torch.tensor(vector).cuda()
                        target=vector
                        data = torch.reshape(data, (len(data), 1, 28, 28))
            classification_otpimizer.zero_grad()
            output = self.global_model(data)
            try:  
              loss = self.loss_fn(output, target)
            except:
              target = target.squeeze(1)
              loss = self.loss_fn(output, target)

            loss.backward()
            classification_otpimizer.step()
        
        return self.global_model.classification.state_dict()
    

    def evaluate_on_query_set(self):
      """
    Évalue le modèle sur le Query Set et retourne uniquement les gradients de la classification layer.

    Arguments :
    - client_model : Modèle après l'inner update.
    - query_loader : DataLoader contenant les données du query set du client.

    Retourne :
    - Un dictionnaire contenant les gradients de la classification layer.
      """
   
      self.global_model.eval()

      classification_grads = {}
      # Désactiver la mise à jour des gradients des couches convolutionnelles
      for param in self.global_model.parameters():
        param.requires_grad = False
      # Réactiver les gradients uniquement pour la classification layer
        for param in self.global_model.classification.parameters():
            param.requires_grad = True


      with torch.enable_grad():  # Activer la computation des gradients pour la classification
        for data, target in self.query_set:
            data, target = data.to(self.device), target.to(self.device)
            if (self.args.dataset=='usps'):
                        vector=[]
                        for i in range(len(target)):
                          vector.append(target[i])
                        vector=torch.tensor(vector).cuda()
                        target=vector
                        data = torch.reshape(data, (len(data), 1, 28, 28))

            output = self.global_model(data)
            try:
              loss = self.loss_fn(output, target)
            except:
                target = target.squeeze(1)
                loss = self.loss_fn(output, target)

            loss.backward()
            # Stocker les gradients des couches fully connected (classification layer)
            for name, param in self.global_model.classification.named_parameters():
                if param.grad is not None:
                    classification_grads[name] = param.grad.clone().detach()

            break  # Arrêter après le premier batch
            # Réactiver tous les gradients pour l'entraînement normal après l'évaluation
        for param in self.global_model.parameters():
                   param.requires_grad = True



      #print(classification_grads)
      #return classification_grads
      #
      return self.global_model.classification.state_dict()



    def compute_f1(self, preds, labels):
        """Calculer le F1-score à partir des prédictions et des étiquettes."""
        from sklearn.metrics import f1_score
        return f1_score(labels, preds, average="macro")
    

    def MAML_edges(self,global_classification_weights=None, round_num=None):
        list_classification_grads=[]
        round_train_loss = []
        round_train_accuracy = []
        round_train_f1 = []
        round_test_loss = []
        round_test_accuracy = []
        round_test_f1 = []
        round_train_ece = []
        round_test_ece = []
        client_accuracies=[]
        round_edges = {}
        for edge in self.edges:
           if global_classification_weights is None:
                classification_grads, loss_update, train_accuracy, train_ece, train_preds, train_labels, test_loss, test_accuracy, test_ece, test_preds, test_labels = edge.inner_update(self.global_model.classification.state_dict(), round_num)
           else:
               classification_grads, loss_update, train_accuracy, train_ece, train_preds, train_labels, test_loss, test_accuracy, test_ece, test_preds, test_labels = edge.inner_update(global_classification_weights, round_num)
           list_classification_grads.append(classification_grads)
           round_train_loss.append(loss_update)
           round_train_accuracy.append(train_accuracy)
           round_train_f1.append(self.compute_f1(train_preds, train_labels))
           round_test_loss.append(test_loss)
           round_test_accuracy.append(test_accuracy)
           round_test_f1.append(self.compute_f1(test_preds, test_labels))
           round_train_ece.append(train_ece)
           round_test_ece.append(test_ece)
           client_accuracies.append(test_accuracy)
           round_edges[edge.id] = {
                    "Train Loss": loss_update if not math.isnan(loss_update) else 0,
                    "Train Accuracy": train_accuracy,
                    "Train F1": self.compute_f1(train_preds, train_labels),
                    "Train ECE": train_ece if not math.isnan(train_ece) else 0,
                    "Test Loss": test_loss if not math.isnan(test_loss) else 0,
                    "Test Accuracy": test_accuracy,
                    "Test F1": self.compute_f1(test_preds, test_labels),
                    "Test ECE": test_ece if not math.isnan(test_ece) else 0,
                }
        # Calcul des moyennes pour ce round
        def safe_mean(data):
               return sum(data) / len(data) if len(data) > 0 and not math.isnan(sum(data)) else 0
            

        if (len(self.edges)>0):
                num_worst_clients = max(1, len(self.edges) // 10)  # Prend au moins un client
                worst_10_percent_acc = safe_mean(sorted(client_accuracies)[:num_worst_clients])
        else :
                worst_10_percent_acc = 0

        results = {
                "Round": round_num,
                "Train Loss": sum(round_train_loss) / len(round_train_loss) if not math.isnan(sum(round_train_loss) / len(round_train_loss)) else 0,
                "Train Accuracy": sum(round_train_accuracy) / len(round_train_accuracy),
                "Train F1": sum(round_train_f1) / len(round_train_f1),
                "Train ECE": sum(round_train_ece) / len(round_train_ece) if not math.isnan(sum(round_train_ece) / len(round_train_ece)) else 0,
                "Test Loss": sum(round_test_loss) / len(round_test_loss) if not math.isnan(sum(round_test_loss) / len(round_test_loss)) else 0,
                "Test Accuracy": sum(round_test_accuracy) / len(round_test_accuracy),
                "Test F1": sum(round_test_f1) / len(round_test_f1) ,
                "Test ECE": sum(round_test_ece) / len(round_test_ece) if not math.isnan(sum(round_test_ece) / len(round_test_ece)) else 0,
                "Worst 10% Accuracy": worst_10_percent_acc,
                "Edges": round_edges,
                
            }

        # Sauvegarder les résultats dans un fichier JSON
        with open(os.path.join(self.results_dir, f"round_{round_num + 1}.json"), "w") as f:
                json.dump(results, f, indent=4)

        print(f"Fog {self.id}: Round {round_num} - Results saved.")

        return list_classification_grads
                

           

