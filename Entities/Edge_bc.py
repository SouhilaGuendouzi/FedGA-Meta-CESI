from .Entities import Entities
import torch
from torch.autograd import Variable
from sklearn.metrics import f1_score, accuracy_score
from datetime import datetime
import json
import pickle
from Blockchain.TransactionRegistration import TransactionRG
from Blockchain.TransactionAuthentication import TransactionAUT
from Blockchain.TransactionModelTransfer import TransactionMT
from Enumeration.TransactionType import TransactionType
from Aggregation.FedGA import FedGA
from Blockchain.Block import Block
import ipfshttpclient
import copy
from Enumeration.MessageType import MessageType
import time
from tenacity import retry, wait_fixed, stop_after_attempt
import io

@retry(wait=wait_fixed(5), stop=stop_after_attempt(3)) # Retry 3 times with a 5 second wait between each attempt, 
class Edge(Entities):
    def __init__(self, id, ip_address, port, location, cpuMips, ramGB, storageGB, latencyMs, networkB, ai_task, do_it=True):
        # Correcting the super().__init__ call
        super().__init__(id, ip_address,port, location, cpuMips, ramGB, storageGB, latencyMs, networkB, ai_task)
        self.edges =  {}  # (id) ==> ipAdress, port, public_key
        self.fog = {
          "id": None,
            "ip_address": None,
            "port": None,
            "public_key": None
        }  
        self.local_model=None
        self.partial_model=None
        self.dataset=None
        self.authentication_results={}
        self.authentication_matrtix={}
        self.received_local_cid=[]
        self.received_models = {} # List of received models from other Edges
        self.received_partial_models = {} # List of received models from other Edges
        self.models_transactions = []  # List of model transactions
        self.first_round=True
        self.do_it=do_it


        self.authentication_transactions = []  # List of authentication transactions
        self.parameters_partial={}
        
        self.other_round=False
        self.this_round=0
        self.edge_decision=True

        use_cuda= torch.cuda.is_available()
        use_mps = torch.backends.mps.is_available()

        self.last_block=None

        self.state="available" #available, learning

        if use_cuda:
           self.device = torch.device("cuda")
        elif use_mps:
          self.device = torch.device("mps")
        else:
          self.device = torch.device("cpu")

       

    
    def attribute_edges(self, edges):
        # Sets the list of connected Edges P2P
        self.edges = edges

    def atribute_new_edge(self, edge):
        # Adds a new edge to the list of edges
        return self.edges.append(edge)
    

    def modify_edge(self, edge_to_modify, **changes):
        # Finds and modifies the specified edge with new parameters
        for i, edge in enumerate(self.edges):
            if edge == edge_to_modify:
                for key, value in changes.items():
                    if key in edge:
                        edge[key] = value
                return True  # Retmodelsurn True if the edge was modified
        return False  # Return False if the edge was not found

    def delete_edge(self, edge):
        # Removes a specified edge from the list of edges, if it exists
        if edge in self.edges:
            self.edges.remove(edge)
        return True  # Assuming removal is always successful if the edge exists
    def attribute_fog(self, fog):
        # Sets the connected Fog
        self.fog = fog
    def modify_fog(self, **changes):
        # Modifies attributes of the connected Fog with new parameters
        if self.fog is not None:
            for key, value in changes.items():
                if key in self.fog:
                    self.fog[key] = value
            return True  # Return True if the Fog was modified
        return False  # Return False if no Fog is connected
    

    def initialize_local_model(self,  
                             model, path):
        self.local_model= model
        self.partial_model=model
        self.create_json_file(path)
        

    def initialize_local_dataset(self,dataset):
        self.dataset=dataset  ##contains train and test

    def create_json_file(self, path):
        fileName = f"{path}/edge {self.id}.json"  
        with open(fileName, 'w') as fichier:
         self.local_model.training_history=fileName


    def train(self, optimizer, criterion, epochs):
      

      #print(self.id, 'training ...',self.device)
      self.local_model.model.to(self.device)
      self.local_model.model.train()
      optimizeR=torch.optim.Adam(self.local_model.model.parameters(), 0.001)
      for epoch in range(epochs):
        total_loss = 0
        all_targets = []
        all_predictions = []   
        for batch_idx, (x, target) in enumerate(self.dataset.train_data):
            optimizeR.zero_grad()
            
            x, target = x.to(self.device), target.to(self.device)
            x, target = Variable(x), Variable(target)
            if (self.dataset.name=='usps'):
                       vector=[]
                       for i in range(len(target)):
                         vector.append(target[i])
                       vector=torch.tensor(vector)#.cuda()
                       target=vector
                       x = torch.reshape(x, (len(x), 1, 28, 28))

            # Clear gradients before each backward pass
            optimizeR.zero_grad()
            self.local_model.model.train()  # Set model to training mode
            #if (next(self.local_model.model.parameters()).device=='cpu' or x.device=='cpu' or  target.device=='cpu' ):
            #    print(self.id, '<hhhhhhh>',next(self.local_model.model.parameters()).device, x.device, target.device)
            out = self.local_model.model(x)
            try: 
                loss = criterion(out, target)
            except:
                target=target.squeeze(1)
                loss = criterion(out, target)
            loss.backward()  # backpropagation
            optimizeR.step()  # update weights
            
            total_loss += loss.item()
            
            # Convert output probabilities to predicted class (0 or 1)
            preds = torch.argmax(out, dim=1)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
        
        # Calculate metrics after each epoch
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')  # 'weighted' accounts for label imbalance
        
        # Print epoch summary
        #print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(self.dataset.trainData)}, Accuracy: {accuracy}, F1 Score: {f1}')
      return  total_loss/len(self.dataset.train_data), accuracy, f1
      
    def test(self, criterion, model=None, fedga=False):
       
       #print(self.id, 'testing ...',self.device)
       
       model_for_test=copy.deepcopy(self.local_model.model)
       if model is not None:
           model_for_test=copy.deepcopy(model)  #Dans le cas des modèles partiellles
        
       
       model_for_test.to(self.device)
       model_for_test.eval() 
       total_loss = 0
       all_targets = []
       all_predictions = []
       if fedga: dataset=self.dataset.validation_data
       else:  dataset=self.dataset.test_data
       with torch.no_grad():  # Pas besoin de calculer les gradients pendant le test
        for batch_idx, (x, target) in enumerate(dataset):
            try : 
                x, target = x.to(self.device), target.to(self.device)
            except:
                pass
            x, target = Variable(x), Variable(target)


            if (self.dataset.name=='usps'):
                       vector=[]
                       for i in range(len(target)):
                         vector.append(target[i])
                       vector=torch.tensor(vector)#.cuda()
                       target=vector
                       x = torch.reshape(x, (len(x), 1, 28, 28))
            
            
            out = model_for_test(x)

            try: 
                loss = criterion(out, target)
            except:
                target=target.squeeze(1)
                loss = criterion(out, target)
            total_loss += loss.item()
            
            preds = torch.argmax(out, dim=1)
            all_targets.extend(target.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())
      
       accuracy = accuracy_score(all_targets, all_predictions)
       f1 = f1_score(all_targets, all_predictions, average='weighted')  # 'weighted' pour tenir compte des déséquilibres de classe
    
        # Affiche un résumé des performances
        #print(f'Test Loss: {total_loss/len(self.dataset.testData)}, Accuracy: {accuracy}, F1 Score: {f1}')
    
       self.local_model.model.train()  # Remet le modèle en mode entraînement
       return total_loss/len(self.dataset.test_data), accuracy,f1

    def learning(self,epochs,first_time=True, fedga=False, attack=False):
       self.state = 'learning'
       start_time = datetime.now()

       #Effectuer l'entraînement et obtenir les résultats
       #self.dataset.execute_augmentation_with_variation()
       train_loss, train_accuracy, trainF1 = self.train(optimizer=self.local_model.optimizer, criterion=self.local_model.loss_function, epochs=epochs)
       end_time = datetime.now()
       testLoss, testAccuracy, testF1 = self.test(criterion=self.local_model.loss_function,fedga=fedga)
       duration = end_time - start_time
       self.local_model.training_duration = duration
       date_str = start_time.isoformat()
       duration = duration.total_seconds()
       if attack: 
         self.local_model.simulate_attack_on_model()

       cid = self.save_model_to_ipfs(True)
    
       # Préparer les nouvelles données à ajouter
       FL_round = self.this_round - 1 if  first_time else self.this_round
       new_data = {
        'date': date_str,
        'model': cid,
        'duration': duration,
        'epochs': epochs,
        'TrainError': train_loss,
        'TrainAccuracy': train_accuracy,
        'TrainF1': trainF1,
        'TestError': testLoss,
        'TestAccuracy': testAccuracy,
        'TestF1': testF1,
        'FL_session': self.FL_session,
        'FL_round': FL_round

    }
       
       #print(self.id,new_data["TestAccuracy"])
       #print(self.id, '<hhhhhhhhhh>', first_time)
       if not first_time:
         # Lire le fichier JSON existant, ajouter les nouvelles données, puis réécrire le fichier
         try:
           with open(self.local_model.training_history, 'r') as json_file:
             existing_data = json.load(json_file)
        # Assurez-vous que le fichier contient une liste pour stocker plusieurs enregistrements
           if not isinstance(existing_data, list):
             existing_data = [existing_data]
           existing_data.append(new_data)
         except FileNotFoundError:
        # Si le fichier n'existe pas, commencez un nouveau fichier avec une liste
            existing_data = [new_data]
    
       # Écrire les données mises à jour dans le fichier
         with open(self.local_model.training_history, 'w') as json_file:
          json.dump(existing_data, json_file, indent=4)

        
       else :
        
        with open(self.local_model.training_history, 'w') as json_file:
         json.dump(new_data, json_file, indent=4)   

    
       return cid
        
    
    def start_FL(self, FL_rounds, FL_session, local_epochs, first_round):
         #print(self.id, 'starting FL ...',self.do_it)
         if (self.do_it):
            self.shard_chain.smart_contracts["aggregation"].fine_tuning=0
            self.shard_chain.smart_contracts["aggregation"].authorised_edges=0
            last_block=self.shard_chain.get_last_block()
            self.last_block=last_block
            self.first_round=first_round
               
            self.this_round+=1
            self.FL_rounds=FL_rounds
            self.FL_session=FL_session
            self.local_epochs=local_epochs
       
            transaction=self.create_authentication_transaction(FL_session, FL_rounds)
            self.share_authentication_transaction(transaction)
         else :
            #last_block=self.shard_chain.get_last_block()
            #self.last_block=last_block
            self.shard_chain.smart_contracts["aggregation"].fine_tuning=0
            self.first_round=first_round
               
            self.this_round+=1
            self.FL_rounds=FL_rounds
            self.FL_session=FL_session
            self.local_epochs=local_epochs
            self.participate_FL()
             

   
    def participate_FL(self):
        #print(self.id, 'participating FL ...')
        last_block=self.shard_chain.get_last_block()
        if (self.do_it):
          if (self.last_block!=None):
            while last_block.block_number==self.last_block.block_number:
                last_block=self.shard_chain.get_last_block()
                time.sleep(0.1)

        self.last_block=last_block  # the block of athenticiation transaction is added 
        
        if (self.do_it):
          if (self.this_round==1):
            cid=self.learning(self.local_epochs, True)
          else:
             cid=self.local_model.cid
        
        else :
             
             malicious_edges=self.shard_chain.smart_contracts["authentication"].malicious_edges
             total=len(self.edges)+1
             a=total-total*malicious_edges  ## number of honest edges (e.g 40)
             b=total-total*(malicious_edges/2) 
             malicious_edges=[int(edge.split("_")[2]) for edge in self.edges if int(edge.split("_")[2])>b and int(edge.split("_")[2])<=total]
             my_index=int(self.id.split("_")[2])
             if my_index>b and my_index<=total:
                 if (self.this_round==1):
                    cid=self.learning(self.local_epochs, first_time=True, fedga=False ,attack=True)
                 else:
                    cid=self.local_model.cid
             else :
                if (self.this_round==1):
                    cid=self.learning(self.local_epochs, first_time=True, fedga=False ,attack=False)
                else:
                    cid=self.local_model.cid
           
        #print(self.id,cid)
        self.state='available'
        transaction=self.create_modelTransfer_transaction(cid,"local",self.FL_session,self.FL_rounds)
        message={
                "subject": "model_transfer",
                "content":
                {
                 "transaction": transaction.to_dict(),
                }          
            }
        for edge in self.edges:
                self.send_message_to_peer(message, edge, self.edges[edge]["ip_address"], self.edges[edge]["port"], True)


    def handle_models(self,data):
      #print(self.id,"Received model", self.edge_decision)
      if (self.edge_decision):
        data=data["content"]
        transaction=data["transaction"]
        integrity=self.verify_transaction(transaction)
        #print(self.id,"Received model",integrity)
        if integrity:
            self.models_transactions.append(transaction)
            self.received_local_cid.append(transaction["model_ipfs"])
            #print(transaction["model_ipfs"])
            try :
              buffer = io.BytesIO(self.load_model_from_ipfs(transaction["model_ipfs"]))
              model = torch.load(buffer)
              self.received_models[transaction["id"]]=model
            except:
              print(self.id,"Error in loading model",transaction["model_ipfs"], transaction["id"])
              pass

       
        if len(self.received_models)==self.shard_chain.smart_contracts["aggregation"].authorised_edges-1:
            while self.state=="learning":
             time.sleep(0.1)
            self.received_local_cid.append(self.local_model.cid)
            cid=self.aggregate_models()
            self.received_local_cid.append(cid)
            time.sleep(0.1)
            transaction=self.create_modelTransfer_transaction(cid,"partial",self.FL_session,self.FL_rounds)
            message= {
                "subject": "model_transfer_partial",
                "content":
                {
                 "transaction": transaction.to_dict(),
                } 
            }
            for edge in self.edges:
                self.send_message_to_peer(message, edge, self.edges[edge]["ip_address"], self.edges[edge]["port"], True)



    def handle_partial_models(self,data):
      if (self.edge_decision):
        data=data["content"]
        transaction=data["transaction"]
        integrity=self.verify_transaction(transaction)
        #print(self.id,"Received partial model",integrity)
        cid=transaction["model_ipfs"]
        if integrity:
            #self.models_transactions.append(transaction)
            buffer = io.BytesIO(self.load_model_from_ipfs(transaction["model_ipfs"]))
            model = torch.load(buffer)
            test_model=copy.deepcopy(self.local_model)
            test_model.model.load_state_dict(model)
            #print(f"I am {self.id} and I received a partial model {test_model.print_last_layer()}")
            self.received_partial_models[transaction["id"]]=model

        if len(self.received_partial_models)==self.shard_chain.smart_contracts["aggregation"].authorised_edges-1:
            self.received_partial_models[self.id]=self.partial_model.model.state_dict()
            print(self.id, 'start testing partial models')
            for id_entity, model in self.received_partial_models.items():
                this_model=copy.deepcopy(self.local_model.model)
                this_model.load_state_dict(model)
                this_model.to(self.device)
                self.parameters_partial[id_entity]=[]
                self.dataset.create_validation_set()
                loss_partial, accuracy_partial,f1_partial=self.test(criterion=self.local_model.loss_function, model=this_model, fedga=True)
                self.parameters_partial[id_entity]=[loss_partial, accuracy_partial,f1_partial]
            
            self.my_results={
                    "id": self.id,
                    "parameters": self.parameters_partial
                }
            print(self.id, 'results', self.my_results)
            
            while True:
               best_partial_model=self.shard_chain.smart_contracts["aggregation"].choose_best_partial_model(self.my_results)
               if best_partial_model!=None: break
            self.shard_chain.smart_contracts["aggregation"].metrics={}
            if (best_partial_model==self.id):
                best_model_transaction=self.create_modelTransfer_transaction(self.received_local_cid[-1],"partial",self.FL_session,self.FL_rounds)
                self.models_transactions.append(best_model_transaction.to_dict())
                #print(self.id, 'I am the best partial model')
                block= Block(self.last_block.block_number + 1, self.last_block.hash,self.models_transactions, self.id)
                self.shard_chain.add_block(block)
                print(self.id, 'block added')

            else :
                pass
                #print(self.id, 'I am not the best partial model')	

            time.sleep(0.1)
            self.received_local_cid=[]
            self.received_models={}
            self.received_partial_models={}
            self.models_transactions=[]


           # while True:
           #     last_last_block = self.shard_chain.get_last_block()
           #     if last_last_block.block_number==last_block.block_number+1: break


         

            self.fine_tuning()

            #execute fine tuning
            #
            #
            #
            #
            #
            #
            #
            #
            #
            #self.start_FL(self.FL_rounds, self.FL_session, self.local_epochs)
            

            
    def fine_tuning(self):
        last_block=self.shard_chain.get_last_block()
        while last_block.block_number==self.last_block.block_number:
            last_block=self.shard_chain.get_last_block()
            time.sleep(0.1)
        self.last_block=last_block  # the block of athenticiation transaction is added 
        last_transactions=last_block.transactions[-1]
        partial_model_cid=last_transactions["model_ipfs"]
        
        buffer = io.BytesIO(self.load_model_from_ipfs(partial_model_cid))
        model = torch.load(buffer)

        self.local_model.model.load_state_dict(model)
        print(self.id,'start fine tuning ')
        self.learning(1, False, False)
        self.shard_chain.smart_contracts["aggregation"].fine_tuning+=1

        
        print(self.id,'fine tuning done')



    def create_authentication_transaction(self, session_id, round_id):    ## sessions_id here is the FL round+ The FL session
        data=[self.ai_task, self.dataset.size, session_id, round_id, self.trust]
        malicious_edges=self.shard_chain.smart_contracts["authentication"].malicious_edges
        
        total=len(self.edges)+1
        a=total-total*malicious_edges  ## number of honest edges (e.g 40)
        b=total-total*(malicious_edges/2) 
   
        transaction = TransactionAUT(self.id,self.id,self.ip_address, self.port, self.public_key,data)
        #create a list of index of malicious edges between a and b, knowing that the id is in fomr of 'edge_1_id'
       
        malicious_edges=[int(edge.split("_")[2]) for edge in self.edges if int(edge.split("_")[2])>a and int(edge.split("_")[2])<=total]
        my_index=int(self.id.split("_")[2])
        if my_index>a and my_index<=total:
            malicious_edges.append(my_index)
        if (my_index in malicious_edges) :
            new_data=copy.deepcopy(data)
            new_data.append("malicious")
            #print ("I am malicious sending false data ",self.id)
            signature=self.sign_data(transaction.data_to_sign_verify_integrity(new_data))
        else:
            signature=self.sign_data(transaction.data_to_sign_verify_integrity(data))
        transaction.signature=signature
        return transaction
    
    def share_authentication_transaction(self, transaction):
        message={
                "subject":"authenticate",
                 "content":
                 {
                    "transaction":transaction.to_dict(),
                    #"from_id":self.id,
                    #"trust":self.trust
                 }
                }
        for edge in self.edges:
            peer=self.edges[edge]
            #print(self.id,'I am sharing the authentication transaction with edge', edge)
            self.send_message_to_peer(message,edge, peer["ip_address"], peer["port"])

        self.send_message_to_peer(message,self.fog["id"], self.fog["ip_address"], self.fog["port"])

    def handle_authentication(self, data):
        data=data["content"]
        transaction=data["transaction"]
        malicious_edges=self.shard_chain.smart_contracts["authentication"].malicious_edges
        total=len(self.edges)+1
        a=total-total*malicious_edges  ## number of honest edges (e.g 40)
        b=total-total*(malicious_edges/2) 

        integrity=self.verify_transaction(transaction)
        malicious_edges=[int(edge.split("_")[2]) for edge in self.edges if int(edge.split("_")[2])>b and int(edge.split("_")[2])<=total]
        my_index=int(self.id.split("_")[2])
        if my_index>b and my_index<=total:
            malicious_edges.append(my_index)
        if (my_index in malicious_edges) :
            #print ("I am malicious sending false results",self.id)
            integrity=not integrity
        
        if (integrity and transaction["from_entity"] in self.edges and transaction["ai_task"]==self.ai_task):
                self.authentication_results[transaction["from_entity"]]=True
        else :
                self.authentication_results[transaction["from_entity"]]=False
        self.authentication_transactions.append(transaction)
        if len(self.authentication_results)==len(self.edges):  ## All edges have sent their results ==> synchrounous process
            message={
                "subject":"authenticate_result",
                    "content":{
                        "results":self.authentication_results,  ## Dict
                        "from_id":self.id,
                        "trust":self.trust
                    }
            }
            for edge in self.edges:
                peer=self.edges[edge]
                self.send_message_to_peer(message,edge, peer["ip_address"], peer["port"])      
            self.send_message_to_peer(message,self.fog["id"], self.fog["ip_address"], self.fog["port"])
        
    def handle_authentication_result(self, data):
        result = data["content"]
       
        if self.authentication_matrtix.get(result["from_id"])==None: 
            self.authentication_matrtix[result["from_id"]]=[]
        self.authentication_matrtix[result["from_id"]].append(result["results"])
        if len(self.authentication_matrtix)==len(self.edges)+1:
            ## appel smart contract and get the results, If I am authenticated then I can start the FL
            result={
                "id":self.id,
                "trust":self.trust,
                "matrix":self.authentication_results
            }
            
            results=self.shard_chain.smart_contracts["authentication"].receive_results(result)
            while results==None:
               results=self.shard_chain.smart_contracts["authentication"].receive_results(result)

            weights, decision=results["weights"], results["decision"]
            print(self.id, weights, decision,decision.count(1))
            self.shard_chain.smart_contracts["aggregation"].authorised_edges=decision.count(1)

            #print(self.id, self.shard_chain.smart_contracts["aggregation"].authorised_edges)



            

            index = int(self.id.split('_')[-1]) - 1  # On obtient '3', on convertit en int et on soustrait 1 pour obtenir l'index correct

             # Obtenir les valeurs correspondantes
            self.trust = weights[index]
            self.edge_decision = decision[index]

            self.edge_decision=True if self.edge_decision==1 else False

            if not self.edge_decision :
                print('Don t participate in the FL', self.id)

            else:
                is_highest = self.trust  == max(weights)
                if is_highest:
                    if self.shard_chain.smart_contracts["authentication"].stop_auth_consensus==False:
                      if  self.id == self.shard_chain.smart_contracts["authentication"].choose_entity(self.id):
                          last_block = self.shard_chain.get_last_block()
                          #print(self.authentication_transactions)
                          #last_block = self.shard_chain.get_last_block()
                          #while last_block.block_number == self.last_block.block_number:
                          #    last_block = self.shard_chain.get_last_block()

                          self.last_block=last_block
                          block=Block(self.last_block.block_number+ 1, self.last_block.hash, self.authentication_transactions, self.id)
                          self.shard_chain.add_block(block)
                          print("Block added", self.id)
                          self.shard_chain.smart_contracts["authentication"].stop_auth_consensus=False

                time.sleep(1)
                print("I am participating in the FL", self.id)
                self.participate_FL()
            
            self.authentication_matrtix={}
            self.authentication_results={}
            self.authentication_transactions=[]


                #self.start_FL(self.shard_chain.smart_contracts["authentication"].FL_rounds,  self.shard_chain.smart_contracts["authentication"].FL_session)

  
        


    def create_modelTransfer_transaction(self,cid,model_type,FL_session,FL_rounds):   
        
        transaction = TransactionMT(self.id, self.id, self.ip_address, self.port, self.public_key, cid,model_type,FL_session,FL_rounds)
        data=[cid,model_type,self.FL_session,self.FL_rounds]
        signature=self.sign_data(transaction.data_to_sign_verify_integrity(data))
        transaction.signature=signature
        return transaction
     

    def save_model_to_ipfs(self, local):
        if local == True:
          model_path="Storage/EdgeModels/"+str(self.id)+'_'+str(self.port)+"_local"+".pt"
          self.local_model.save_model(model_path)
        else : 
          model_path="Storage/EdgeModels/"+str(self.id)+'_'+str(self.port)+"_partial"+".pt"
          self.partial_model.save_model(model_path)
        

        with ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http') as client:
            # Ajoute le fichier du modèle à IPFS
            res = client.add(model_path)
            model_cid = res['Hash']
            #print(f"Model stored in IPFS with CID: {model_cid}")
            self.local_model.cid=model_cid
            return model_cid
        

    
    def load_model_from_ipfs(self, cid):
        #model_path="Storage/EdgeModels/"+str(self.id)+"_extern_"+str(cid)+".pt"
        with ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http') as client:
            variable=client.cat(cid) #.get
            #print(f"Model downloaded from IPFS with CID: {cid}")
            return variable
        
    
    def aggregate_models(self):
        ## Add Sockets Here
        print(self.id, "Start Aggregating Models")
        self.received_models[self.id]=self.local_model.model.state_dict()
        weights=[]
        for model, value in self.received_models.items():
            weights.append(value)
        
        model=self.local_model.model.to("cpu")
        dataset=self.dataset.create_validation_set()
        #(weights, model_, dataset_, dataset_name_, config=None):
        FL=FedGA(weights, copy.deepcopy(model),dataset,self.dataset.name)
        #partial_weight=FL.run(multi_objective=True)
        #print(self.id,"<hhhhhhhhhhhhh> \n" )
        #self.local_model.print_last_layer()
        partial_weight=FL
        self.partial_model.model.load_state_dict(partial_weight)
        time.sleep(1)
        #print(self.id,"<kkkkkkkkkkkkk> \n" )
        #self.partial_model.print_last_layer()
        cid=self.save_model_to_ipfs(False)
        print(self.id,'FedGA Done',self.test(criterion=self.local_model.loss_function, model=self.partial_model.model))

        
        
        #print(f"Iam {self.id} with this partial model {self.partial_model.print_last_layer()}")

        return cid


    ## Overriding the handle CLient method
    def handle_client(self, client_socket):
        with client_socket:
            while not self.stop_event.is_set():
                data = client_socket.recv(100000)              
                if data:
                    data_decoded = data #.decode('utf-8')
                   
                    try:                 
                       decoded_data = pickle.loads(data_decoded)  #json.loads(data_decoded)
                    except (pickle.UnpicklingError, EOFError) as e:
                      print(f"Erreur lors de la désérialisation avec pickle : {e}")
                    else:
                       pass
                  
                    if (decoded_data["subject"]== MessageType.FL_train): self.learning(10)
                    elif (decoded_data["subject"]== MessageType.register_result): self.receive_registration_result(decoded_data)
                    elif(decoded_data["subject"]== 'request'): self.handle_request(decoded_data)
                    elif (decoded_data["subject"]== "register_result"): self.receive_registration_result(decoded_data)
                    elif (decoded_data["subject"]== "authenticate"): self.handle_authentication(decoded_data)
                    elif (decoded_data["subject"]== "authenticate_result"): self.handle_authentication_result(decoded_data)
                    elif (decoded_data["subject"]== "model_transfer"): self.handle_models(decoded_data)
                    elif (decoded_data["subject"]== "model_transfer_partial"): self.handle_partial_models(decoded_data)
                    elif (decoded_data["subject"]== "test"): print(decoded_data["content"])

    
    def handle_request(self, data):
        peer=None
        data= data["content"]
        if 'edge' in data["id"]:
           peer =self.edges.get((data["id"]))
        elif 'fog' in data["id"]:
            peer =self.fog["id"]
        if peer:
                pass
        else:
            message={
             "subject": 'request',
              "content": {
                    "id": self.id,
                    "ip_address": self.ip_address,
                    "port": self.port,
                    "public_key": str(self.public_key),
              }
          }
            peer={      
                "ip_address": data["ip_address"],
                "port": data["port"],
                "public_key": data["public_key"],
               }
            if 'edge' in data["id"]:
               self.edges[(data["id"])]=peer
            elif 'fog' in data["id"]:
                self.fog={
                    "id":data["id"],
                    "ip_address": data["ip_address"],
                    "port": data["port"],
                    "public_key": data["public_key"],
                }
            self.send_message_to_peer(message, data["id"], data["ip_address"], data["port"])



    ########## Registration Tools  ##########   
    #    
    def broadcast_registration_request(self, peers, data):
        message={
            "subject":"register",
             "content":data.to_dict()
        }
        new_peers =[]
        for peer in peers:
           
            entity=peers.get(peer)
            new_peers.append((peer,entity["ip_address"], entity["port"]))
       
        self.send_request_to_unknown_peers(new_peers, message)
    

    def broadcast_registration_request_pow(self, peers, data):
        message={
            "subject":"register_pow",
             "content":data.to_dict()
        }
        new_peers =[]
        for peer in peers:
           
            entity=peers.get(peer)
            new_peers.append((peer,entity["ip_address"], entity["port"]))
       
        self.send_request_to_unknown_peers(new_peers, message)
        
    def create_registration_transaction(self): 
        data=[self.location[0], self.location[1], self.ai_task]
        transaction = TransactionRG(self.id, self.id,self.ip_address, self.port, self.public_key,data)
        signature=self.sign_data(transaction.data_to_sign_verify_integrity(data))
        transaction.signature=signature
   
        return transaction
    

    def receive_registration_result(self, answer): 
        data = answer["content"]
        if data["status"]==True:
            self.shard_chain= data["shard_chain"] # shard_chain Instance if possible
            #threading.Thread(target=self.hear_smart_contract, daemon=True).start()
            peers= data["peers"] # id, ip, port, public_key
            self.fog={
                "id": data["fog_id"],
                "ip_address": data["fog_adr"],
                 "port": data["fog_port"],
                  "public_key":data["fog_publickey"]    
                 }
            
            for id in peers:
                message={
                    "subject": 'request',
                    "content": {
                        "id": self.id,
                        "ip_address": self.ip_address,
                        "port": self.port,
                        "public_key": str(self.public_key),
                    }
                }
                self.send_message_to_peer(message,self.fog["id"],self.fog["ip_address"], self.fog["port"], True)
                peer= peers.get(id)
                self.send_message_to_peer(message, id, peer["ip_address"], peer["port"], True)
                self.edges[id]=peer


            print(self.id,"Registration Done")
        else :
            print("Registration Failed")
        








     
     
     
    