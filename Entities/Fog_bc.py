from .Entities  import Entities
#from Models.Model_abs import Model
from Enumeration.ModelScope import ModelScope
import ipfshttpclient
from Blockchain.ShardChain import ShardChain
from Blockchain.TransactionModelTransfer import TransactionMT
from Blockchain.Block import Block
from Entities.Location import Location
from Enumeration.TransactionType import TransactionType
from Enumeration.MessageType import MessageType
import random
import pickle
import time
import psutil
class Fog(Entities):

    def __init__(self, id_entity, ip_address, port, location, cpu_mips, ram_GB, storage_GB, latency_Ms, network_B, ai_task, capacity_edges):
        # Correcting the super().__init__ call
        super().__init__(id_entity, ip_address, port, location, cpu_mips, ram_GB, storage_GB, latency_Ms, network_B, ai_task)
        self.edges = {}  # Initialize the list of edges
        self.fogs = {}   # Initialize the list of connected Fogs
        self.capacity_edges=capacity_edges
        self.partial_model = None
        self.global_model = None
        self.admin = None

         ###### Other needed attributes
        self.count_transactions=0
        self.registration_results = {}
        self.registration_transactions=  {}
        self.received_registration_results=  {}
        self.temp_edges={}
        self.execution_time=0
        self.block_results = {}
        self.waiting_block=None
        self.authentication_matrix={}
        self.authentication_results={}
        self.authentication_transactions = []  # List of authentication transactions
        self.total_transactions=0
        self.many_times=0
        self.wait=False
        self.list_copy={}

        

    def attribute_new_edge(self, edge):
        # Adds a new edge to the list of edges
        return self.edges.append(edge)
    
    def attribute_edge(self, edges):
        # Sets the list of connected Edges P2P
        self.edges = edges
        return self.edges
   
    def delete_edge(self, edge):
        # Removes a specified edge from the list of edges, if it exists
        if edge in self.edges:
            self.edges.remove(edge)
        return self.edges

    def modify_edge(self, edge_to_modify, **changes):
        # Finds and modifies the specified edge with new parameters
        for edge in self.edges:
            if edge == edge_to_modify:
                for key, value in changes.items():
                    if key in edge:
                        edge[key] = value
                return True  # Return True if the edge was modified
        return False  # Return False if the edge was not found

    def attribute_fogs(self, fogs):
        # Sets the list of connected Fogs
        self.fogs = fogs
        return self.fogs

    def attribute_fog(self, fog):
        # Adds a new Fog to the list of connected Fogs
        self.fogs.append(fog)
        return self.fogs
    
    def modify_fog(self, fog_to_modify, **changes):
        # Finds and modifies the specified Fog with new parameters
        for fog in self.fogs:
            if fog == fog_to_modify:
                for key, value in changes.items():
                    if key in fog:
                        fog[key] = value
                return True  # Return True if the Fog was modified
        return False  # Return False if the Fog was not found

    def delete_fog(self, fog):
        # Removes a specified Fog from the list of Fogs, if it exists
        if fog in self.fogs:
            self.fogs.remove(fog)
        return self.fogs
    
    """
    def initialize_partial_model(self,                                  
                                    type,
                                    model,
                                   ):
            self.partial_model = Model(self.id_entity, ModelScope.PARTIAL, type, model, None, None, None, None, self.device)

    
    def initialize_global_model(self,                                  
                                    type,
                                    model,
                                   ):
            self.global_model = Model(self.id_entity, ModelScope.GLOBAL, type, model, None, None, None, None, self.device)
    """

    def save_model_to_ipfs(self, partial):
        if partial == True:
          model_path="Storage/FogModels/fog"+str(self.id_entity)+"_partial"+".pt"
          self.localModel.SaveModel(model_path)
        else : 
          model_path="Storage/FogModels/fog"+str(self.id_entity)+"_global"+".pt"
          self.partial_model.SaveModel(model_path)

        with ipfshttpclient.connect() as client:
            # Ajoute le fichier du modèle à IPFS
            res = client.add(model_path)
            model_cid = res['Hash']
            print(f"Model stored in IPFS with CID: {model_cid}")
            return model_cid
        

    def load_model_from_ipfs(self, cid):
        model_path="Storage/FogModels/fog"+str(self.id_entity)+"_extern"+str(cid)+".pt"
        with ipfshttpclient.connect() as client:
            client.get(cid, model_path)
            print(f"Model downloaded from IPFS with CID: {cid}")
            return model_path

   
    def initialize_shard_chain(self, hash, consensus_algorithm, max_block_size):
 
        self.shard_chain=ShardChain(consensus_algorithm, hash, max_block_size, self.id)
    

    def create_modelTransfer_transaction(self):  ## Transfer The parial model
        cid = self.save_model_to_ipfs(True)
        data=str(self.ip_addrss)+str(self.public_key)+str(cid)
        transaction = TransactionMT(self.ip_addrss, self.public_key, self.sign_data(data), cid)
        return transaction

    def receive_transaction(self, transaction):
        transaction.status=self.verify_transaction(transaction)
        if transaction.status==True:
           if transaction.type==TransactionType.REGISTRATION:
             self.shard_chain.self.smart_contracts["regitration"].notify_fog_about_transaction(transaction)
           elif transaction.type==TransactionType.AUTHENTICATION:
               self.shard_chain.smart_contracts["authentication"].receive_transaction(transaction)

  
    def evaluate_registration_assignment(self, location, ai_task):
        result = {
            "accepted": False,
            "distance": None,
            "fog_id": self.id
        }
        if self.ai_task == ai_task:
           if (len(self.edges)<=self.capacity_edges):
               self_location=Location(self.location[0], self.location[1])
               location=Location(location[0], location[1])
               LocalisationDistance= self_location.distance_to(location)

               result["distance"] = LocalisationDistance
               result["accepted"] = True
        #print(f"Result of evaluation {self.id}: {result}")
        return result

    def send_registration_results(self):
        items = list(self.registration_results.items())
        if len(items) == 0:
            items=list(self.registration_transactions.items())
        tableau = [dict(items[i:i+self.main_chain.max_block_size]) for i in range(0, len(items), self.main_chain.max_block_size)]
        round = 0
        if self.main_chain.consensus == "PoR":
            for batch in tableau:
               
                while self.wait:
                    time.sleep(0.1)
                if not self.wait:
                    self.list_copy = batch.copy()                                                                                                               
                    #print(f"list_copy by {self.id}: ", len(self.list_copy))
                    round += 1
                    self.wait = True 
                    
                    for fog_id, peer in self.fogs.items():
                        message = {
                            "subject": "register_result",
                            "content": {"fog_id": self.id, "results": self.list_copy}
                        }
                       
                        self.send_message_to_peer(message, fog_id, peer["ip_address"], peer["port"], True)

        elif self.main_chain.consensus == "PoW":
            #print(f"Tableau by {self.id}: ", tableau)
            round = 0
            for batch in tableau:
                round += 1
                while self.wait:
                    time.sleep(0.1)
                if not self.wait:
                    self.wait = True  # Devrait être réinitialisé quelque part
                    new_list = [self.registration_transactions[id_transaction] for id_transaction in batch]
                    while not self.main_chain.smart_contracts["registration"].block_added:
                        time.sleep(0.1)  # Petite pause pour éviter la surcharge

                    last_block = self.main_chain.get_last_block()
                    self.waiting_block = Block(last_block.block_number + 1, last_block.hash, new_list, self.id)
                    nonce = self.waiting_block.proof_of_work(2)
                    #print(f"Block by {self.id}: nonce {nonce}, round {round}")
                    if not self.main_chain.smart_contracts["registration"].stop_pow and self.main_chain.smart_contracts["registration"].block_added:
                        if self.main_chain.get_last_block().previous_hash != self.waiting_block.previous_hash:
                            winner = self.main_chain.smart_contracts["registration"].pow_consensus(self.id, self.waiting_block,round)
                            if winner == self.id:
                                message = {
                                    "subject": "pow_winner",
                                    "content": {"fog_id": self.id, "block": self.waiting_block}
                                }
                                print(f"Winner is {self.id} with nonce {nonce}", self.waiting_block.transactions)
                                for fog_id, peer in self.fogs.items():
                                    self.send_message_to_peer(message, fog_id, peer["ip_address"], peer["port"], True)
    
  
    def handle_registration(self, data):
          #print(f"Handling registration transaction {data}")
          if (len(self.registration_results)==0 or len(self.block_results)==0):  self.execution_time = time.perf_counter()
          transaction =data["content"]
          evaluation=self.shard_chain.smart_contracts["registration"].evaluate_transaction(transaction)
          if evaluation: 
            test=self.verify_transaction(transaction)
            if (test == True):
              if (data["subject"]== 'register'):
                result= self.evaluate_registration_assignment([transaction["location_lt"],transaction["location_lg"]], transaction["ai_task"])
                
                if ( self.registration_results.get(transaction["id"])) == None: 
                  self.registration_results[(transaction["id"])]=[]
                  self.registration_results[(transaction["id"])].append(result)
                else :
                   self.registration_results[(transaction["id"])].append(result)

                # ici il me faut un truc pour lancer dès qu'on atteint la taille du bloc
                self.registration_transactions[(transaction["id"])]=transaction
                
              
                self.temp_edges[(transaction["from_entity"])]={
                    "ip_address":transaction["from_address"],
                      "port":transaction["from_port"]
                }
              
                if len(self.registration_transactions)== self.total_transactions: 
                    self.send_registration_results()

              elif (data["subject"]== 'register_pow'):
                  if len(self.registration_transactions)==0: self.execution_time = time.perf_counter()
                  self.registration_transactions[(transaction["id"])]=transaction
                
                  if len(self.registration_transactions)==self.total_transactions:
                            self.send_registration_results()

            else :
                print("Transaction not verified")

    def receive_regitration_results(self, data):

        data=data["content"]
        #print(f"{self.id} Received registration results from {data['fog_id']}")
        
        self.received_registration_results[(data["fog_id"])]=data["results"].copy()
        
        if len(self.received_registration_results)==len(self.fogs):
            
            time.sleep(0.5)
            self.received_registration_results[(self.id)]=self.list_copy.copy()
            
            # Nouvelle structure de données avec aggregation par edge
            edge_aggregated = {}
            for fog, edges in self.received_registration_results.items():
                for edge, values in edges.items():
                       if edge not in edge_aggregated:
                           edge_aggregated[edge] = []
                       edge_aggregated[edge].extend(values)

            list_results = self.shard_chain.smart_contracts["registration"].evaluate_registration_submissions(edge_aggregated)
            
            
            for tranaction_id in list_results :
               if list_results.get(tranaction_id)==self.id:  
                   message={
                       "subject":"register_result",
                       "content":
                       {   "status":True,
                           "shard_chain":self.shard_chain,
                           "peers":self.edges,
                           "fog_id":self.id,
                           "fog_adr":   self.ip_address,
                           "fog_port":self.port,
                            "fog_publickey":str(self.public_key)
                       }
                   }
                   transaction=self.registration_transactions.get(tranaction_id)
                   transaction["approvalStatus"]="CONFIRMED"       
                   peer=self.temp_edges.get(transaction["from_entity"])

                   self.send_message_to_peer(message,transaction["from_entity"],peer["ip_address"],int(peer["port"]),True)

               elif list_results.get(tranaction_id)==None:  # No Fog was selected, notify the admin system         
                   # Notify the admin systemmmmmmmmmmmmmmmmmmmm
                   self.registration_results.pop(tranaction_id)
                   

            
            new_list=[]
            
            for id_transaction in list_results:
               
                transaction=self.registration_transactions.get(id_transaction)
                new_list.append(transaction)
                #self.registration_results.pop(id_transaction)
                #self.registration_transactions.pop(id_transaction)
            self.received_registration_results.clear()

            
            #if (len(new_list)>self.main_chain.max_block_size):  new_list.pop() # remove the last transaction
            last_block=self.main_chain.get_last_block()   
            #print(self.id, 'tRansaction prepared, new list', new_list)
            self.waiting_block=Block(last_block.block_number+1, last_block.hash, new_list, self.id,self.main_chain.max_block_size)
            self.waiting_block.hash=self.waiting_block.calculate_hash()

            random_fog=self.main_chain.smart_contracts["registration"].por_consensus(len(self.fogs)+1)
            #print(self.id, random_fog,self.main_chain.smart_contracts["registration"].fixe )
            if random_fog==self.id:
                
                
                print('I am the random Fog', self.id, self.waiting_block.transactions, list_results)
                message={
                         "subject":"por_winner",
                          "content":
                                {
                                  "fog_id":self.id,
                                   "block":self.waiting_block
                                }
                        }
                for id in self.fogs:
                            peer =self.fogs.get(id)
                            self.send_message_to_peer(message, id, peer["ip_address"], peer["port"], True)
         

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
                    if (decoded_data["subject"]== 'register'):
                        self.handle_registration(decoded_data)
                    elif (decoded_data["subject"]== 'register_pow'):
                        self.handle_registration(decoded_data)
                    elif (decoded_data["subject"]== 'register_result'):
                        self.receive_regitration_results(decoded_data)
                    elif (decoded_data["subject"]== "authenticate"): self.handle_authentication(decoded_data)
                    elif (decoded_data["subject"]== "authenticate_result"): self.handle_authentication_result(decoded_data)
                    elif (decoded_data["subject"]== 'request'):
                        self.handle_request(decoded_data)
                    elif (decoded_data["subject"]== 'por_winner'):
                        self.receive_block_for_verification(decoded_data)
                    elif(decoded_data["subject"]== 'pow_winner'):
                        self.receive_block_for_verification(decoded_data)
                    elif (decoded_data["subject"]== 'block_result'):
                        self.receive_results_block_verification(decoded_data)
                    elif (decoded_data["subject"]== "test"):
                            pass
                    
    def handle_authentication(self, data):
        data=data["content"]
        transaction=data["transaction"]
        integrity=self.verify_transaction(transaction)
        #random_trust=random.choice([True, False])
        random_trust=True
        if (integrity and transaction["from_entity"] in self.edges and transaction["ai_task"]==self.ai_task):
                self.authentication_results[transaction["from_entity"]]=True and random_trust
        else :
                self.authentication_results[transaction["from_entity"]]=False and random_trust
        self.authentication_transactions.append(transaction)
        if len(self.authentication_results)==len(self.edges):
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
        
    def handle_authentication_result(self, data):
        result = data["content"]
        if self.authentication_matrix.get(result["from_id"])==None: 
            self.authentication_matrix[result["from_id"]]=[]
        self.authentication_matrix[result["from_id"]].append(result["results"])
        if len(self.authentication_matrix)==len(self.edges):
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
            self.trust = weights[-1]
            is_highest = self.trust  == max(weights)
            if is_highest:
              if self.shard_chain.smart_contracts["authentication"].stop_auth_consensus==False:
                if  self.id == self.shard_chain.smart_contracts["authentication"].choose_entity(self.id):
                          last_block = self.shard_chain.get_last_block()
                          block=Block(last_block.block_number + 1, last_block.hash, self.authentication_transactions,self.id)
                          self.shard_chain.add_block(block)
                          self.shard_chain.smart_contracts["authentication"].stop_auth_consensus=False
                          print("Block added", self.id)
              else : print ('consenus is True', self.id)

              self.authentication_results={}
              self.authentication_matrix={}
              self.authentication_transactions=[]
              

    def receive_results_block_verification(self,data):
        result=data["content"]
        self.block_results[result["fog_id"]]=result["status"]

        #vérifie si le block n'existe pas déja dans la blockchain
       
       
        if len(self.block_results)==len(self.fogs):
            print('All results received of block verification',self.id, self.block_results)
            good=True
            for fog_id in self.block_results:
                if self.block_results[fog_id]==False:
                    good=False
                    break
           
            last_block=self.main_chain.get_last_block()
            if last_block.previous_hash!=self.waiting_block.previous_hash:
             if good==True:
                self.main_chain.add_block(self.waiting_block)
                print('Block added to the main chain',self.id)
                self.execution_time = time.perf_counter()-self.execution_time
                self.main_chain.smart_contracts["registration"].set_execution_time(self.execution_time)                 
                self.count_transactions=0           
                self.block_results={}
                transactions=self.waiting_block.transactions
                for transaction in transactions:
                    print('Transaction',transaction)
                self.main_chain.smart_contracts["registration"].fixe=False
                self.main_chain.smart_contracts["registration"].stop_pow=False
                self.main_chain.smart_contracts["registration"].block_added=True
                self.wait=False
                transactions=self.waiting_block.transactions

                ### C'est ici que je devrai informer les edgessssssss


    def receive_block_for_verification(self,data):
        data = data["content"]
        block = data["block"]
        previous_block=self.main_chain.get_last_block()
        test_block=block.verify_block(previous_block)
        #print('Block received for verification',self.id, block)
        

        message ={
                "subject":"block_result",
                "content":
                {
                    "status":test_block,
                    "fog_id":self.id
                }
            }
       
        for fog in self.fogs:
            if fog==data["fog_id"]:
                self.send_message_to_peer(message,fog, self.fogs[fog]["ip_address"],int( self.fogs[fog]["port"]),True)
       
        if test_block:
            transactions=block.transactions
            for transaction in transactions:
                print('Transaction received',self.id, transaction) 

        time.sleep(0.5)
        self.wait=False



    def handle_request(self, data):
        data= data["content"]
        if "edge" in data["id"] :
           peer =self.edges.get((data["id"]))
        elif "fog" in data["id"]:
              peer =self.fogs.get((data["id"]))
        elif "admin"    in data["id"]:
                peer =self.admin
        if not peer:
            message={
             "subject": "request",
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
            if "edge" in data["id"] :
                  self.edges[(data["id"])]=peer
            elif "fog" in data["id"]:
                    self.fogs[(data["id"])]=peer
            elif "admin"    in data["id"]:
                    self.admin=peer

            self.send_message_to_peer(message, data["id"], data["ip_address"], data["port"])       
        else:
            pass
            #print("Peer already exists")
                
    
    

    def cpu_usage(self):
       return psutil.cpu_percent(interval=1)  # Measures CPU percent usage over one second
    



    def inform_edges_about_reponses(self, transactions, results):
        print(self.id, results)

