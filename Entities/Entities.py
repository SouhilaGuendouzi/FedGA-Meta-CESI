
from Enumeration.EntityStatus import  EntityStatus
from .Servers import Peer
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import base64
import torch
from Blockchain.MainChain import MainChain
from Blockchain.ShardChain import ShardChain


class Entities(Peer):
    def __init__(self,
                  id_entity,
                  ip_address,
                  port,
                  location, 
                  cpu_mips,
                  ram_GB, 
                  storage_GB,
                  latency_Ms,
                  network_B, 
                  ai_task) :
        super().__init__(id_entity,ip_address, port,[])
        self.id=id_entity
        self.ip_address=ip_address
        self.location=location
        self.cpu_mips=cpu_mips
        self.ram_GB=ram_GB
        self.storage_GB=storage_GB
        self.latency_Ms=latency_Ms
        self.network_B=network_B
        self.ai_task= ai_task
        self.port=port
        self.main_chain=None
        self.shard_chain=None
    
        ##### Others 

        self.status=EntityStatus.ONLINE
        self.trust=0
        self.public_key=0
        self.private_key=0
        self.use_cuda= torch.cuda.is_available()
        self.use_mps = torch.backends.mps.is_available()
        if self.use_cuda:
            self.device = torch.device("cuda")
        elif self.use_mps:
              self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")



        self.key_generation()



    def key_generation(self):
         # Générer la clé privée EdDSA (Ed25519)
        self.private_key = ed25519.Ed25519PrivateKey.generate()
    
          # Générer la clé publique correspondante
        self.public_key =  self.private_key.public_key()

        return self.private_key, self.public_key
    
    def sign_data(self, data):
    # Assurez-vous que les données sont des octets
     if isinstance(data, str):
        data = data.encode('utf-8')
     signature = self.private_key.sign(data)
     return signature

    def verify_signature(self, public_key_hex, message, signature_base64):
    # Convertir la chaîne hexadécimale de la clé publique en bytes
      public_key_bytes = bytes.fromhex(public_key_hex)
    # Recréer l'objet de la clé publique à partir des bytes
      public_key = ed25519.Ed25519PublicKey.from_public_bytes(public_key_bytes)
    
    # Assurez-vous que le message est des octets pour la vérification
      if isinstance(message, str):
        message = message.encode('utf-8')
    
    # Convertir la signature de base64 en bytes
      signature = base64.b64decode(signature_base64)
  
    
      try:
        # Effectuer la vérification de la signature
        public_key.verify(signature, message)
        return True
      except InvalidSignature:
        # La vérification de la signature a échoué
        return False
      except Exception as e:
        # Pour gérer tout autre type d'erreur qui pourrait survenir
        print(f"Erreur lors de la vérification de la signature: {e}")
        return False
     
    def hash_256(message):
      # Calculer le hash SHA-256 du message
      digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
      digest.update(message)
      return digest.finalize()
    
    def verify_transaction(self, transaction):
      
    # Initialiser data_to_verify en fonction du type de transaction
      if hasattr(transaction, 'data_to_sign_verify_integrity'):  # Si transaction est un objet
        data_to_verify = transaction.data_to_sign_verify_integrity()
        public_key = transaction.public_key
        signature = transaction.signature
      else:  # Si transaction est un dictionnaire
        data_to_verify = ""
        for key, value in transaction.items():
            if key not in ["signature", "public_key", "status", "approval_status"]:
                data_to_verify += str(value)
        public_key = transaction["public_key"]
        signature = transaction["signature"]

    # Vérifier la signature de la transaction
      try:
        status = self.verify_signature(public_key, data_to_verify, signature)
      except Exception as e:
        print(f"Erreur lors de la vérification de la signature : {e}")
        status = False  # Assumer que la transaction est rejetée en cas d'erreur

    # Mettre à jour le statut de la transaction en fonction de la vérification
      if status:
        new_status = "CONFIRMED"
      else:
        new_status = "REJECTED"

    # Appliquer le nouveau statut à la transaction
      if hasattr(transaction, 'status'):  # Si transaction est un objet
        transaction.status = new_status
      else:  # Si transaction est un dictionnaire
        transaction["status"] = new_status

      return status


    
    ## Proof of Authentication 
   


        
    
  
           
           
            
           

     

    


        
        