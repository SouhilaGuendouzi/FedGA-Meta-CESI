from Entities.Servers import Peer
from Entities.Fog_bc import Fog
import pickle
import random
class admin(Peer):
   
   def __init__(self, host, port):
       super().__init__('admin', host, port)
       self.fogs = {}

    
   def handle_client(self, client_socket):
        with client_socket:
            while not self.stop_event.is_set():
                data = client_socket.recv(10000)
                if data:
                    message = pickle.loads(data)
                    print(f"{self.idEntity}: Message de {client_socket.getpeername()}: {message}")
                    if message["subject"] == "allocation":
                          data= message["content"]
                          fog_attributs = [random.randint(500, 1000) for _ in range(5)]
                          fog_latitude = random.uniform(-90, 90)
                          fog_longitude = random.uniform(-180, 180)
                          fog= Fog('fog_6', '127.0.0.1', 6500, [fog_latitude, fog_longitude], *fog_attributs, data["task"], random.randint(1,100))
                          message = {"subject": "allocation", "content": fog}
                          self.send_message_to_all_peers(message)
                          return fog
                    
                    elif message["subject"] == "request":
                          
                          self.handle_request(message["content"])


   def handle_request(self, data):
        data= data["content"]
        if "fog" in data["id"]:
              peer =self.fogs.get((data["id"]))
        if not peer:
            message={
             "subject": "request",
              "content": {
                    "id": self.id,
                    "ipAddress": self.ipAddress,
                    "port": self.port,
              }
          }
            
            peer={      
                "ipAddress": data["ipAddress"],
                "port": data["port"],
                "publicKey": data["publicKey"],
              }
            if "fog" in data["id"]:
                    self.fogs[(data["id"])]=peer

            self.send_message_to_peer(message, data["id"], data["ipAddress"], data["port"])
        
        else:
            print("Peer already exists")
                
                       
                       
                 
      