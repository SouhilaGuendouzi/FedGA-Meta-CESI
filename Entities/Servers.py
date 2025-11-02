import socket
import threading
import select
import pickle

class Peer:

    def __init__(self, id_entity, host, port, known_peers):
        self.id_entity = id_entity
        self.host = host
        self.port = port
        self.known_peers = {}  # Dictionnaire pour stocker les sockets des pairs connus
        for peer_host, peer_port in known_peers:
            self.known_peers[(peer_host, peer_port)] = {"peer_id":None, "peer_socket":None}  # Initialiser avec None
        self.stop_event = threading.Event()

        # Créer un socket TCP/IP
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.start_server()

    def start_server(self):
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(100)
        #print(f"{self.id_entity} écoute à l'adresse {self.host}:{self.port}")
        threading.Thread(target=self.listen, daemon=True).start()

    def listen(self):
        # Écouter les connexions entrantes
        threading.Thread(target=self.accept_connections, daemon=True).start()
        
        while not self.stop_event.is_set():
            readable, _, _ = select.select([self.server_socket], [], [], 0.1)
            for s in readable:
                client_socket, addr = s.accept()
                threading.Thread(target=self.handle_client, args=(client_socket,), daemon=True).start()

    def handle_client(self, client_socket):

        with client_socket:
            while not self.stop_event.is_set():
                data = client_socket.recv(1024)
                if data:
                    print
                    #print(f"{self.id_entity}: Message de {client_socket.getpeername()}: {data.decode('utf-8')}")
                else:
                    break
        

    def accept_connections(self):
        while not self.stop_event.is_set():
            try:
                client_socket, addr = self.server_socket.accept()
                #print(f"{self.id_entity}: Connexion entrante de {addr}")
                threading.Thread(target=self.handle_client, args=(client_socket,), daemon=True).start()
            except Exception as e:
                #print(f"{self.id_entity}:", e)
                break



    def send_message_to_all_peers(self, message):
        for peer_host, peer_port in self.known_peers:
            peer=self.known_peers.get((peer_host, peer_port))
            self.send_message_to_peer(message,peer["peer_id"] ,peer_host, peer_port,True)


    def send_request_to_unknown_peers(self, peers, message):
        for peer_id, peer_host, peer_port in peers:
            self.send_message_to_peer(message,peer_id, peer_host, peer_port,False)


    def stop(self):
     self.stop_event.set()
     self.server_socket.close()
    # Fermer proprement tous les sockets clients
     for peer_info in self.known_peers.values():
        peer_socket = peer_info["peer_socket"]
        if peer_socket:
            peer_socket.close()

    def send_message_to_peer(self, message, id_entity, peer_host, peer_port,known_peer=True):
      operation=False
      
      if known_peer:
        peer_info = self.known_peers.get((peer_host, peer_port))
        if not peer_info or not peer_info["peer_socket"]:
           try:
              peer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
              #peer_socket.bind((self.host, peer_port-self.port+2000))  ## Critère de vérification 
              peer_socket.connect((peer_host, peer_port))
              self.known_peers[(peer_host, peer_port)] = {"peer_id": id_entity, "peer_socket": peer_socket}
              operation=True
           except ConnectionRefusedError:
              print(f"{self.id_entity}: Le pair {peer_host}:{peer_port} n'est pas disponible")
       

        try:
          peer_socket = self.known_peers[(peer_host, peer_port)]["peer_socket"]
          data=pickle.dumps(message)
          peer_socket.sendall(data) #data.encode('utf-8')
          #print(f"{self.id_entity}: Envoyé à {peer_host}:{peer_port}")
          operation=True
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
          print(f"{self.id_entity}: Erreur lors de l'envoi à {peer_host}:{peer_port}. Détail de l'erreur : {e}")
          # Assurez-vous de fermer le socket et de le retirer de la liste des pairs connus
          peer_socket.close()
          del self.known_peers[(peer_host, peer_port)]
         
      else:
        ## ici, je ne veux pas inclure la connnection dans la liste des pairs connus puisque c'est unee connection temporraire, c'est tout
        try:
          #print(f"Connexion à {peer_host}:{peer_port}")
          peer_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
          peer_socket.connect((peer_host, peer_port))
          ## Attention change le message ceci est un message de demande de regsitration
          data = pickle.dumps(message)
          peer_socket.sendall(data) #data.encode('utf-8')
          #print(f"{self.id_entity}: Envoyé à {peer_host}:{peer_port}")
          peer_socket.close()
          operation=True
        except ConnectionRefusedError:
          print(f"{self.id_entity}: Le pair {peer_host}:{peer_port} n'est pas disponible")
        
        return operation



      


