
from torchvision.datasets import MNIST
from torchvision import transforms
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try: 
  from data.createSets import iid_split, non_iid_split, non_equitable_random_split, dirichlet_non_iid_split
except: 
  from createSets import iid_split, non_iid_split, non_equitable_random_split, dirichlet_non_iid_split
  
from torch.utils.data import ConcatDataset
import torch

def transform_target(y):
    return torch.tensor([y])

def get_mnist(path,train=True):

      transform_list = [transforms.ToTensor() ,transforms.Resize((28,28))]
      mnist_dataset = MNIST(root=path, train=train,
         download=True,transform=transforms.Compose(transform_list),  target_transform=transform_target ) # Ensure shape is (1,)) 

      
      return  mnist_dataset



def get_mnist_small(path,train=True):

      transform_list = [transforms.ToTensor() ,transforms.Resize((28,28))]
      mnist_dataset = MNIST(root=path, train=train,
         download=True,transform=transforms.Compose(transform_list)) 

      mnist_dataset.data = mnist_dataset.data[:1000]
      mnist_dataset.targets = mnist_dataset.targets[:1000]
      
      return  mnist_dataset

"""
def get_mnist_distribution(type="iid", n_samples_train=200, n_samples_test=100, n_clients=3, batch_size=25, shuffle=True):
    dataset_loaded_train = get_mnist("./data",train=True)
    dataset_loaded_test = get_mnist("./data",train=False)
    #merge the two datasets and put them into one new dataset variable
    global_dataset = ConcatDataset([dataset_loaded_train, dataset_loaded_test])
    n_samples=len(global_dataset)*0.15
  
    train, test, index, node_labels = None, None, None, None


    if type=="iid":
        train=iid_split(dataset_loaded_train, n_clients, n_samples_train, batch_size, shuffle)
        test=iid_split(dataset_loaded_test, n_clients, n_samples_test, batch_size, shuffle)
    elif type=="non_iid":
        train=non_iid_split(dataset_loaded_train, n_clients, n_samples_train, batch_size, shuffle)
        test=non_iid_split(dataset_loaded_test, n_clients, n_samples_test, batch_size, shuffle)
    elif type=="random":
           #non_equitable_random_split(dataset, nb_nodes, min_samples_per_node, max_samples_per_node, batch_size, shuffle, train_ratio=0.8)
           train, test, index, node_labels=non_equitable_random_split(global_dataset, n_clients,int(n_samples*0.3), int(n_samples), batch_size, shuffle)
    elif type=="dirichlet":
          #complete le code
            
    return train, test, index, node_labels
    """

def get_mnist_distribution(type="iid", n_samples_train=200, n_samples_test=100, n_clients=3, 
                           batch_size=25, shuffle=True, alpha=0.5):
    """
    Répartit le dataset MNIST entre plusieurs clients selon différentes stratégies (IID, Non-IID, Random, Dirichlet).

    Args:
        type (str): Type de répartition des données ('iid', 'non_iid', 'random', 'dirichlet').
        n_samples_train (int): Nombre d'échantillons d'entraînement par client.
        n_samples_test (int): Nombre d'échantillons de test par client.
        n_clients (int): Nombre de clients (noeuds).
        batch_size (int): Taille des batchs pour les DataLoaders.
        shuffle (bool): Mélanger les données avant de les répartir.
        alpha (float): Paramètre de la distribution de Dirichlet (plus petit = plus déséquilibré).

    Returns:
        train_loaders (list): Liste des DataLoaders d'entraînement.
        test_loaders (list): Liste des DataLoaders de test.
        index (dict): Indices des données utilisées pour chaque client.
        node_labels (dict): Labels distribués pour chaque client.
    """

    # Charger les datasets MNIST
    dataset_loaded_train = get_mnist("./data", train=True)
    dataset_loaded_test = get_mnist("./data", train=False)

    # Fusion des datasets train et test pour le mode random et dirichlet
    global_dataset = ConcatDataset([dataset_loaded_train, dataset_loaded_test])
    n_samples = int(len(global_dataset) * 0.15)  # 15% du dataset total utilisé

    train, test, index, node_labels = None, None, None, None

    if type == "iid":
        train = iid_split(dataset_loaded_train, n_clients, n_samples_train, batch_size, shuffle)
        test = iid_split(dataset_loaded_test, n_clients, n_samples_test, batch_size, shuffle)

    elif type == "non_iid":
        train = non_iid_split(dataset_loaded_train, n_clients, n_samples_train, batch_size, shuffle)
        test = non_iid_split(dataset_loaded_test, n_clients, n_samples_test, batch_size, shuffle)

    elif type == "random":
        train, test, index, node_labels = non_equitable_random_split(global_dataset, n_clients, 
                                                                     int(n_samples * 0.3), int(n_samples), 
                                                                     batch_size, shuffle)

    elif type == "dirichlet":
        train, test, class_distribution = dirichlet_non_iid_split(global_dataset, n_clients, alpha, batch_size, shuffle)
        index = None  # Pas d'index spécifique dans ce cas
        node_labels = None  # Pas de labels spécifiques

    return train, test, index, node_labels