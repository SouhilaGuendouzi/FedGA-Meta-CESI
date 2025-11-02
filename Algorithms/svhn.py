
from torchvision.datasets import SVHN
from torchvision import transforms
import sys 
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try: 
  from data.createSets import iid_split, non_iid_split, non_equitable_random_split, dirichlet_non_iid_split
except: 
  from createSets import iid_split, non_iid_split, non_equitable_random_split, dirichlet_non_iid_split
from torch.utils.data import ConcatDataset

def normalize(data_tensor):
    '''Re-scale image values to [-1, 1].'''
    return (data_tensor / 255.) * 2. - 1. 


def get_svhn(path='./data', split='train'):
    """
    Charge le dataset SVHN, redimensionne les images à 28x28,
    et applique une normalisation dans l'intervalle [-1, 1].

    Args:
        path (str): Chemin vers le répertoire de téléchargement.
        split (str): 'train' ou 'test' pour le split du dataset.

    Returns:
        torchvision.datasets.SVHN: Dataset transformé.
    """
    # Liste des transformations
    from torchvision.transforms import Compose, ToTensor, Normalize

# Define transformations
    data_transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))  # Standard normalization
])

    svhn_dataset = SVHN(
    root=path, split=split,
    transform=data_transform,  # Use a proper transform object
    download=True
)
    
    return svhn_dataset



def get_svhn_distribution(type="iid", n_samples_train=200, n_samples_test=100, n_clients=3, batch_size=25, shuffle=True, alpha=0.5):
    dataset_loaded_train = get_svhn("./data", split='train')
    dataset_loaded_test = get_svhn("./data", split='test')
     #merge the two datasets and put them into one new dataset variable
    global_dataset = ConcatDataset([dataset_loaded_train, dataset_loaded_test])
    n_samples=len(global_dataset)*0.15
    train, test, index, node_labels = None, None, None, None
    alpha = 0.5  # Dirichlet distribution parameter

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
        train, test, class_distribution = dirichlet_non_iid_split(global_dataset, n_clients, alpha, batch_size, shuffle)
        index = None  # Pas d'index spécifique dans ce cas
        node_labels = None  # Pas de labels spécifiques
    return train, test, index, node_labels