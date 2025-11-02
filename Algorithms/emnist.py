
from torchvision.datasets import EMNIST
from torchvision import transforms
try: 
  from data.createSets import iid_split, non_iid_split, non_equitable_random_split, dirichlet_non_iid_split
except: 
  from createSets import iid_split, non_iid_split, non_equitable_random_split, dirichlet_non_iid_split
from torch.utils.data import ConcatDataset
#import params

def normalize(data_tensor):
    '''re-scale image values to [-1, 1]'''
    return (data_tensor / 255.) * 2. - 1. 


def get_emnist(path='.',split='digits',train=True):

      transform_list = [transforms.ToTensor()] #,transforms.Lambda(lambda x: normalize(x))
      
                                      

      emnist_dataset = EMNIST(root=path, split=split, train=train,
         download=True,transform=transforms.Compose(transform_list)) # transform=transforms.Compose(transform_list)

      
      return  emnist_dataset


def get_emnist_distribution(type="iid", n_samples_train=200, n_samples_test=100, n_clients=3, batch_size=25, shuffle=True, alpha=0.5):
    dataset_loaded_train = get_emnist("./data",split='digits',train=True)
    dataset_loaded_test = get_emnist("./data",split='digits',train=False)
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
    elif type == "dirichlet":
        train, test, class_distribution = dirichlet_non_iid_split(global_dataset, n_clients, alpha, batch_size, shuffle)
        index = None  # Pas d'index spécifique dans ce cas
        node_labels = None  # Pas de labels spécifiques
    return train, test, index, node_labels