import torch
import copy
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.optim as optim
from options import args_parser
from Entities.fog import Fog
from models.modelcom import CNN
from Algorithms.mnist import get_mnist_distribution
from Algorithms.usps import get_usps_distribution
from Algorithms.svhn import get_svhn_distribution
from Algorithms.mnistm import get_mnistm_distribution
from Algorithms.emnist import get_emnist_distribution
import torchvision.transforms as transforms
from data.data_saving import load_train_test_dataloaders, save__statistics, save_train_test_dataloaders, plot_distribution
from data.createSets import non_iid_split, iid_split
from Algorithms.mnistm import MNISTM
from Algorithms.mnist import get_mnist
from Algorithms.usps import get_usps
from Algorithms.svhn import get_svhn
from Algorithms.emnist import get_emnist
from Aggregation.FedGA import FedGA
import concurrent.futures
args = args_parser()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Starting Fed_MAML framework...")


# Liste pour stocker les Fogs
fogs = []

#args.num_users = 3
# Fog 1 : MNIST
args.dataset = 'mnist'  #
folder= "../data/mnist_data_dirichlet"
#train, test = load_train_test_dataloaders(folder)
trainset = get_mnist(path='../data', train=True)
train_fog = iid_split(trainset,args.num_users, 60000 //args.num_users , 64, True)
if (not os.path.exists(folder)):
    os.makedirs(folder)
    train_loaders1, test_loaders1, _, _ = get_mnist_distribution(
    type="dirichlet", n_samples_train=200, n_samples_test=100, 
    n_clients=10, batch_size=32, shuffle=True, alpha=0.5
     )
    save__statistics(train_loaders1, test_loaders1, n_clients=10, file_path=folder+"/mnist_summary_2.xlsx")
    save_train_test_dataloaders(train_loaders1, test_loaders1, folder)
    plot_distribution(folder+"/mnist_summary_2.xlsx", save_path=folder+"/mnist_summary_2.png", dataset_name="MNIST")

else:
    print("Data already exists")
    train_loaders1, test_loaders1 = load_train_test_dataloaders(folder)

model_mnist = CNN(1).to(device)
optimizer_mnist = optim.SGD(model_mnist.parameters(), lr=args.lr, momentum=args.momentum)
loss_fn = torch.nn.CrossEntropyLoss()
fog_1 = Fog(1, model_mnist, [(train_loaders1[i], test_loaders1[i]) for i in range(args.num_users)], optimizer_mnist, loss_fn, copy.deepcopy(args),'Fed_MAML', train_fog[0])
fogs.append(fog_1)


# Fog 2 : USPS
args.dataset = 'usps'
usps_directory= "../data/usps_data_dirichlet"
trainset = get_usps(path='../data', train=True)

#train, test = load_train_test_dataloaders(folder)
train_fog = iid_split(trainset,args.num_users, 7291 //args.num_users , 64, True)
if (not os.path.exists(usps_directory)):
    os.makedirs(usps_directory)
    train_loaders2, test_loaders2, _, _ = get_usps_distribution(
    type="dirichlet", n_samples_train=200, n_samples_test=100, 
    n_clients=10, batch_size=32, shuffle=True, alpha=0.5
     )
    save__statistics(train_loaders2, test_loaders2, n_clients=10, file_path=usps_directory+"/usps_summary_2.xlsx")
    save_train_test_dataloaders(train_loaders2, test_loaders2, usps_directory)
    plot_distribution(usps_directory+"/usps_summary_2.xlsx", save_path=usps_directory+"/usps_summary_2.png", dataset_name="USPS")
else :
    print("Data already exists")
    train_loaders2, test_loaders2 = load_train_test_dataloaders(usps_directory)
    


model_usps = CNN(1).to(device)
optimizer_usps = optim.SGD(model_usps.parameters(), lr=args.lr, momentum=args.momentum)
fog_2 = Fog(2, model_usps, [(train_loaders2[i], test_loaders2[i]) for i in range(args.num_users)], optimizer_usps, loss_fn, copy.deepcopy(args),'Fed_MAML', train_fog[0])
fogs.append(fog_2)


# Fog 3 : SVHN
args.dataset = 'svhn'
svhn_directory = '../data/svhn_data_dirichlet'
trainset = get_svhn(path='../data', split='train')

#train, test = load_train_test_dataloaders(folder)
train_fog = iid_split(trainset,args.num_users, 73257 //args.num_users , 128, True)
if (not os.path.exists(svhn_directory)):
    os.makedirs(svhn_directory)
    train_loaders3, test_loaders3, _, _ = get_svhn_distribution(
    type="dirichlet", n_samples_train=200, n_samples_test=100, 
    n_clients=10, batch_size=32, shuffle=True, alpha=0.5
     )
    save__statistics(train_loaders3, test_loaders3, n_clients=10, file_path=svhn_directory+"/svhn_summary_2.xlsx")
    save_train_test_dataloaders(train_loaders3, test_loaders3, svhn_directory)
    plot_distribution(svhn_directory+"/svhn_summary_2.xlsx", save_path=svhn_directory+"/svhn_summary_2.png", dataset_name="SVHN")
else :
    print("Data already exists")
    train_loaders3, test_loaders3 = load_train_test_dataloaders(svhn_directory)

model_svhn = CNN(3, 'svhn').to(device)
optimizer_svhn = optim.SGD(model_svhn.parameters(), lr=args.lr, momentum=args.momentum)
fog_3 = Fog(3, model_svhn, [(train_loaders3[i], test_loaders3[i]) for i in range(args.num_users)], optimizer_svhn, loss_fn, copy.deepcopy(args),'Fed_MAML', train_fog[0])
fogs.append(fog_3)


# Fog 4 : MNISTM
args.dataset = 'mnistm'
#folder= "../data/mnistm_data"
mnistm_directory = "../data/mnistm_data_dirichlet"
trainset=MNISTM(root='../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))

#train, test = load_train_test_dataloaders(folder)
train_fog = iid_split(trainset,args.num_users, 59000 //args.num_users , 128, True)
if (not os.path.exists(mnistm_directory)):
    os.makedirs(mnistm_directory)
    train_loaders4, test_loaders4, _, _ = get_mnistm_distribution(
    type="dirichlet", n_samples_train=200, n_samples_test=100, 
    n_clients=10, batch_size=32, shuffle=True, alpha=0.5
     )
    save__statistics(train_loaders4, test_loaders4, n_clients=10, file_path=mnistm_directory+"/mnistm_summary_2.xlsx")
    save_train_test_dataloaders(train_loaders4, test_loaders4, mnistm_directory)
    plot_distribution(mnistm_directory+"/mnistm_summary_2.xlsx", save_path=mnistm_directory+"/mnistm_summary_2.png", dataset_name="MNISTM")
else :
    print("Data already exists")
    train_loaders4, test_loaders4 = load_train_test_dataloaders(mnistm_directory)
model_mnistm = CNN(3).to(device)
optimizer_mnistm = optim.SGD(model_mnistm.parameters(), lr=args.lr, momentum=args.momentum)
fog_4 = Fog(4, model_mnistm, [(train_loaders4[i], test_loaders4[i]) for i in range(args.num_users)], optimizer_mnistm, loss_fn, copy.deepcopy(args),'Fed_MAML', train_fog[0])
fogs.append(fog_4)


# Fog 5 : EMNIST
args.dataset = 'emnist'
folder= "../data/emnist_data"
emnist_directory= "../data/emnist_data_dirichlet"
trainset = get_emnist(path='../data', split='digits', train=True)

#train, test = load_train_test_dataloaders(folder)
train_fog = iid_split(trainset,args.num_users, 60000 //args.num_users , 128, True)
if (not os.path.exists(emnist_directory)):
    os.makedirs(emnist_directory)
    train_loaders5, test_loaders5, _, _ = get_emnist_distribution(
    type="dirichlet", n_samples_train=200, n_samples_test=100, 
    n_clients=10, batch_size=32, shuffle=True, alpha=0.5
     )
    save__statistics(train_loaders5, test_loaders5, n_clients=10, file_path=emnist_directory+"/emnist_summary_2.xlsx")
    save_train_test_dataloaders(train_loaders5, test_loaders5, emnist_directory)
    plot_distribution(emnist_directory+"/emnist_summary_2.xlsx", save_path=emnist_directory+"/emnist_summary_2.png", dataset_name="EMNIST")
else :
    print("Data already exists")
    train_loaders5, test_loaders5 = load_train_test_dataloaders(emnist_directory)
model_emnist= CNN(1).to(device)
optimizer_emnist = optim.SGD(model_emnist.parameters(), lr=args.lr, momentum=args.momentum)
fog_5 = Fog(5, model_emnist, [(train_loaders5[i], test_loaders5[i]) for i in range(args.num_users)], optimizer_emnist, loss_fn, copy.deepcopy(args),'Fed_MAML', train_fog[0])
fogs.append(fog_5)





def federated_outer_update(global_model, client_grads, beta=0.001):
    """
    Effectue l'outer update en agrégeant les gradients des clients uniquement pour la classification layer.

    Arguments :
    - global_model : Modèle global du serveur.
    - client_grads : Liste contenant les gradients des couches classification des clients.
    - beta : Learning rate pour la mise à jour globale.

    Retourne :
    - Le modèle global mis à jour.
    """

    # Copier les poids globaux pour éviter la modification directe
    global_weights = copy.deepcopy(global_model.state_dict())
    print("Starting outer update")

    # Calculer la moyenne des gradients sur tous les clients
    avg_grads = {}
    for key in client_grads[0].keys():  # Parcours des clés des gradients
        avg_grads[key] = sum(client[key] for client in client_grads) / len(client_grads)

    # Appliquer les gradients moyens pour mettre à jour les poids de la classification layer
    with torch.no_grad():
        for name, param in global_model.classification.named_parameters():
            if name in avg_grads:  # Assurer que la clé existe dans les gradients
                global_weights[f"classification.{name}"] -= beta * avg_grads[name]  # Descente de gradient manuelle

    # Charger les nouveaux poids dans le modèle global
    global_model.load_state_dict(global_weights)

    return global_model  # Retourne le modèle global mis à jour



global_model = CNN(3).to(device)
global_classification_weights=None

def federated_outer_update(global_model, client_grads, beta=0.001):
    """
    Effectue l'outer update en agrégeant les gradients des clients uniquement pour la classification layer.

    Arguments :
    - global_model : Modèle global du serveur.
    - client_grads : Liste contenant les gradients des couches classification des clients.
    - beta : Learning rate pour la mise à jour globale.

    Retourne :
    - Le modèle global mis à jour.
    """

    # Copier les poids globaux pour éviter la modification directe
    global_weights = copy.deepcopy(global_model.state_dict())
    print("Starting outer update")

    # Calculer la moyenne des gradients sur tous les clients
    avg_grads = {}
    for key in client_grads[0].keys():  # Parcours des clés des gradients
        avg_grads[key] = sum(client[key] for client in client_grads) / len(client_grads)

    # Appliquer les gradients moyens pour mettre à jour les poids de la classification layer
    with torch.no_grad():
        for name, param in global_model.classification.named_parameters():
            if name in avg_grads:  # Assurer que la clé existe dans les gradients
                global_weights[f"classification.{name}"] -= beta * avg_grads[name]  # Descente de gradient manuelle

    # Charger les nouveaux poids dans le modèle global
    global_model.load_state_dict(global_weights)

    return global_model  # Retourne le modèle global mis à jour
for Round in range(args.epochs+1):
  all_classification_gradients=[]
  print(f"Startig round {Round+1} with Fed_MAML Framework")
  for fog in fogs:
      global_classification_weights=None #je l'ai ajouté pour scalabilité
      lists=fog.MAML_edges(global_classification_weights, Round)
      for item in lists:
          all_classification_gradients.append(item)
  global_model=copy.deepcopy(federated_outer_update(global_model,all_classification_gradients, args.meta_lr))
  global_classification_weights=copy.deepcopy(global_model.classification.state_dict())



      
"""  
global_model = CNN(3).to(device)
global_classification_weights=None
for Round in range(args.epochs):
    all_classification_gradients = []
    print(f"Starting round {Round+1} with FedGA_Meta Framework")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(fogs)) as executor:
        results = list(executor.map(lambda fog: fog.fedga_meta_inner(global_classification_weights, Round), fogs))

    all_classification_gradients.extend(results)

    # Mise à jour du modèle global
    global_model = copy.deepcopy(federated_outer_update(global_model, all_classification_gradients))
    global_classification_weights = copy.deepcopy(global_model.classification.state_dict())

# Exécution finale de `fedga_meta_inner()` pour chaque `Fog`
with concurrent.futures.ThreadPoolExecutor(max_workers=len(fogs)) as executor:
    executor.map(lambda fog: fog.fedga_meta_inner(global_classification_weights, args.epochs), fogs)"""  