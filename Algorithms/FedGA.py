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
from mnist import get_mnist
from data.data_saving import load_train_test_dataloaders, save__statistics, save_train_test_dataloaders, plot_distribution
from data.createSets import non_iid_split, iid_split
from Algorithms.mnistm import MNISTM
from Aggregation.FedGA import FedGA
args = args_parser()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print("Starting FedGA framework...")
# Liste pour stocker les Fogs
fogs = []

### Préparation des données et des modèles pour chaque Fog ###
# Fog 1 : MNIST
args.dataset = 'mnist'  #
folder= "../data/mnist_data_dirichlet"
#trainset = get_mnist(path='./data', train=True)
#testset = get_mnist(path='./data', train=False)
#train = non_iid_split(trainset,args.num_users, 60000 //args.num_users , 32, True)
#test = non_iid_split(testset,args.num_users, 10000//args.num_users, 32, True)
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
#print the size of the first dataset in the dataset mnist

#model_mnist = Model_mnist().to(device)
model_mnist = CNN(1).to(device)
optimizer_mnist = optim.SGD(model_mnist.parameters(), lr=args.lr, momentum=args.momentum)
loss_fn = torch.nn.CrossEntropyLoss()
fog_1 = Fog(1, model_mnist, [(train_loaders1[i], test_loaders1[i]) for i in range(args.num_users)], optimizer_mnist, loss_fn, copy.deepcopy(args),'FedGA')
fogs.append(fog_1)





# Fog 2 : USPS
args.dataset = 'usps'
usps_directory= "../data/usps_data_dirichlet"
#trainset = get_usps(path='./data', train=True)
#testset = get_usps(path='./data', train=False)
#train = non_iid_split(trainset,args.num_users, 7291 //args.num_users, 32, True)
#test = non_iid_split(testset,args.num_users, 2007 //args.num_users , 32, True)
#print the size of one image in the dataset usps
#image, label = trainset[0]
#print('usps',image.size())
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
    


#model_usps = Model_usps().to(device)
model_usps = CNN(1).to(device)
optimizer_usps = optim.SGD(model_usps.parameters(), lr=args.lr, momentum=args.momentum)
fog_2 = Fog(2, model_usps, [(train_loaders2[i], test_loaders2[i]) for i in range(args.num_users)], optimizer_usps, loss_fn, copy.deepcopy(args),'FedGA')
fogs.append(fog_2)


# Fog 3 : SVHN
args.dataset = 'svhn'
svhn_directory = '../data/svhn_data_dirichlet'
#trainset = get_svhn(path='./data', split='train')
#testset = get_svhn(path='./data', split='test')
#train = non_iid_split(trainset,args.num_users, 73257 //args.num_users, 32, True)
#test = non_iid_split(testset,args.num_users, 26032 //args.num_users , 32, True)
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
fog_3 = Fog(3, model_svhn, [(train_loaders3[i], test_loaders3[i]) for i in range(args.num_users)], optimizer_svhn, loss_fn, copy.deepcopy(args),'FedGA')
fogs.append(fog_3)


# Fog 4 : MNISTM
args.dataset = 'mnistm'

mnistm_directory = "../data/mnistm_data_dirichlet"
#trainset=MNISTM(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
#testset=MNISTM(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
#train = non_iid_split(trainset,args.num_users, 59000 //args.num_users, 32, True)
#test = non_iid_split(testset,args.num_users, 9000 //args.num_users, 32, True)
#image, label = trainset[0]
#print('mnistm',image.size())
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
#model_mnistm = Model_mnistm().to(device)
model_mnistm = CNN(3).to(device)
optimizer_mnistm = optim.SGD(model_mnistm.parameters(), lr=args.lr, momentum=args.momentum)
fog_4 = Fog(4, model_mnistm, [(train_loaders4[i], test_loaders4[i]) for i in range(args.num_users)], optimizer_mnistm, loss_fn, copy.deepcopy(args),'FedGA')
fogs.append(fog_4)


# Fog 5 : EMNIST
args.dataset = 'emnist'
emnist_directory= "../data/emnist_data_dirichlet"
#trainset = get_emnist(path='./data', split='digits', train=True)
#testset = get_emnist(path='./data', split='digits', train=False)
#train = non_iid_split(trainset,args.num_users, 60000 //args.num_users, 32, True)
#test = non_iid_split(testset,args.num_users, 10000 //args.num_users, 32, True)
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

#model_emnist = Model_emnist().to(device)
model_emnist = CNN(1).to(device)
optimizer_emnist = optim.SGD(model_emnist.parameters(), lr=args.lr, momentum=args.momentum)
fog_5 = Fog(5, model_emnist, [(train_loaders5[i], test_loaders5[i]) for i in range(args.num_users)], optimizer_emnist, loss_fn, copy.deepcopy(args),'FedGA')
fogs.append(fog_5)




# preparethe global benchmark dataset 
trainset=get_mnist(path='../data/data', train=True)
train_cloud = iid_split(trainset,args.num_users, 60000 //args.num_users , 64, True)

#Federated Learning Global avec FedGA sur le cloud 

all_classification_layers=[]
global_classification_weights=None
for Round in range(args.epochs+1):
  all_classification_layers=[]
  for fog in fogs:
    #if (fog.id==1) or (fog.id==2):
       fog_models=fog.get_classification_layers(global_classification_weights, Round)
       for model in fog_models:
          all_classification_layers.append(model)
  
  print(f"Round {Round} - Aggregating classification layers with FedGA...")

  global_classification_weights=FedGA(all_classification_layers, model_mnist, train_cloud[args.num_users-1],"mnitm",Round, classification=True)




### Entraînement des Fogs avec FedGA 
"""
for fog in fogs:
    print(f"Starting FedGA for Fog {fog.id}")
    fog.federated_training()
    print(f"Completed FedGA for Fog {fog.id}")
"""
print("FedGA training completed for all Fogs.")


