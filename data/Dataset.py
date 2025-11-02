import datetime
import torch
import random
from Enumeration import DataType
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
import os
import pickle
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt

class Dataset:
    def __init__(
        self, 
        id: str, 
        name: str, 
        description: str, 
        data,
        data_type: DataType, 
    ):
        self.id = id
        self.name = name
        self.description = description
       ## self.feature_columns = feature_columns  // no need in Image processing because each column is a RGB pixel
       ## self.target_column = target_column
        self.data_type = data_type
        self.size = len(data)
        self.data=data
        self.train_data=data[0]
        self.test_data=data[1]
        self.creation_date = datetime.datetime.now()
        """
        self.samples = self.extract_samples_labels()[0]
        self.labels = self.extract_samples_labels()[1]
        self.unique_labels= self.extract_samples_labels()[2]
       """
        self.validation_data=None
        #self.create_validation_set(100)

        


      
    

    def create_validation_set(self, validation_size=100):
      self.validation_data = self.test_data
      """
      Creates a validation set by extracting `validation_size` samples from the test set.

    Args:
        validation_size (int): Number of samples for validation.

    Returns:
        self.validation_data (DataLoader): DataLoader for the validation set.
    
    

    # Récupérer le dataset depuis le DataLoader
      test_dataset = self.test_data.dataset  

    # Si le dataset est un Subset, récupérer le dataset d'origine et les indices
      if isinstance(test_dataset, Subset):
        original_dataset = test_dataset.dataset  # Dataset original
        test_indices = test_dataset.indices  # Indices du subset
        
        # Extraire les images et labels en utilisant les indices
        test_images = original_dataset.data[test_indices]
        test_labels = original_dataset.targets[test_indices]
    
      # Si le dataset est un TensorDataset
      elif isinstance(test_dataset, TensorDataset):
        test_images, test_labels = test_dataset.tensors

      # Si le dataset est un Dataset classique comme MNIST
      elif hasattr(test_dataset, "data") and hasattr(test_dataset, "targets"):
        test_images = test_dataset.data
        test_labels = test_dataset.targets

      else:
        raise TypeError("❌ `self.test_data.dataset` is not a supported dataset type (Subset, TensorDataset, or Standard Dataset).")

      # Vérifier si le test set a assez d'exemples
      if len(test_images) < validation_size:
        self.validation_data = self.test_data
      else:
        # Sélectionner aléatoirement des indices pour le set de validation
        validation_indices = random.sample(range(len(test_images)), validation_size)

        validation_images = test_images[validation_indices]
        validation_labels = test_labels[validation_indices]

        # Retirer les indices sélectionnés du set de test
        remaining_indices = list(set(range(len(test_images))) - set(validation_indices))
        test_images = test_images[remaining_indices]
        test_labels = test_labels[remaining_indices]

        # Création des DataLoaders
        self.validation_data = DataLoader(TensorDataset(validation_images, validation_labels), batch_size=25, shuffle=True)
        self.test_data = DataLoader(TensorDataset(test_images, test_labels), batch_size=25, shuffle=True)
"""
      return self.validation_data


    """
    def print_data_distribution(self):
        train_samples = sum(len(batch[0]) for batch in self.train_data)
        test_samples = sum(len(batch[0]) for batch in self.test_data)
        print(f"Number of images - Training: {train_samples}, Testing: {test_samples}")

    def extract_samples_labels(self):
     samples, labels, unique_labels = [], [], []
     # Itérer directement sur le DataLoader
     for sample, label in self.train_data:
        samples.append(sample)
        labels.append(label)
     unique_labels = set(labels)  # Convertir la liste en un ensemble pour éliminer les doublons
     return samples, labels, list(unique_labels)

    

    # Fonction pour augmenter les données
    def augment_dataset(self,dataset, factor):
       augmented_data = []
       augmented_targets = []
    
    # Définit les transformations aléatoires pour l'augmentation des données
       transform_augment = transforms.Compose([
        transforms.RandomRotation(10),  # Rotation aléatoire
        transforms.RandomHorizontalFlip(),  # Flip horizontal aléatoire
        transforms.ToTensor()
       ])
    
       for img, target in dataset:
          augmented_data.append((img, target))
          for _ in range(factor - 1):
            img_aug = transform_augment(img)
            augmented_data.append((img_aug, target))
            augmented_targets.append(target)
    
       augmented_dataset = torch.utils.data.TensorDataset(torch.stack([img for img, _ in augmented_data]), torch.tensor([target for _, target in augmented_data]))
       return augmented_dataset




    def augment_dataset_with_variation(self,dataset, min_factor, max_factor):
      augmented_data = []
      augmented_targets = []

      # Définit les transformations aléatoires pour l'augmentation des données
      transform_augment = transforms.Compose([
        transforms.RandomRotation(10),  # Rotation aléatoire
        transforms.RandomHorizontalFlip(),  # Flip horizontal aléatoire
        transforms.ToTensor()
      ])
      # Extrait les images et les labels du dataset
      images, targets = dataset.tensors
      class_counts = torch.bincount(targets)
      #class_counts = torch.bincount(dataset.targets)
      for i in range(len(class_counts)):
        class_indices = torch.where(targets == i)[0]
        n_samples = random.randint(min_factor, max_factor) * len(class_indices)
        for idx in class_indices:
            img, target = dataset[idx]
            augmented_data.append((img, target))
            for _ in range(n_samples - 1):
                img_pil = self.tensor_to_pil(img)
                img_aug = transform_augment(img_pil)
                augmented_data.append((img_aug, target))
                augmented_targets.append(target)

      augmented_dataset = TensorDataset(torch.stack([img for img, _ in augmented_data]), torch.tensor([target for _, target in augmented_data]))
      return augmented_dataset
    

    def tensor_to_pil(self,tensor):
      return transforms.ToPILImage()(tensor)
    

    def augment_dataset_with_variation_percent(self, dataset, percentage=10):

        augmented_data = []
        augmented_targets = []

        transform_augment = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        if isinstance(dataset, ConcatDataset):
            images_list, targets_list = [], []
            for sub_dataset in dataset.datasets:
                images_list.append(sub_dataset.tensors[0])
                targets_list.append(sub_dataset.tensors[1])
            images = torch.cat(images_list)
            targets = torch.cat(targets_list)
        else:
            images, targets = dataset.tensors
        
        n_samples_to_add = int(len(images) * percentage / 100)

        class_counts = torch.bincount(targets)
        
        for i in range(len(class_counts)):
            class_indices = torch.where(targets == i)[0]
            if len(class_indices) == 0:
                continue
            n_samples = max(1, random.randint(1, 2) * len(class_indices) // 10)  # Variation in the number of samples
            for _ in range(n_samples):
                idx = random.choice(class_indices)
                img, target = images[idx], targets[idx]
                img_pil = transforms.ToPILImage()(img)
                img_aug = transform_augment(img_pil)
                augmented_data.append(img_aug)
                augmented_targets.append(target)

        if augmented_data:
            augmented_data = torch.stack(augmented_data)
            augmented_targets = torch.tensor(augmented_targets)
            new_dataset = ConcatDataset([dataset, TensorDataset(augmented_data, augmented_targets)])
        else:
            new_dataset = dataset

        return new_dataset

    def execute_transformations(self):
          
           #factor = random.choice([1,2, 3])
           #augmented_train_dataset = self.augment_dataset(self.train_data.dataset, factor)
           #self.train_data = DataLoader(augmented_train_dataset, batch_size=25, shuffle=True)
           original_dataset=self.train_data.dataset
           print(len(original_dataset))
           augmented_train_dataset = self.augment_dataset_with_variation(original_dataset, 1, 1)
           self.train_data = DataLoader(augmented_train_dataset, batch_size=25, shuffle=True)
           #factor = random.choice([1,2, 3])
           #augmented_test_dataset = self.augment_dataset(self.test_data.dataset, factor)
           #self.test_data = DataLoader(augmented_test_dataset, batch_size=25, shuffle=True)
           original_dataset=self.test_data.dataset
           #print(len(original_dataset))
           augmented_test_dataset = self.augment_dataset_with_variation(original_dataset, 1, 1)
           self.test_data = DataLoader(augmented_test_dataset, batch_size=25, shuffle=True)


           print(f"Data augmentation done successfully --> training: { len(self.train_data),self.count_samples_per_class(augmented_train_dataset)}, ---- testing: {len(self.test_data),self.count_samples_per_class(augmented_test_dataset)}  " )
         

        

    def show_data_distribution(self):
        train_counter = self.count_samples_per_class(self.train_data)
        test_counter = self.count_samples_per_class(self.test_data)
        print(f"{self.id, len(self.train_data), len(self.test_data)}: Training samples per class: {train_counter}")
        print(f"{self.id, len(self.train_data), len(self.test_data)}: Test samples per class: {test_counter}")

    def execute_augmentation_with_variation(self):
          original_dataset=self.train_data.dataset
          augmented_train_dataset = self.augment_dataset_with_variation_percent(original_dataset, 10)
          self.train_data = DataLoader(augmented_train_dataset, batch_size=25, shuffle=True)

          original_dataset=self.test_data.dataset
          augmented_test_dataset = self.augment_dataset_with_variation_percent(original_dataset, 10)
          self.test_data = DataLoader(augmented_test_dataset, batch_size=25, shuffle=True)
          print(f"Data augmentation done successfully --> training: {len(augmented_train_dataset)}, ---- testing: {len( augmented_test_dataset)}")

          

    def count_samples_per_class(self, loader):
        counter = Counter()
        for _, labels in loader:
            labels = labels.numpy()
            if labels.ndim == 0:  # Ensure labels is iterable
                labels = labels.reshape(1)
            counter.update(labels)
        return counter
    

def save_train_test_dataloaders(train_loaders, test_loaders, directory):

    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Enregistrer les DataLoaders d'entraînement
    for i, loader in enumerate(train_loaders):
        with open(os.path.join(directory, f"train_dataloader_{i}.pkl"), 'wb') as f:
            pickle.dump(loader.dataset, f)
    
    # Enregistrer les DataLoaders de test
    for i, loader in enumerate(test_loaders):
        with open(os.path.join(directory, f"test_dataloader_{i}.pkl"), 'wb') as f:
            pickle.dump(loader.dataset, f)

### Fonction pour Charger les DataLoaders

def load_train_test_dataloaders(directory, batch_size, shuffle):

    train_loaders = []
    test_loaders = []
    
    # Charger les DataLoaders d'entraînement
    train_files = [f for f in os.listdir(directory) if f.startswith('train')]
    for file in train_files:
        with open(os.path.join(directory, file), 'rb') as f:
            dataset = pickle.load(f)
            train_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))
    
    # Charger les DataLoaders de test
    test_files = [f for f in os.listdir(directory) if f.startswith('test')]
    for file in test_files:
        with open(os.path.join(directory, file), 'rb') as f:
            dataset = pickle.load(f)
            test_loaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))
    
    return train_loaders, test_loaders
 



def verify_directory_contents(directory, nb_edge):

       if not os.path.exists(directory):
        print(f"Le répertoire {directory} n'existe pas.")
        return False
    
    # Compter le nombre de fichiers dans le répertoire
       file_count = len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])
    
    # Vérifier si le nombre de fichiers est égal à nb_edge
       if file_count == nb_edge:
        print(f"Le répertoire {directory} existe et contient {file_count} fichiers, comme attendu.")
        return True
       else:
        print(f"Le répertoire {directory} existe mais contient {file_count} fichiers au lieu de {nb_edge}.")
        return False



import pandas as pd

def summarize_data_by_node(train_loaders, test_loaders, node_labels, file):
    # Création des DataFrame pour accumuler les résultats
    train_summary = pd.DataFrame()
    test_summary = pd.DataFrame()

    # Itérer sur chaque noeud pour le train
    for i, (loader, labels) in enumerate(zip(train_loaders, node_labels)):
        # Compter les occurrences de chaque label
        #label_counts = pd.Series(labels.numpy()).value_counts().sort_index()
        label_counts = pd.Series(labels.numpy().flatten()).value_counts().sort_index()

        train_summary[f'Node {i+1}'] = label_counts

    # Itérer sur chaque noeud pour le test
    for i, (loader, labels) in enumerate(zip(test_loaders, node_labels)):
        # Compter les occurrences de chaque label
        #label_counts = pd.Series(labels.numpy()).value_counts().sort_index()
        label_counts = pd.Series(labels.numpy().flatten()).value_counts().sort_index()
        test_summary[f'Node {i+1}'] = label_counts

    # Exporter les résultats en Excel
    with pd.ExcelWriter(file) as writer:
        train_summary.to_excel(writer, sheet_name='Train Summary')
        test_summary.to_excel(writer, sheet_name='Test Summary')

# Appel de la fonction
# Assurez-vous que train_loaders, test_loaders, et node_labels sont définis comme il faut
# summarize_data_by_node(train_loaders, test_loaders, node_labels)

"""