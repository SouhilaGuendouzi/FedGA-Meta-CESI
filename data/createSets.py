import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset

def non_iid_split(dataset, nb_nodes, n_samples_per_node, batch_size, shuffle, shuffle_digits=False):
    assert(nb_nodes>0 and nb_nodes<=10)

    digits=torch.arange(10) if shuffle_digits==False else torch.randperm(10, generator=torch.Generator().manual_seed(0))

    # split the digits in a fair way
    digits_split=list()
    i=0
    for n in range(nb_nodes, 0, -1):
        inc=int((10-i)/n)
        digits_split.append(digits[i:i+inc])
        i+=inc

    # load and shuffle nb_nodes*n_samples_per_node from the dataset
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=nb_nodes*n_samples_per_node,
                                        shuffle=shuffle)
    dataiter = iter(loader)
    images_train_mnist, labels_train_mnist = dataiter.next()

    data_splitted=list()
    for i in range(nb_nodes):
        idx=torch.stack([y_ == labels_train_mnist for y_ in digits_split[i]]).sum(0).bool() # get indices for the digits
        data_splitted.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images_train_mnist[idx], labels_train_mnist[idx]), batch_size=batch_size, shuffle=shuffle))

    return data_splitted



def iid_split(dataset, nb_nodes, n_samples_per_node, batch_size, shuffle):
    # load and shuffle n_samples_per_node from the dataset
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=n_samples_per_node,
                                        shuffle=shuffle)
    dataiter = iter(loader)
    
    data_splitted=list()
    for _ in range(nb_nodes):
        data_splitted.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(dataiter.next())), batch_size=batch_size, shuffle=shuffle))

    return data_splitted


def plot_samples(data, channel:int, title=None, plot_name="", n_examples =20):

    n_rows = int(n_examples / 5)
    plt.figure(figsize=(1* n_rows, 1*n_rows))
    if title: plt.suptitle(title)
    X, y= data
    for idx in range(n_examples):
        
        ax = plt.subplot(n_rows, 5, idx + 1)

        image = 255 - X[idx, channel].view((28,28))
        ax.imshow(image, cmap='gist_gray')
        ax.axis("off")

    if plot_name!="":plt.savefig(f"plots/"+plot_name+".png")

    plt.tight_layout()
   


"""
def non_equitable_random_split(dataset, nb_nodes, min_samples_per_node, max_samples_per_node, batch_size, shuffle, train_ratio=0.8):

    
    assert nb_nodes > 0
    print("Total dataset size:", len(dataset))
    
    # Load and shuffle the entire dataset
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=shuffle)
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    remaining_indices = list(range(len(labels)))
    random.shuffle(remaining_indices)
    
    data_splitted = [[] for _ in range(nb_nodes)]
    node_labels = []

    # Distribute indices ensuring constraints are met
    for node_index in range(nb_nodes):
        num_samples = random.randint(min_samples_per_node, max_samples_per_node)
        num_samples = min(num_samples, len(remaining_indices))
        
        selected_indices = remaining_indices[:num_samples]
        remaining_indices = remaining_indices[num_samples:]
        
        data_splitted[node_index].extend(selected_indices)
        node_labels.append(labels[selected_indices])

    # Create DataLoader for each node
    train_loaders, test_loaders, index_data = [], [], []

    for i, indices in enumerate(data_splitted):
        num_train_samples = int(len(indices) * train_ratio)
        train_indices = indices[:num_train_samples]
        test_indices = indices[num_train_samples:]
        
        train_images, train_labels = images[train_indices], labels[train_indices]
        test_images, test_labels = images[test_indices], labels[test_indices]
        
        train_loaders.append(DataLoader(TensorDataset(train_images, train_labels), batch_size=batch_size, shuffle=shuffle))
        test_loaders.append(DataLoader(TensorDataset(test_images, test_labels), batch_size=batch_size, shuffle=shuffle))
        
        index_data.append((train_indices, test_indices))

    return train_loaders, test_loaders, index_data, node_labels

"""
def non_equitable_random_split(dataset, nb_nodes, min_samples_per_node, max_samples_per_node, batch_size, shuffle, train_ratio=0.8):
    """
    Randomly splits the dataset into subsets for a specified number of nodes,
    ensuring each node has unique samples and a train-test split of 80%-20%.

    Args:
        dataset: The dataset to split (can be a ConcatDataset).
        nb_nodes: Number of nodes to split the data into.
        min_samples_per_node: Minimum number of samples per node.
        max_samples_per_node: Maximum number of samples per node.
        batch_size: Batch size for DataLoader.
        shuffle: Whether to shuffle the dataset before splitting.
        train_ratio: Proportion of the dataset to use for training for each node (default: 0.8).

    Returns:
        train_loaders: List of DataLoaders for training data for each node.
        test_loaders: List of DataLoaders for testing data for each node.
        index_data: Dictionary mapping node IDs to the dataset indices used for training and testing.
        node_labels: Dictionary mapping node IDs to unique labels in train and test datasets.
    """
    assert nb_nodes > 0, "Number of nodes must be greater than zero."
    assert min_samples_per_node > 0, "Minimum samples per node must be greater than zero."
    assert max_samples_per_node >= min_samples_per_node, "Max samples per node must be greater than or equal to min samples per node."

    train_loaders = []
    test_loaders = []
    index_data = {}
    node_labels = {}

    # Shuffle dataset indices
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle:
        random.shuffle(indices)

    for node in range(nb_nodes):
        # Randomly determine the number of samples for this node
        node_sample_count = random.randint(min_samples_per_node, max_samples_per_node)

        # Ensure we have enough indices remaining
        if len(indices) < node_sample_count:
            node_sample_count = len(indices)

        # Select unique samples for this node
        node_indices = indices[:node_sample_count]
        indices = indices[node_sample_count:]  # Remove used indices

        # Split into train and test for this node
        train_size = int(train_ratio * len(node_indices))
        train_indices = node_indices[:train_size]
        test_indices = node_indices[train_size:]

        # Ensure no empty DataLoader is created
        if len(train_indices) > 0:
            train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=shuffle)
            train_loaders.append(train_loader)
        else:
            train_loaders.append(None)

        if len(test_indices) > 0:
            test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=shuffle)
            test_loaders.append(test_loader)
        else:
            test_loaders.append(None)

        # Collect labels for train and test
        def get_labels(indices):
            labels = []
            for idx in indices:
                dataset_idx = 0
                while idx >= len(dataset.datasets[dataset_idx]):  # Identify sub-dataset
                    idx -= len(dataset.datasets[dataset_idx])
                    dataset_idx += 1
                    try:
                        # Try to get labels dynamically
                         sub_dataset = dataset.datasets[dataset_idx]
                         if hasattr(sub_dataset, 'train_labels'):  # For USPS
                            labels.append(sub_dataset.train_labels[idx])
                         elif hasattr(sub_dataset, 'targets'):  # For MNIST or similar datasets
                             labels.append(sub_dataset.targets[idx])
                         elif hasattr(sub_dataset, 'labels'):  # Other datasets with 'labels'
                            labels.append(sub_dataset.labels[idx])
                         elif hasattr(sub_dataset, 'test_labels'):  # For mnistm
                             labels.append(sub_dataset.test_labels[idx])
                         else:
                            print(dir(sub_dataset))
                            raise AttributeError(f"Dataset {type(sub_dataset)} does not have 'train_labels', 'targets', or 'labels'.")
                    except (KeyError, IndexError) as e:
                      print(f"Label index issue: {e}. Attempting to handle...")
            return set(labels)

        train_labels = get_labels(train_indices)
        test_labels = get_labels(test_indices)

        # Store DataLoaders, indices, and labels
        index_data[node] = {"train": train_indices, "test": test_indices}
        node_labels[node] = {"train": train_labels, "test": test_labels}

    return train_loaders, test_loaders, index_data, node_labels









def dirichlet_non_iid_split(dataset, nb_nodes, alpha, batch_size, shuffle=True, train_ratio=0.8):
    """
    Répartit un dataset entre plusieurs clients de manière non IID en utilisant la distribution de Dirichlet.

    Args:
        dataset: Le dataset PyTorch à diviser.
        nb_nodes: Nombre de clients (noeuds).
        alpha: Paramètre de concentration de la distribution de Dirichlet (plus petit = plus déséquilibré).
        batch_size: Taille des batchs pour les DataLoaders.
        shuffle: Indique s'il faut mélanger les données avant la division.
        train_ratio: Proportion des données de chaque client utilisée pour l'entraînement.

    Returns:
        train_loaders: Liste des DataLoaders pour l'entraînement.
        test_loaders: Liste des DataLoaders pour le test.
        class_distribution: Matrice des proportions de classes par client.
    """

    # Récupérer les labels du dataset
    # Récupérer les labels de chaque sous-dataset dans ConcatDataset
    """
    try:
      labels = np.concatenate([ds.targets if hasattr(ds, "targets") else ds.train_labels for ds in dataset.datasets])
    except AttributeError:
        labels = np.concatenate([ds.targets if hasattr(ds, "targets") else (ds.labels if hasattr(ds, "labels") else ds.train_labels) for ds in dataset.datasets])
    """
    labels_list = [ds.targets if hasattr(ds, "targets") else 
               (ds.labels if hasattr(ds, "labels") else 
                (ds.test_labels if hasattr(ds, "test_labels") else 
                 (ds.train_labels if hasattr(ds, "train_labels") else None))) 
               for ds in dataset.datasets]

    # Convert labels to numpy array
    labels = np.concatenate(labels_list)


    num_classes = len(np.unique(labels))

    # Générer des distributions de classe pour chaque client à partir d'une Dirichlet
    class_distribution = np.random.dirichlet(alpha=[alpha] * nb_nodes, size=num_classes)
    # Stocker les indices attribués à chaque client
    client_indices = {i: [] for i in range(nb_nodes)}

    for c in range(num_classes):
        # Indices des échantillons appartenant à la classe c
        class_indices = np.where(labels == c)[0]
        np.random.shuffle(class_indices)

        # Répartition des indices entre les clients selon Dirichlet
        split = (class_distribution[c] * len(class_indices)).astype(int)

        # Correction pour s'assurer que tous les indices sont attribués
        split[-1] = len(class_indices) - sum(split[:-1])

        start = 0
        for i in range(nb_nodes):
            client_indices[i].extend(class_indices[start : start + split[i]])
            start += split[i]

    # Création des DataLoaders train/test
    train_loaders, test_loaders = [], []

    for i in range(nb_nodes):
        # Mélanger les indices de chaque client
        np.random.shuffle(client_indices[i])

        # Découper en train/test
        split_idx = int(len(client_indices[i]) * train_ratio)
        train_indices, test_indices = client_indices[i][:split_idx], client_indices[i][split_idx:]

        train_loaders.append(DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=shuffle))
        test_loaders.append(DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=shuffle))

    return train_loaders, test_loaders, class_distribution
