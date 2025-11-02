
import pandas as pd
import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

try :
    from Algorithms.mnist import get_mnist_distribution
    from Algorithms.usps import get_usps_distribution
    from Algorithms.svhn import get_svhn_distribution
    from Algorithms.mnistm import get_mnistm_distribution
    from Algorithms.emnist import get_emnist_distribution
except:
    from Algorithms.mnist import get_mnist_distribution
    from Algorithms.usps import get_usps_distribution
    from Algorithms.svhn import get_svhn_distribution
    from Algorithms.mnistm import get_mnistm_distribution
    from Algorithms.emnist import get_emnist_distribution



def summarize_data_by_node(train_loaders, test_loaders, node_labels, file):
    # Create DataFrames for accumulating results
    train_summary = pd.DataFrame()
    test_summary = pd.DataFrame()

    # Iterate over each node for training data
    for i, loader in enumerate(train_loaders):
        if loader is not None:
            # Collect all labels from the DataLoader
            all_labels = []
            for _, labels in loader:
                all_labels.extend(labels.numpy())  # Convert to a list of labels

            # Count occurrences of each label
            label_counts = pd.Series(all_labels).value_counts().sort_index()

            # Flatten the index if labels are wrapped in lists/arrays
            label_counts.index = label_counts.index.map(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

            # Debugging: Print type and contents of label_counts
            #(f"Node {i+1} label counts type (train): {type(label_counts)}")
            #print(f"Node {i+1} label counts (train): {label_counts}")

            # Assign to train_summary
            train_summary[f'Node {i+1}'] = label_counts

    # Iterate over each node for testing data
    for i, loader in enumerate(test_loaders):
        if loader is not None:
            # Collect all labels from the DataLoader
            all_labels = []
            for _, labels in loader:
                all_labels.extend(labels.numpy())  # Convert to a list of labels

            # Count occurrences of each label
            label_counts = pd.Series(all_labels).value_counts().sort_index()

            # Flatten the index if labels are wrapped in lists/arrays
            label_counts.index = label_counts.index.map(lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x)

            # Debugging: Print type and contents of label_counts
            print(f"Node {i+1} label counts type (test): {type(label_counts)}")
            print(f"Node {i+1} label counts (test): {label_counts}")

            # Assign to test_summary
            test_summary[f'Node {i+1}'] = label_counts

    # Replace NaN with 0 to handle missing labels
    train_summary = train_summary.fillna(0).astype(int)
    test_summary = test_summary.fillna(0).astype(int)

    # Export the results to Excel
    with pd.ExcelWriter(file) as writer:
        train_summary.to_excel(writer, sheet_name='Train Summary')
        test_summary.to_excel(writer, sheet_name='Test Summary')

def save_train_test_dataloaders(train_loaders, test_loaders, directory):
    """
    Saves the train and test DataLoaders' datasets to the specified directory.

    Args:
        train_loaders: List of training DataLoaders.
        test_loaders: List of testing DataLoaders.
        directory: Directory where the DataLoaders will be saved.

    Returns:
        None
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the training DataLoaders' datasets
    for i, loader in enumerate(train_loaders):
        if loader is not None:
            with open(os.path.join(directory, f"train_dataloader_{i}.pkl"), 'wb') as f:
                pickle.dump(loader.dataset, f)
    
    # Save the testing DataLoaders' datasets
    for i, loader in enumerate(test_loaders):
        if loader is not None:
            with open(os.path.join(directory, f"test_dataloader_{i}.pkl"), 'wb') as f:
                pickle.dump(loader.dataset, f)

def load_train_test_dataloaders(directory, batch_size=128, shuffle=True):
    """
    Loads datasets from the specified directory and recreates DataLoaders.

    Args:
        directory: Directory where the datasets are saved.
        batch_size: Batch size for the DataLoaders.
        shuffle: Whether to shuffle the data.

    Returns:
        train_loaders: List of training DataLoaders.
        test_loaders: List of testing DataLoaders.
    """
    train_loaders, test_loaders = [], []

    # Load training datasets and create DataLoaders
    for file in sorted(os.listdir(directory)):
        if file.startswith("train_dataloader_"):
            with open(os.path.join(directory, file), 'rb') as f:
                dataset = pickle.load(f)
                train_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle))

    # Load testing datasets and create DataLoaders
    for file in sorted(os.listdir(directory)):
        if file.startswith("test_dataloader_"):
            with open(os.path.join(directory, file), 'rb') as f:
                dataset = pickle.load(f)
                test_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False))

    return train_loaders, test_loaders

def plot_stacked_bar_chart_and_labels(file_path):
    print("file_path",file_path)
    directoty=file_path.split("/")[0]+'/'+file_path.split("/")[1]
    print("directoty",directoty)
    

    # Load data from the Excel file
    data_train = pd.read_excel(file_path, sheet_name='Train Summary', index_col=0)
    data_test = pd.read_excel(file_path, sheet_name='Test Summary', index_col=0)

    # Combine train and test data by summing their counts for each node
    combined_data = pd.DataFrame({
        "Train": data_train.sum(axis=0),  # Sum across labels for train
        "Test": data_test.sum(axis=0)    # Sum across labels for test
    })

    # Combine train and test label distributions
    label_data = data_train.add(data_test, fill_value=0)

    # Create the first figure: Stacked bar chart for train and test
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    combined_data.plot(kind="bar", stacked=True, ax=ax1, color=["#1f77b4", "#ff7f0e"])
    ax1.set_title("Sample Distribution: Train vs Test", fontsize=16)
    ax1.set_xlabel("Nodes", fontsize=12)
    ax1.set_ylabel("Number of Samples", fontsize=12)
    ax1.legend(title="Dataset", loc='upper left', fontsize=10)

    # Display the total number of samples above each bar
    for i, (train, test) in enumerate(zip(combined_data["Train"], combined_data["Test"])):
        total = train + test
        ax1.text(i, total + 100, f"{int(total)}", ha="center", fontsize=10)

    # Save the first figure
    plt.tight_layout()
    fig1=directoty+"/sample_distribution_train_test.png"
    plt.savefig(fig1)

    # Create the second figure: Label distribution for each node
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    label_data.plot(kind="bar", ax=ax2, colormap="tab10", width=0.8)
    ax2.set_title("Label Distribution Across Nodes (Train + Test)", fontsize=16)
    ax2.set_xlabel("Nodes", fontsize=12)
    ax2.set_ylabel("Number of Samples", fontsize=12)
    ax2.legend(title="Labels", fontsize=10, bbox_to_anchor=(1, 1), loc='upper left')

    # Save the second figure
    plt.tight_layout()
    fig2=directoty+"/label_distribution_per_node.png"
    plt.savefig(fig2)

    # Show both figures
    plt.show()

def save__statistics(train_loaders, test_loaders, n_clients, file_path="mnist_summary.xlsx"):
    """
    Sauvegarde les statistiques du nombre d'échantillons par classe pour chaque client dans un fichier Excel.

    Args:
        train_loaders (list): Liste des DataLoaders d'entraînement.
        test_loaders (list): Liste des DataLoaders de test.
        n_clients (int): Nombre de clients (noeuds).
        file_path (str): Nom du fichier Excel de sortie.
    """
    # Initialiser les dictionnaires de comptage des échantillons par classe
    train_stats = {f"Node {i+1}": [0] * 10 for i in range(n_clients)}
    test_stats = {f"Node {i+1}": [0] * 10 for i in range(n_clients)}

    # Remplir les statistiques d'entraînement
    for client_id, loader in enumerate(train_loaders):
        for images, labels in loader:
            for label in labels:
                train_stats[f"Node {client_id+1}"][label.item()] += 1

    # Remplir les statistiques de test
    for client_id, loader in enumerate(test_loaders):
        for images, labels in loader:
            for label in labels:
                test_stats[f"Node {client_id+1}"][label.item()] += 1

    # Convertir les statistiques en DataFrame
    train_df = pd.DataFrame(train_stats)
    train_df.insert(0, "Class", list(range(10)))  # Ajouter colonne Class

    test_df = pd.DataFrame(test_stats)
    test_df.insert(0, "Class", list(range(10)))  # Ajouter colonne Class

    # Sauvegarde dans un fichier Excel
    with pd.ExcelWriter(file_path) as writer:
        train_df.to_excel(writer, sheet_name="Train Summary", index=False)
        test_df.to_excel(writer, sheet_name="Test Summary", index=False)

    print(f"✅ Statistiques enregistrées dans {file_path}")

def plot_distribution(file_path, save_path="save_path", dataset_name="MNIST"):
    xls = pd.ExcelFile(file_path)

    # Charger les données des feuilles "Train Summary" et "Test Summary"
    train_summary = pd.read_excel(xls, sheet_name="Train Summary")
    test_summary = pd.read_excel(xls, sheet_name="Test Summary")

    # Fusionner les statistiques d'entraînement et de test
    train_summary.set_index("Class", inplace=True)
    test_summary.set_index("Class", inplace=True)

    # Somme des échantillons train et test
    combined_summary = train_summary + test_summary



    plt.figure(figsize=(8, 5))
    ax = sns.heatmap(combined_summary, cmap="Blues", xticklabels=True, yticklabels=True)
    plt.xlabel("Edge Participant")
    plt.ylabel("Class")
    plt.title(f"Distribution of {dataset_name} Samples per Class and Edge Participant")

    # Ajouter une colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label("Number of Samples")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    # Afficher la figure
    plt.show()
    #save the figure
    


"""

mnist_directory = 'mnist_data_dirichlet'
if (not os.path.exists(mnist_directory)):
    os.makedirs(mnist_directory)
    train_loaders, test_loaders, _, _ = get_mnist_distribution(
    type="dirichlet", n_samples_train=200, n_samples_test=100, 
    n_clients=10, batch_size=32, shuffle=True, alpha=0.5
     )
    save__statistics(train_loaders, test_loaders, n_clients=10, file_path=mnist_directory+"/mnist_summary_2.xlsx")
    save_train_test_dataloaders(train_loaders, test_loaders, mnist_directory)
    plot_distribution(mnist_directory+"/mnist_summary_2.xlsx", save_path=mnist_directory+"/mnist_summary_2.png", dataset_name="MNIST")

else:
    print("Data already exists")
    train_loaders1, test_loaders1 = load_train_test_dataloaders(mnist_directory)
    
    
# The same code can be used for the other datasets
usps_directory = 'usps_data_dirichlet'
if (not os.path.exists(usps_directory)):
    os.makedirs(usps_directory)
    train_loaders, test_loaders, _, _ = get_usps_distribution(
    type="dirichlet", n_samples_train=200, n_samples_test=100, 
    n_clients=10, batch_size=32, shuffle=True, alpha=0.5
     )
    save__statistics(train_loaders, test_loaders, n_clients=10, file_path=usps_directory+"/usps_summary_2.xlsx")
    save_train_test_dataloaders(train_loaders, test_loaders, usps_directory)
    plot_distribution(usps_directory+"/usps_summary_2.xlsx", save_path=usps_directory+"/usps_summary_2.png", dataset_name="USPS")
else :
    print("Data already exists")
    train_loaders2, test_loaders2 = load_train_test_dataloaders(usps_directory)

svhn_directory = 'svhn_data_dirichlet'
if (not os.path.exists(svhn_directory)):
    os.makedirs(svhn_directory)
    train_loaders, test_loaders, _, _ = get_svhn_distribution(
    type="dirichlet", n_samples_train=200, n_samples_test=100, 
    n_clients=10, batch_size=32, shuffle=True, alpha=0.5
     )
    save__statistics(train_loaders, test_loaders, n_clients=10, file_path=svhn_directory+"/svhn_summary_2.xlsx")
    save_train_test_dataloaders(train_loaders, test_loaders, svhn_directory)
    plot_distribution(svhn_directory+"/svhn_summary_2.xlsx", save_path=svhn_directory+"/svhn_summary_2.png", dataset_name="SVHN")
else :
    print("Data already exists")
    train_loaders3, test_loaders3 = load_train_test_dataloaders(svhn_directory)

mnistm_directory = 'mnistm_data_dirichlet'
if (not os.path.exists(mnistm_directory)):
    os.makedirs(mnistm_directory)
    train_loaders, test_loaders, _, _ = get_mnistm_distribution(
    type="dirichlet", n_samples_train=200, n_samples_test=100, 
    n_clients=10, batch_size=32, shuffle=True, alpha=0.5
     )
    save__statistics(train_loaders, test_loaders, n_clients=10, file_path=mnistm_directory+"/mnistm_summary_2.xlsx")
    save_train_test_dataloaders(train_loaders, test_loaders, mnistm_directory)
    plot_distribution(mnistm_directory+"/mnistm_summary_2.xlsx", save_path=mnistm_directory+"/mnistm_summary_2.png", dataset_name="MNISTM")
else :
    print("Data already exists")
    train_loaders4, test_loaders4 = load_train_test_dataloaders(mnistm_directory)

emnist_directory = 'emnist_data_dirichlet'
if (not os.path.exists(emnist_directory)):
    os.makedirs(emnist_directory)
    train_loaders, test_loaders, _, _ = get_emnist_distribution(
    type="dirichlet", n_samples_train=200, n_samples_test=100, 
    n_clients=10, batch_size=32, shuffle=True, alpha=0.5
     )
    save__statistics(train_loaders, test_loaders, n_clients=10, file_path=emnist_directory+"/emnist_summary_2.xlsx")
    save_train_test_dataloaders(train_loaders, test_loaders, emnist_directory)
    plot_distribution(emnist_directory+"/emnist_summary_2.xlsx", save_path=emnist_directory+"/emnist_summary_2.png", dataset_name="EMNIST")
else :
    print("Data already exists")
    train_loaders5, test_load_loaders5 = load_train_test_dataloaders(emnist_directory)
"""
"""
# Use the function with the file path
# Parameters
nb_nodes = 10
min_samples_per_node = 500
max_samples_per_node = 70000
batch_size = 32
shuffle = True

## Function to get the mnist dataset
mnist_directory = 'data/mnist_data'
if (not os.path.exists(mnist_directory)):
    os.makedirs(mnist_directory)
    train_loaders, test_loaders, index_data, node_labels = get_mnist_distribution("random", 200, 100, nb_nodes, batch_size, shuffle)
    # Generate the summary and save it to an Excel file
    for i in range(len(train_loaders)):
        print("train_loader",i, len(train_loaders[i]))
        # print batch size of the first node
        print("batch size",i,train_loaders[i].batch_size)
    output_file = "mnist_summary.xlsx"
    file= mnist_directory + "/" + output_file
    summarize_data_by_node(train_loaders, test_loaders, node_labels,file)
    save_train_test_dataloaders(train_loaders, test_loaders, mnist_directory)
    plot_stacked_bar_chart_and_labels(file)
else :
    print("Data already exists")
    train_loaders1, test_loaders1 = load_train_test_dataloaders(mnist_directory)



# function to get the usps dataset

usps_directory = 'data/usps_data'
if (not os.path.exists(usps_directory)):
    os.makedirs(usps_directory)
    train_loaders, test_loaders, index_data, node_labels = get_usps_distribution("random", 200, 100, nb_nodes, batch_size, shuffle)
    # Generate the summary and save it to an Excel file
    output_file = "usps_summary.xlsx"
    file= usps_directory + "/" + output_file
    summarize_data_by_node(train_loaders, test_loaders, node_labels,file)
    save_train_test_dataloaders(train_loaders, test_loaders, usps_directory)
    plot_stacked_bar_chart_and_labels(file)
else :
    print("Data already exists")
    train_loaders2, test_loaders2 = load_train_test_dataloaders(usps_directory)


# function to get the svhn dataset
svhn_directory = 'data/svhn_data'
if (not os.path.exists(svhn_directory)):
    os.makedirs(svhn_directory)
    train_loaders, test_loaders, index_data, node_labels = get_svhn_distribution("random", 200, 100, nb_nodes, batch_size, shuffle)
    # Generate the summary and save it to an Excel file
    output_file = "svhn_summary.xlsx"
    file= svhn_directory + "/" + output_file
    summarize_data_by_node(train_loaders, test_loaders, node_labels,file)
    save_train_test_dataloaders(train_loaders, test_loaders, svhn_directory)
    plot_stacked_bar_chart_and_labels(file)

else:
    print("Data already exists")
    train_loaders3, test_loaders3 = load_train_test_dataloaders(svhn_directory)

# function to get the mnistm dataset
mnistm_directory = 'data/mnistm_data'
if (not os.path.exists(mnistm_directory)):
    os.makedirs(mnistm_directory)
    train_loaders, test_loaders, index_data, node_labels = get_mnistm_distribution("random", 200, 100, nb_nodes, batch_size, shuffle)
    # Generate the summary and save it to an Excel file
    output_file = "mnistm_summary.xlsx"
    file= mnistm_directory + "/" + output_file
    summarize_data_by_node(train_loaders, test_loaders, node_labels,file)
    save_train_test_dataloaders(train_loaders, test_loaders, mnistm_directory)
    plot_stacked_bar_chart_and_labels(file)
else:
    print("Data already exists")
    train_loaders4, test_loaders4 = load_train_test_dataloaders(mnistm_directory)

# function to get the emnist dataset
emnist_directory = 'data/emnist_data'
if (not os.path.exists(emnist_directory)):
    os.makedirs(emnist_directory)
    train_loaders, test_loaders, index_data, node_labels = get_emnist_distribution("random", 200, 100, nb_nodes, batch_size, shuffle)
    # Generate the summary and save it to an Excel file
    output_file = "emnist_summary.xlsx"
    file= emnist_directory + "/" + output_file
    summarize_data_by_node(train_loaders, test_loaders, node_labels,file)
    save_train_test_dataloaders(train_loaders, test_loaders, emnist_directory)
    plot_stacked_bar_chart_and_labels(file)
else:
    print("Data already exists")
    train_loaders5, test_loaders5 = load_train_test_dataloaders(emnist_directory)


#

"""