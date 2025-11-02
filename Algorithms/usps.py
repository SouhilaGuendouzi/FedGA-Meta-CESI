"""Dataset setting and data loader for USPS.
Modified from
https://github.com/mingyuliutw/CoGAN/blob/master/cogan_pytorch/src/dataset_usps.py
"""

import gzip
import os
import pickle
import urllib

import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try: 
  from data.createSets import iid_split, non_iid_split, non_equitable_random_split, dirichlet_non_iid_split
except: 
  from createSets import iid_split, non_iid_split, non_equitable_random_split, dirichlet_non_iid_split
from torch.utils.data import ConcatDataset
#import params


class USPS(data.Dataset):
    """USPS Dataset.
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, resample from dataset randomly.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, train=True, transform=None, download=True):
        """Init USPS dataset."""
        # init params
        self.root = os.path.expanduser(root)
        self.filename = "usps_28x28.pkl"
        self.train = train
        # Num of Train = 7438, Num ot Test 1860
        self.transform = transform
        self.dataset_size = None

        # download dataset.
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        self.train_data, self.train_labels = self.load_samples()
       
        print(self.train_data.shape)
        if self.train:
            total_num_samples = self.train_labels.shape[0]
            indices = np.arange(total_num_samples)
            np.random.shuffle(indices)
            self.train_data = self.train_data[indices[0:self.dataset_size], ::]
            self.train_labels = self.train_labels[indices[0:self.dataset_size]]
        self.train_data *= 255.0
        self.train_data = self.train_data.transpose(
            (0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, label = self.train_data[index, ::], self.train_labels[index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label).item()])
        # label = torch.FloatTensor([label.item()])
        return img, label

    def __len__(self):
        """Return size of dataset."""
        return self.dataset_size

    def _check_exists(self):
        """Check if dataset is download and in right place."""
        return os.path.exists(os.path.join(self.root, self.filename))

    def download(self):
        """Download dataset."""
        filename = os.path.join(self.root, self.filename)
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        if os.path.isfile(filename):
            return
        print("Download %s to %s" % (self.url, os.path.abspath(filename)))
        urllib.request.urlretrieve(self.url, filename)
        print("[DONE]")
        return

    def load_samples(self):
        """Load sample images from dataset."""
        filename = os.path.join(self.root, self.filename)
        f = gzip.open(filename, "rb")
        data_set = pickle.load(f, encoding="bytes")
        f.close()
        if self.train:
            images = data_set[0][0]
            labels = data_set[0][1]
            self.dataset_size = labels.shape[0]
        else:
            images = data_set[1][0]
            labels = data_set[1][1]
            self.dataset_size = labels.shape[0]

        
        return images, labels


def get_usps(path='.', train=True):
    """Get USPS dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(), 
                                      ]) 
                                    

    # dataset and data loader
    usps_dataset = USPS(root=path,
                        train=train,
                        transform=pre_process,
                        download=True)

    

    usps_data_loader = torch.utils.data.DataLoader(
        dataset=usps_dataset,
       batch_size=64,   #params.batch_size
  shuffle=True)
    


    return  usps_dataset


def get_usps_distribution(type="iid", n_samples_train=200, n_samples_test=100, n_clients=3, batch_size=25, shuffle=True, alpha=0.5):
    dataset_loaded_train = get_usps("./data",train=True)
    dataset_loaded_test = get_usps("./data",train=False)
     #merge the two datasets and put them into one new dataset variable
    global_dataset = ConcatDataset([dataset_loaded_train, dataset_loaded_test])
    n_samples=len(global_dataset)*0.15
    print("n_samples",n_samples, 'length global dta',len(global_dataset))
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
        print("dirichlet")
        train, test, class_distribution = dirichlet_non_iid_split(global_dataset, n_clients, alpha, batch_size, shuffle)
        index = None  # Pas d'index spécifique dans ce cas
        node_labels = None  # Pas de labels spécifiques
    return train, test, index, node_labels



