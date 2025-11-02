
import torchvision.datasets as datasets
from torchvision import transforms    
from Algorithms.usps import get_usps
from trash.svhn import get_svhn
from Algorithms.emnist import get_emnist
from Algorithms.mnist import get_mnist
from Algorithms.mnistm import MNISTM
from sklearn.manifold import TSNE
from data.TSNE import TSNE_f
import numpy as np
import matplotlib.pyplot as plt
import  pandas as pd
import seaborn as sns
if __name__ == '__main__': 


    print('MNIST:')
    mnist_trainset = get_mnist(path='.',train=True)
    mnist_testset = get_mnist(path='.',train=False)
    mnist_trainset.data=mnist_trainset.data.reshape(-1,28*28)

    X = mnist_trainset.data[:500]
    y = mnist_trainset.targets[:500]

    tsne = TSNE(n_components=3)
    tsne_result = tsne.fit_transform(X)

    print(y)
    from matplotlib import pyplot as plt
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'aqua', 'orange', 'purple'
    for i, c, label in zip(range(len(X)), colors, y):
       print(i, c, label)
       plt.scatter(tsne_result[y == i, 0], tsne_result[y == i, 1], c=c, label=i)
    plt.legend()
    plt.show()





'''
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': mnist_trainset.targets})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    #ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    plt.show()



    #plt.scatter(embedding[:,0], np.zeros(len(embedding)), c=mnist_testset.targets)
 
    #plt.show()

    #TSNE_f(mnist_trainset.data, mnist_trainset.targets, 3)





    print(mnist_trainset.data.shape)
    print(mnist_testset.data.shape)
   
    
    print('USPS:')
    usps_trainset=get_usps(train=True) #prepared with dataloader
    usps_testset=get_usps(train=False) ##prepared with dataloader
    
    print('SVHN:')
    svhn_trainset=get_svhn(path='.',split='train')
    svhn_testset=get_svhn(path='.',split='test')

    print(svhn_trainset.data.shape)
    print(svhn_testset.data.shape)
    
    print('MNISTM:')
    mnistm_trainset=MNISTM(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    mnistm_testset=MNISTM(root='.', train=False, download=True, transform=None)

    print(mnistm_trainset.train_data.shape)
    print(mnistm_testset.test_data.shape)
    
    print('EMNIST:')

    
    emnist_trainset=get_emnist(path='.',split='digits',train=True)
    emnist_testset=get_emnist(path='.',split='digits',train=False)

    print(emnist_trainset.data.shape)
    print(emnist_testset.data.shape)


'''


    






