from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import sklearn


'''
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.1)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

tsne = TSNE(n_components=1, perplexity=30)
embedding = tsne.fit_transform(X)
plt.scatter(embedding[:,0], np.zeros(len(embedding)), c=y)
plt.show()

'''


def TSNE_f(X,y,dim):

    
    X = X[:500]
    y = y[:500]

    tsne = TSNE(n_components=dim)
    tsne_result = tsne.fit_transform(X)
    from matplotlib import pyplot as plt
    plt.figure(figsize=(6, 5))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'aqua', 'orange', 'purple'
    for i, c, label in zip(range(len(X)), colors, y):
       plt.scatter(tsne_result[y == i, 0], tsne_result[y == i, 1], c=c, label=i)
    plt.legend()
    plt.show()


'''
    tsne = TSNE(n_components=dim, perplexity=30)
    embedding = tsne.fit_transform(X)
    plt.scatter(embedding[:,0], np.zeros(len(embedding)), c=y)
    plt.show()

    return embedding
    
    '''