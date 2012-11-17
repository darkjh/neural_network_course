"""This module provides the data and a whitening function for the 
mini-project 1 of the Unsupervised and Reinforcement Learning class.

Usage:

    >>> import mix
    >>> data2 = mix.miximages2()
    >>> data2 = whiten(data2)
    >>> data8 = mix.miximages8()
    >>> data8 = whiten(data8)
    >>> # etc...
"""

import numpy as np
import matplotlib.image as mpimg

def miximages2():
    """Return 2 linear mixtures of grayscale images.
    """

    return miximages(2)

def miximages8():
    """Return 8 linear mixtures of grayscale images.
    """
    return miximages(8)

    
def miximages(N):
    files = [('i%i.gif' % i) for i in range(1,N+1)]
    source = np.zeros((N,256*256))
    for i in range(N):
        source[i,:] = mpimg.imread(files[i]).flatten()
    
    mix = np.random.rand(N,N)
    data = np.dot(mix,source)
    return data


def whiten(data):
    """Return a whitened version of data.

    data should be a numpy array of size (data_width, n_samples).
    """
    data = data.T
    C = np.cov(data.T)
    d, v = np.linalg.eig(C)
    wm = v / np.sqrt(d)
    return np.dot(data, wm)
