import numpy as np
from matplotlib import pyplot as plt


# The first 5000 examples of the test set are taken from the original NIST training set. The last 5000 are taken from the original NIST test set. The first 5000 are cleaner and easier than the last 5000.
# train-images-idx3-ubyte: training set images
# train-labels-idx1-ubyte: training set labels
# t10k-images-idx3-ubyte:  test set images
# t10k-labels-idx1-ubyte:  test set labels


def plot_single_sample(data,label):

    plt.figure()
    plt.imshow(data, cmap = 'gray')
    plt.title(str(label))
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.show()

def plot_multiple_samples(data,labels,n):
    m = len(n)
    fig, axes = plt.subplots(ncols = m,sharex=False, sharey=True, figsize=(10, 4))
    for i in range(m):
        label = int(labels[n[i]])
        axes[i].set_title(label)
        axes[i].imshow(data[n[i],:,:], cmap='gray')
        axes[i].get_xaxis().set_visible(False)
        axes[i].get_yaxis().set_visible(False)
        
    plt.show()
