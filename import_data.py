import idx2numpy
import numpy as np


def import_training_data():
    imageFile = 'data/train-images-idx3-ubyte'
    labelsFile = 'data/train-labels-idx1-ubyte'
    data = idx2numpy.convert_from_file(imageFile)
    # arr is a np.ndarray type of object of shape (60000, 28, 28)
    labels = idx2numpy.convert_from_file(labelsFile)
    # labels is a np.ndarray type of object of shape (60000, )
    
    return data, labels