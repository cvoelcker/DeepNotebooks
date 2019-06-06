import numpy as np


def find_nearest(array, value):
    idx = (np.abs(np.reshape(array, (-1, 1)) - np.reshape(value, (1, -1)))).argmin(axis = 0)
    return idx

