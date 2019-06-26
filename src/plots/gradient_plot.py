import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.util.spn_util import fast_conditional_gradient
from src.util.CSVUtil import learn_piecewise_from_file
from src.util.text_util import printmd, strip_dataset_name
from src.util.spn_util import get_categoricals

from src.shap_sampling import shap_sampling

from spn.structure.Base import Leaf, get_nodes_by_type
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear

from spn.algorithms.Inference import log_likelihood

from colour import Color
white = Color("white")
colors = list(white.range_to(Color("blue"),10))

print([c.hex for c in colors])

def get_bins_from_spn(spn, feature_id):
    leaves = get_nodes_by_type(spn, Leaf)
    all_bins = np.array([])
    for leaf in leaves:
        if feature_id in leaf.scope:
            all_bins = np.append(all_bins, leaf.bin_repr_points)
    all_bins.sort()
    all_bins = np.unique(all_bins)
    return all_bins


def find_nearest(array, value):
    idx = (np.abs(np.reshape(array, (-1, 1)) - np.reshape(value, (1, -1)))).argmin(axis = 0)
    return idx

if __name__ == "__main__":

    dataset = '../../example_data/top20medical.csv'
    pickle.dump(dataset, open('bullshit.save', 'wb'))
    min_instances = 50
    independence_threshold = 0.1
    spn, dictionary = learn_piecewise_from_file(
        data_file=dataset, 
        header=0, 
        min_instances=min_instances, 
        independence_threshold=independence_threshold, )
    df = pd.read_csv(dataset)
    context = dictionary['context']
    context.dataset = strip_dataset_name(dataset)
    categoricals = get_categoricals(spn, context)
    
    data = df.values.copy()
    
    from time import time

    all_shap = []
    results = []

    for i in range(3):
        data[:, -1] = i
        t = time()
        print('Still wök {}'.format(i))
        shap = shap_sampling(spn, data, 20, N=50)
        all_shap.append(shap)
        print(t - time())

    pickle.dump(all_shap, open('shap_values_all.save', 'wb'))

    all_shap = np.array(pickle.load(open('shap_values_all.save', 'rb')))
    all_shap = np.concatenate(all_shap, axis=0)

