import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.util.spn_util import fast_conditional_gradient
from src.util.CSVUtil import learn_piecewise_from_file
from src.util.text_util import printmd, strip_dataset_name
from src.util.spn_util import get_categoricals

from spn.structure.Base import Leaf, get_nodes_by_type
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear

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

    # path to the dataset you vor allem darstellen training
    dataset = '../../example_data/top20medical.csv'
    
    # the minimum number of datapoints that are included in a child of a 
    # sum node
    min_instances = 50
    
    # the parameter which governs how strict the independence test will be
    # 1 results in all features being evaluated as independent, 0 will 
    # result in no features being acccepted as truly independent
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
    
    all_bins = get_bins_from_spn(spn, 0)
    print(find_nearest(all_bins, np.array([0, 0])))
    print(all_bins)
    

