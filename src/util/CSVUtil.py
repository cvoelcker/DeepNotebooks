"""
created 18/11/01
@author Claas Völcker
"""
import numpy as np
import pandas as pd
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from spn.algorithms.LeafLearning import learn_leaf_from_context
from spn.algorithms.LearningWrappers import learn_mspn
from spn.algorithms.StructureLearning import is_valid

from spn.structure.Base import Context
from spn.structure.leaves.histogram.Histograms import Histogram
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear, create_piecewise_leaf


def load_from_csv(data_file, header=0):
    df = pd.read_csv(data_file, delimiter=",", header=header)
    df = df.dropna(axis=0, how='any')

    feature_names = df.columns.values.tolist() if header == 0 else [
        "X_{}".format(i) for i in range(len(df.columns))]

    dtypes = df.dtypes
    feature_types = []
    for feature_type in dtypes:
        if feature_type.kind == 'O':
            feature_types.append('hist')
        else:
            feature_types.append('piecewise')

    data_dictionary = {
        'features': [{"name": name,
                      "type": typ,
                      "pandas_type": dtypes[i]}
                     for i, (name, typ)
                     in enumerate(zip(feature_names, feature_types))],
        'num_entries': len(df)
    }

    idx = df.columns

    for id, name in enumerate(idx):
        if feature_types[id] == 'hist':
            lb = LabelEncoder()
            data_dictionary['features'][id]["encoder"] = lb
            df[name] = df[name].astype('category')
            df[name] = lb.fit_transform(df[name])
            data_dictionary['features'][id]["values"] = lb.transform(
                lb.classes_)
        if dtypes[id].kind == 'M':
            df[name] = (df[name] - df[name].min()) / np.timedelta64(1, 'D')

    data = np.array(df)

    return data, feature_types, data_dictionary


def learn_piecewise_from_file(data_file, header=0, min_instances=25, independence_threshold=0.1):
    """
    Learning wrapper for automatically building an SPN from a datafile

    :param data_file: String: location of the data csv
    :param header: Int: row of the data header
    :param min_instances: Int: minimum data instances per leaf node
    :param independence_threshold: Float: threshold for the independence test
    :param histogram: Boolean: use histogram for categorical data?
    :return: a valid spn, a data dictionary
    """
    data, feature_types, data_dictionary = load_from_csv(data_file, header)
    feature_classes = [Histogram if name == 'hist' else PiecewiseLinear for name in feature_types]
    context = Context(parametric_types=feature_classes).add_domains(data)
    context.add_feature_names([entry['name']
                                  for entry in data_dictionary['features']])
    spn = learn_mspn(data,
                     context,
                     min_instances_slice=min_instances,
                     threshold=independence_threshold,
                     ohe=False,
                     leaves=create_piecewise_leaf)
    assert is_valid(spn), 'No valid spn could be created from datafile'
    data_dictionary['context'] = context
    return spn, data_dictionary
