'''
18/11/02
@author Claas Voelcker
'''
import numpy as np

from spn.algorithms.Inference import likelihood
from spn.algorithms.MPE import mpe
from spn.algorithms.Condition import condition
import spn.algorithms.Gradient
from spn.algorithms.Gradient import feature_gradient

from spn.structure.StatisticalTypes import MetaType

def func_from_spn(spn, feature_id):
    size = len(spn.scope)
    def func(x):
        query = np.zeros((len(x), size))
        query[:] = np.nan
        query[:, feature_id] = x
        return likelihood(spn, query)
    return func


def predict_mpe(spn, feature_id, query, context):
    query[:, feature_id] = np.nan
    return mpe(spn, query, in_place=True)[:, feature_id]


def get_categoricals(spn, context):
    return [i for i in spn.scope if context.meta_types[i] == MetaType.DISCRETE]


def fast_conditional_gradient(spn, evidence_data, data):
    cond_spn = condition(spn, evidence_data)
    gradients = feature_gradient(cond_spn, data)
    return gradients
