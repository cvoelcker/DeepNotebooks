import numpy as np

from spn.algorithms.Inference import log_likelihood

from tqdm import tqdm


def shap_sampling(spn, data, target, features=None, N=1000):
    if features is None:
        features = list(spn.scope)
        features.remove(target)
    shap_values = np.zeros_like(data)
    data_save = data.copy().repeat(N, axis=0)
    query_with = np.zeros_like(data_save)
    query_without = np.zeros_like(data_save)
    for j in features:
        query_with[:] = data_save[:]
        query_without[:] = data_save[:]
        mask = np.random.choice([True, False], size=(N * data.shape[0], query_with.shape[1]))

        # calculation phi with
        mask[:, j] = False
        mask[:, target] = False
        query_with[mask] = np.nan
        ll_with = log_likelihood(spn, query_with)
        mask[:, target] = True
        query_with[mask] = np.nan
        evidence_with = log_likelihood(spn, query_with)
        cond_ll_with = np.exp(ll_with-evidence_with)

        # calculation phi without
        mask[:, j] = True
        mask[:, target] = False
        query_without[mask] = np.nan
        ll_without = log_likelihood(spn, query_without)
        mask[:, target] = True
        query_without[mask] = np.nan
        evidence_without = log_likelihood(spn, query_without)
        cond_ll_without = np.exp(ll_without-evidence_without)
     
        for i in range(len(data)):
            shap_values[i, j] = np.mean(cond_ll_with[i * N:(i + 1) * N] - cond_ll_without[i * N:(i + 1) * N])
    return shap_values
