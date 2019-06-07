import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.util.spn_util import fast_conditional_gradient
from src.util.CSVUtil import learn_piecewise_from_file
from src.util.text_util import printmd, strip_dataset_name
from src.util.spn_util import get_categoricals
from src.util.plot_util import IndexColormap

from src.shap_sampling import shap_sampling

from spn.structure.Base import Leaf, get_nodes_by_type
from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear

from spn.algorithms.Inference import log_likelihood


colors = IndexColormap('viridis', 20)
colors = [colors[i] for i in range(20)]

matplotlib.use('pgf')
plt.style.use('seaborn')

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
    # pickle.dump(dataset, open('bullshit.save', 'wb'))
    # min_instances = 50
    # independence_threshold = 0.1
    # spn, dictionary = learn_piecewise_from_file(
    #     data_file=dataset, 
    #     header=0, 
    #     min_instances=min_instances, 
    #     independence_threshold=independence_threshold, )
    df = pd.read_csv(dataset)
    # context = dictionary['context']
    # context.dataset = strip_dataset_name(dataset)
    # categoricals = get_categoricals(spn, context)
    
    data = df.values.copy()
    
    # results = []
    # for i in range(3):
    #     data[:, -1] = i
    #     results.append(log_likelihood(spn, data))
    # prediction = np.array(results).squeeze()
    # prediction = np.argmax(prediction, axis=0)
    # print(prediction.shape)

    # data[:, -1] = prediction

    prediction_data = pickle.load(open('prediction.save', 'rb'))

    for i in range(20):
        if np.nanmax(prediction_data, axis=0)[i] > 100:
            prediction_data[:, i] = np.log(prediction_data[:, i])


    feature_types = []
    with open('../../example_data/top20medical.features', 'r') as ff:
        for line in ff.readlines():
            type_info = line.strip('\n').split(',')[1]
            feature_types.append(type_info)
    
    # from time import time
    
    # all_shap = []
    # for i in range(5):
    #     t = time()
    #     print('Still wök {}'.format(i))
    #     shap = shap_sampling(spn, data[i * 100: (i + 1) * 100], 20, N=50)
    #     all_shap.append(shap)
    #     print(t - time())

    # pickle.dump(all_shap, open('shap_values.save', 'wb'))

    all_shap = pickle.load(open('shap_values_all.save', 'rb'))

    # bins_visualization = 
    n_bins = 20

    print(df.columns)
    for prediction in [0, 1, 2]:
        for feature in []:
            if feature_types[feature] == 'cat':
                domains = np.unique(prediction_data[:, feature][~np.isnan(prediction_data[:, feature])])
                shap_samples_per_feature = []
                for domain in domains:
                    selection = np.where((prediction_data[:, feature] == domain) & (prediction_data[:, -1] == prediction) & (~np.isnan(prediction_data[:, feature])))
                    data_points = prediction_data[selection]
                    shap_samples_per_feature.append(all_shap[selection, feature].reshape(-1))
                plt.hist(shap_samples_per_feature, 20, range=(-.5, .5), histtype='bar', stacked=True, label=np.round(domains, 2))
                plt.legend(loc="upper right")
                plt.title('Influence of feature {} on prediction {} for diagnosis'.format(df.columns[feature], prediction))
                plt.show()
            else:
                _min = np.min(prediction_data[:, feature][~np.isnan(prediction_data[:, feature])])
                _max = np.max(prediction_data[:, feature][~np.isnan(prediction_data[:, feature])])
                domains = np.linspace(_min, _max, 20)
                all_nearest = find_nearest(domains, prediction_data[:, feature])
                shap_samples_per_feature = []
                for i, domain in enumerate(domains):
                    selection = np.where((all_nearest == i) & (prediction_data[:, -1] == prediction) & (~np.isnan(prediction_data[:, feature])))
                    data_points = prediction_data[selection]
                    shap_samples_per_feature.append(all_shap[selection, feature].reshape(-1))
                plt.hist(shap_samples_per_feature, 20, range=(-.5, .5), histtype='bar', stacked=True, color=colors, label=np.round(domains, 2))
                plt.legend(loc="upper right")
                plt.title('Influence of feature {} on prediction {} for diagnosis'.format(df.columns[feature], prediction))
                plt.show()

    from shap import summary_plot

    summary_plot(all_shap, prediction_data, df.columns, class_names=['negative', 'type 1', 'type 2'], show = False)
    plt.tight_layout()
    plt.savefig('pdf/all_classes_summary.pdf')
    plt.clf()

    s = all_shap[0]
    summary_plot(s, prediction_data, df.columns, sort=False, auto_size_plot=False, plot_type='dot', show = False)
    plt.xlim(-.4,.4)
    plt.tight_layout()
    plt.savefig('pdf/class_1_summary.pdf')
    plt.clf()

    s = all_shap[1]
    summary_plot(s, prediction_data, df.columns, sort=False, auto_size_plot=False, plot_type='dot', show = False)
    plt.xlim(-.4,.4)
    plt.tight_layout()
    plt.savefig('pdf/class_2_summary.pdf')
    plt.clf()

    s = all_shap[2]
    summary_plot(s, prediction_data, df.columns, sort=False, auto_size_plot=False, plot_type='dot', show = False)
    plt.xlim(-.4,.4)
    plt.tight_layout()
    plt.savefig('pdf/class_3_summary.pdf')
    plt.clf()

