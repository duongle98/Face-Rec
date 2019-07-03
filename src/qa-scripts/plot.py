# plot by https://krasserm.github.io/2018/02/07/deep-face-recognition/

from qa_utils import chop_extension
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def print_metrics(metric_1,
                  metric_2,
                  thresholds,
                  opt_idx,
                  save_path):
    # Using save path to find out name of metric_1 and metric_2
    # for EXp: /home/loingo/Precision-Recall.jpg -> metric_1:Precision, metric_2:Recall
    metrics = chop_extension(save_path).split('-')
    metric_1_name = metrics[0]
    metric_2_name = metrics[1]
    opt_threshold = round(thresholds[opt_idx], 2)
    opt_metric_1 = round(metric_1[opt_idx], 2)
    opt_metric_2 = round(metric_2[opt_idx], 2)

    fig = plt.figure()
    plt.plot(thresholds, metric_1, label=metric_1_name)
    plt.plot(thresholds, metric_2, label=metric_2_name)
    plt.axvline(x=opt_threshold, linestyle='--', lw=1, c='lightgrey', Label='Threshold')
    plt.title('Precision = {}, Recall = {} at threshold {}'.format(opt_metric_1,
                                                                   opt_metric_2,
                                                                   opt_threshold))
    plt.xlabel('Distance threshold')
    plt.legend()
    fig.savefig(save_path)
    # print('Metric plot saved at', save_path)


def print_distance_histogram(dists,
                             actual_issame,
                             opt_tau,
                             save_path):
    dist_pos = dists[actual_issame == 1]
    dist_neg = dists[actual_issame == 0]
    # print(actual_issame)

    fig = plt.figure(figsize=(12, 4))

    plt.subplot(121)
    plt.hist(dist_pos)
    plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    plt.title('Distances (pos. pairs)')
    plt.legend()

    plt.subplot(122)
    plt.hist(dist_neg)
    plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
    plt.title('Distances (neg. pairs)')
    plt.legend()

    fig.savefig(save_path)
    # print('Histogram plot saved at', save_path)


def TSNE_visualize(embeddings, labels, save_path):
    X_embedded = TSNE(n_components=2).fit_transform(embeddings)
    labels = np.array(labels)
    fig = plt.figure()
    for i, t in enumerate(set(labels)):
        idx = labels == t
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

    plt.legend(bbox_to_anchor=(1, 1))
    fig.savefig(save_path, bbox_inches='tight')
