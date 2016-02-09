"""A module for analysis and plotting

Summary of available functions:

    - Create histogram of benign and malignant probabilities
    - Create sensitiviy/specificity curve
    - Make analysis plots and save to disk

"""

import os
import numpy as np
import json
from scipy import stats
from matplotlib import pyplot, rcParams, patches

rcParams.update({'figure.autolayout': True})


def _get_hist(p, y):
    """Create histogram of benign and malignant probabilities

    Args:
        p (numpy.array): 1D array of malignant probabilities
        y (numpy.array): 1D array of label integers

    Returns:
        fig (matplotlib.figure): The histogram
    """

    nbins = 30
    bins = np.linspace(0, 1, nbins)

    fig = pyplot.figure(figsize=(8,8))
    pyplot.hist(p[y==0], bins=bins, alpha=0.5, normed=True, facecolor='green')
    pyplot.hist(p[y==1], bins=bins, alpha=0.5, normed=True, facecolor='red')
    pyplot.xlim(0, 1)
    pyplot.ylim(0, nbins/2)
    pyplot.ylabel('normalized count', fontsize=18, labelpad=20)
    pyplot.xlabel('probability malignant', fontsize=18, labelpad=10)
    
    return fig


def _get_roc(p, y, data=None, confidence_interval=None):
    """Create sensitiviy/specificity curve

    Args:
        p (numpy.array): 1D array of malignant probabilities
        y (numpy.array): 1D array of label integers
        data (dict): Other data to plot (e.g. dermatologists results)
        confidence_interval (float): Confidence interval [0,1] of
            uncertainty ellipses

    Returns:
        fig (matplotlib.figure): The SS curve
    """
    
    thresholds = np.sort(np.concatenate([
        np.linspace(0, 1, 50),
        np.logspace(0, -5, 50),
        1 - np.logspace(0, -5, 50)
    ]))

    plot_kwargs = {
        'net': {'c':'b', 'ls': '-', 'marker': 'o'},
        'derms': {'c':'r', 'ls': 'None', 'marker': 'o'},
        'novices': {'c':'y', 'ls': 'None', 'marker': 'o'},
        'melafind': {'c': '0.5', 'ls': '-', 'marker': 'o'}
    }

    if data is None: data = {}

    data['net'] = [
        {
            'specificity': np.mean(p[y==0]<t),
            'sensitivity': np.mean(p[y==1]>t),
            'num_benign': sum(y==0),
            'num_malignant': sum(y==1)
        }
        for t in thresholds
    ]

    fig = pyplot.figure(figsize=(8,8))
    ax = pyplot.subplot(111, aspect='equal')

    for group in data:

        p0 = [i['specificity'] for i in data[group]]
        p1 = [i['sensitivity'] for i in data[group]]

        pyplot.plot(p0, p1, **plot_kwargs[group])

        if confidence_interval:

            n0 = [i['num_benign'] for i in data[group]]
            n1 = [i['num_malignant'] for i in data[group]]

            p0_intervals = [
                stats.beta.interval(confidence_interval, p*n+1, (1-p)*n+1)
                for p, n in zip(p0, n0) if n > 0
            ]

            p1_intervals = [
                stats.beta.interval(confidence_interval, p*n+1, (1-p)*n+1) 
                for p, n in zip(p1, n1) if n > 0
            ]

            for dp0, dp1 in zip(p0_intervals, p1_intervals):

                ax.add_artist(patches.Ellipse(
                    xy = [
                        np.mean(dp0),
                        np.mean(dp1)
                    ], 
                    width = np.diff(dp0),
                    height = np.diff(dp1),
                    facecolor = 'none',
                    linestyle = 'dotted',
                    edgecolor = plot_kwargs[group]['c']
                ))

    pyplot.ylabel('sensitivity', fontsize=18, labelpad=30)
    pyplot.xlabel('specificity', fontsize=18, labelpad=30)

    return fig


def make_plots(net_name, step, phase, confidence_interval=0.8):
    """Make analysis plots and save to disk

    Args:
        net_name (string): Directory where net stuff is saved
        step (int): Training iteration to analyze
        phase (string): Either `train` or `test`
        confidence_interval (float): Confidence interval [0,1] of
            uncertainty ellipses. If `None`, no ellipses are plotted
    """

    logits = np.load(net_name+'/logits_{}_{}.npy'.format(phase, step))
    with open(net_name+'/label_names.txt') as f:
        label_names = json.load(f)
    with open(net_name+'/logits_info_{}.txt'.format(phase)) as f:
        logits_info = json.load(f)
    with open('../data/competition.txt') as f:
        competition_data = json.load(f)

    save_path = os.path.join(net_name, 'plots', phase)

    b = np.logical_not(np.isnan(logits.sum(1)))
    Y, Z = np.array(logits_info['labels'])[b], logits[b].astype('float32')

    for category in ['pigmented', 'epidermal']:

        i = label_names.index('{} benign'.format(category))
        j = label_names.index('{} malignant'.format(category))

        if category not in competition_data: competition_data[category] = None

        IJ = np.logical_or(Y==i, Y==j)
        y = Y[IJ].copy()
        y[y==i] = 0
        y[y==j] = 1

        p = np.exp(Z[IJ][:,[i,j]])
        p = p[:,1]/p.sum(axis=1)

        hist_fig = _get_hist(p, y)
        hist_path = os.path.join(save_path, category, 'hist')
        if not os.path.exists(hist_path): os.makedirs(hist_path)
        hist_fig.savefig(hist_path + '/step{}.svg'.format(step))

        roc_fig = _get_roc(p, y, competition_data[category], confidence_interval)
        roc_path = os.path.join(save_path, category, 'roc')
        if not os.path.exists(roc_path): os.makedirs(roc_path)
        roc_fig.savefig(roc_path + '/step{}.svg'.format(step))

    pyplot.close('all')