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
import matplotlib; matplotlib.use('Agg')
from matplotlib import pyplot, rcParams, patches

rcParams.update({'figure.autolayout': True})


def get_hist(p, y):
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


def get_roc(p, y, data=None, confidence_interval=None):
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