import mainAlg as ea
import numpy as np
from operator import itemgetter
import random, math
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
from collections import OrderedDict
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn.svm import SVR
from scipy.optimize import curve_fit
# in first run asymmetric gave the best result.
# print(ea.muCommaLambdaEA(10, 100, 20))
from sklearn.preprocessing import PolynomialFeatures


def findTournSelEAMean(runs, times, str_size, n_type, p_noise):
    """

    :param runs:  How many means will be taken
    :param times: How many times an EA will run so one mean will be taken each run for how many times.
    :param str_size:
    :param n_type: Noise type if this is 'one', one Bit Noise will be used, asymmetric one Bit if 'asym' and no noise in other cases
    :param p_noise: Noise strength
    :return:
    """
    means = []
    results = []
    p_mut = 0.3 / str_size
    for t in range(times):

        for i in range(runs):
            result = ea.tournSelEA(str_size, p_mut, n_type, p_noise)
            if result is None:
                break
            results.append(result)
        # print(np.mean(asym_results))
        means.append(np.mean(results))
        results = []
    return means


def plotTournSelEAMeans():
    """

    :return: Boxplot variables for tournament selection EA without noise
    """
    str_sizes = [5, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 70, 80]
    orig_plot = []
    oneB_plot = []
    asym_plot = []
    result = []
    for i in str_sizes:
        muCommaLambda_stuff = []
        print(i)
        muCommaLambda_stuff = findTournSelEAMean(100, 10, i, '', 0)
        orig_plot.append(muCommaLambda_stuff)
        muCommaLambda_stuff = findTournSelEAMean(100, 10, i, 'one', 0.5)
        oneB_plot.append(muCommaLambda_stuff)
        muCommaLambda_stuff = findTournSelEAMean(100, 10, i, 'asym', 0.5)
    return orig_plot, oneB_plot, asym_plot


def findMuCommaLambdaMean(runs, times, str_size, n_type, p_noise):
    """

    :param runs:  How many means will be taken
    :param times: How many times an EA will run so one mean will be taken each run for how many times.
    :param str_size:
    :return: An array which includes 10 means of 100 runs of (Mu, Lambda) EA
    """
    means = []
    results = []
    for t in range(times):

        for i in range(runs):

            result = ea.muCommaLambdaEA(str_size, n_type, p_noise)
            if result is None:
                break
            results.append(result)
        # print(np.mean(asym_results))
        means.append(np.mean(results))
        results = []
    return means


def plotOneBitTournSelEAMeansSize():
    """

    :return: Boxplot variables for (Mu, Lambda)-EA with One-Bit Noise for different string lengths
    """

    noise_str = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    plot_stuff = []
    result = []
    for i in noise_str:
        print(i)
        str_size = 25
        muCommaLambda_stuff = []
        p_mut = 0.2 / str_size
        muCommaLambda_stuff = findTournSelEAMean(100, 10, str_size, 'one', i)
        plot_stuff.append(muCommaLambda_stuff)

    return plot_stuff


def plotMuCommaLambdaMeans():
    """

    :return: Boxplot variables for (Mu, Lambda)-EA for different string lengths
    """

    str_sizes = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 100, 200, 300]
    plot_stuff = []
    result = []
    for i in str_sizes:
        print(i)
        muCommaLambda_stuff = findMuCommaLambdaMean(100, 10, i)
        plot_stuff.append(muCommaLambda_stuff)
    return plot_stuff


def thirdPolyModel(x, a, b):
    """

    :param x: String length
    :param a: Constant
    :param b: Power
    :return: Third degree polynomial model in form of a*x^b
    """
    return (a * np.power(x, b))


# Runtime in initial proof is 8n^3 ln(12n^2ln(n^4))

def plotTournSelAll():
    """

    :return: Boxplots for tournament selection EA for:
                without noise
                with one-bit noise
                with asymmetric one-bit noise
    """
    str_sizes = [5, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50, 60, 70, 80]

    noise_strengths = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7,
                       0.8]  # I could have used a for loop etc.
    noises = ['0.1', '0.15', '0.2', '0.25', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
    fig_two = plt.figure(3, figsize=(20, 40))
    oneB = fig_two.add_subplot(221)
    asymB = fig_two.add_subplot(222)

    muTournPlot, oneBTournPlot, asymTournPlot = plotTournSelEAMeans()
    oneBitSizePlot = plotOneBitTournSelEAMeansSize()
    oneB.set_title("One Bit Tournament Selection EA")
    oneB.set_xticklabels(noises)
    asymB.set_xticklabels(noises)
    asymB.set_title("Asym-One Bit Tourn Sel EA")
    second_plot = oneB.boxplot(oneBitSizePlot)
    third_plot = asymB.boxplot(asymTournPlot)
    fig_two.show()
    plt.show()

    mediansMuTournPlot = [np.median(data) for data in muTournPlot]
    mediansOneBitTournPlot = [np.median(data) for data in oneBitTournPlot]

    fig = plt.figure(2, figsize=(20, 40))
    ax = fig.add_subplot(331)
    ax.set_title('Runtime of Tournament Selection EA ')
    bx = fig.add_subplot(332)

    dx = fig.add_subplot(333)
    bx.set_title('One-Bit Noise Tournament Selection EA')
    muTPlot = ax.boxplot(muTournPlot, showmeans=True)
    oneBTPlot = bx.boxplot(oneBitTournPlot, showmeans=True)
    print(mediansOneBitTournPlot)
    popt, pcov = curve_fit(thirdPolyModel, str_sizes, mediansOneBitTournPlot, bounds=(0, [50., 4.]))
    print(popt)
    print(thirdPolyModel(str_sizes, *popt))
    dx.plot(str_sizes, thirdPolyModel(str_sizes, *popt), 'g--')
    dx.plot(str_sizes, mediansOneBitTournPlot)

    ax.set_xticklabels(str_sizes)
    bx.set_xticklabels(str_sizes)

    ax.set_xlabel('String Length')
    bx.set_xlabel('String Length')

    ax.set_ylabel('Fitness Iterations')
    bx.set_ylabel('Fitness Iterations')

    fig.show()
    plt.show()


# Create a method with mutation probability 0.07 and show that it is expo





# plotMuCommaLambdaMeansAll()
plotTournSelAll()
