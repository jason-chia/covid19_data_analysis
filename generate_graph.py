import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plot
from matplotlib import pylab as pylab

def display_daily_num(cohort):

    # separate the cohort dataset into three datasets, one with 'Death No', one with 'Death Yes', one with 'Death Unknown'
    v1 = cohort[cohort['death']==0].episode_days
    v2 = cohort[cohort['death']==1].episode_days
    v3 = cohort[cohort['death']==9].episode_days
    v4 = np.concatenate((v1,v2,v3))
    
    # generate a histgram of the daily number of new cases, stratified by their death status as of May 8, 2020
    print('\n')
    plot.figure(figsize=(15, 10))
    plot.xticks(size=15)
    plot.yticks(size=15)
    plot.hist([v1, v2, v3], color = ['lightgrey', 'navajowhite', 'lightsteelblue'], histtype='barstacked', label = ('Death No', 'Death Yes', 'Death Unknown'))
    plot.grid(axis='y', alpha=0.75)
    plot.title('Daily Number of New COVID-19 Confirmed Cases in Canada', size=20)
    plot.legend(loc="upper left", prop={'size': 20})
    plot.xlabel('Number of Days Since Jan 15, 2020', size=20)
    plot.ylabel('Number of Cases', size=20)
    plot.show()

def display_daily_density(cohort):
    # separate the cohort dataset into three datasets, one with 'Death No', one with 'Death Yes', one with 'Death Unknown'
    v1 = cohort[cohort['death']==0].episode_days
    v2 = cohort[cohort['death']==1].episode_days
    v3 = cohort[cohort['death']==9].episode_days
    v4 = np.concatenate((v1,v2,v3))
    
    # generate histogram of the density and overlay a curve on top of the density using the seaborn library
    print('\n')
    plot.figure(figsize=(15, 10))
    plot.xticks(size=15)
    plot.yticks(size=15)
    plot.hist([v1, v2, v3], color = ['lightgrey', 'navajowhite', 'lightsteelblue'], histtype='barstacked', density = True, label = ('Death No', 'Death Yes', 'Death Unknown'))
    sns.kdeplot(v4, color='r')
    plot.grid(axis='y', alpha=0.75)
    plot.title('Density of Daily New COVID-19 Confirmed Cases', size=20)
    plot.legend(loc="upper left", prop={'size': 20})
    plot.xlabel('Number of Days Since Jan 15, 2020', size=20)
    plot.ylabel('Density', size=20)
    plot.show()

def display_cumulative_num(cohort):
    # separate the cohort dataset into three datasets, one with 'Death No', one with 'Death Yes', one with 'Death Unknown'
    v1 = cohort[cohort['death']==0].episode_days
    v2 = cohort[cohort['death']==1].episode_days
    v3 = cohort[cohort['death']==9].episode_days
    v4 = np.concatenate((v1,v2,v3))
    
    # generate a histgram of the cumulative number of cases, stratified by their death status as of May 8, 2020
    print('\n')
    plot.figure(figsize=(15, 10))
    plot.xticks(size=15)
    plot.yticks(size=15)
    plot.hist([v1, v2, v3], color = ['lightgrey', 'navajowhite', 'lightsteelblue'], histtype='barstacked', cumulative = True, label = ('Death No', 'Death Yes', 'Death Unknown'))
    plot.grid(axis='y', alpha=0.75)
    plot.title('Cumulative Number of COVID-19 Confirmed Cases in Canada', size=20)
    plot.legend(loc="upper left", prop={'size': 20})
    plot.xlabel('Number of Days Since Jan 15, 2020', size=20)
    plot.ylabel('Number of Cases', size=20)
    plot.show()

def display_cumulative_density(cohort):
    # separate the cohort dataset into three datasets, one with 'Death No', one with 'Death Yes', one with 'Death Unknown'
    v1 = cohort[cohort['death']==0].episode_days
    v2 = cohort[cohort['death']==1].episode_days
    v3 = cohort[cohort['death']==9].episode_days
    v4 = np.concatenate((v1,v2,v3))
    
    # generate histogram of the cumulative density and overlay a curve on top of the cumulative density using the seaborn library
    print('\n')
    plot.figure(figsize=(15, 10))
    plot.xticks(size=15)
    plot.yticks(size=15)
    plot.hist([v1,v2,v3],color = ['lightgrey', 'navajowhite', 'lightsteelblue'], histtype='barstacked',cumulative = True,density=True,label = ('Death No','Death Yes','Death Unknown'))
    sns.kdeplot(v4, cumulative=True, color='r')
    plot.grid(axis='y', alpha=0.75)
    plot.title('Cumulative Density of COVID-19 Confirmed Cases', size=20)
    plot.legend(loc="upper left", prop={'size': 20})
    plot.xlabel('Number of Days Since Jan 15, 2020', size=20)
    plot.ylabel('Cumulative Density', size=20)
    plot.show()

def display_sigmoid(parameters):
    # plot the Sigmoid function using the parameter coefficient estimate from a logistic regression model
    print('\n')
    a = pylab.linspace(0,160,160)
    b1 = 1 / (1 + np.exp(-(parameters.values[0][1]+ a*parameters.values[1][1])))
    b2 = 1 / (1 + np.exp(-(parameters.values[0][1]+ a*parameters.values[1][1]+parameters.values[2][1])))
    pylab.rcParams['figure.figsize'] = 15, 10
    pylab.xticks(size=15)
    pylab.yticks(size=15)
    pylab.plot(a, b1, label = 'Community Exposure')
    pylab.plot(a, b2, label = 'Travel Exposure')
    pylab.xlabel('Approximated Age in Years', fontsize=20)
    pylab.ylabel('Probability of COVID-19 Death', fontsize=20)
    pylab.legend(loc='upper left', prop={'size': 20})
    pylab.text(50, 0.75, r'$\sigma(x)=\frac{1}{1+e^{-x}}$', fontsize=25)
    pylab.grid()
    pylab.title('Sigmoid Function', fontsize=20)
    pylab.show()