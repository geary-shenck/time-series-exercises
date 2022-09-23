#### Import Section
from lib2to3.refactor import get_all_fix_names
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import requests
import acquire
import os

from itertools import product

from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFE, f_regression, SelectKBest
import statsmodels.api as sm



    #train, validate, test, y_train, y_validate, y_test = explore_split_time_series(df,target)
    #explore_target_mean_time_series(y_train,target)
    #explore_target_diff_time_series(y_train,target)
    #explore_target_weekly_time_series(train,target)
    #explore_target_bin_compare_time_series(train,target,bin="D",scale="M",year=str(i))



def explore_split_time_series(df,target="",train_size=.6,validate_size=.2):
    ''' 
    input dataframe, target(as string), and train size(as float)
    splits based on size, seperates out into y(series) and train df (includes target)
    plots a few examples
    returns train, test,y_train,y_test
    '''
    n = df.shape[0]
    test_start_index = round((train_size+validate_size) * n)
    validate_start_index = round((train_size) * n)

    train = df[:validate_start_index] # everything up (not including) to the test_start_index
    validate = df[validate_start_index:test_start_index]
    test = df[test_start_index:] # everything from the test_start_index to the end

    y_train = train[target]
    print(train.shape,"train shape")

    y_validate = validate[target]
    print(validate.shape,"validate shape")
    
    y_test = test[target]
    print(test.shape,"test shape")

    plt.plot(train.index, train[target])
    plt.plot(validate.index, validate[target])
    plt.plot(test.index, test[target])
    plt.title(f"{target} over time")
    plt.ylabel(f"{target}")
    plt.xlabel("Time")
    plt.show()

    y_train.plot.hist()
    plt.title(f"{target} distribution (train)")
    plt.xlabel(f"{target}")
    plt.ylabel("Count")
    plt.show()

    return train, validate, test, y_train, y_validate, y_test

def explore_target_mean_time_series(y_train,target):
    ''' 
    
    '''
    y_train.resample('D').mean().plot(alpha=.5, label='Daily')
    y_train.resample('W').mean().plot(alpha=.8, label='Weekly')
    y_train.resample('M').mean().plot(label='Montly')
    y_train.resample('Y').mean().plot(label='Yearly')
    plt.title(f"{target} over time")
    plt.legend()
    plt.show()


    y_train.groupby(y_train.index.day_name()).mean().\
        sort_values().plot.bar(width=.9, 
                               ec='black',
                               title=f'Average {target} by Weekday', 
                               xlabel='Weekday', 
                               ylabel=target)
    plt.xticks(rotation=0)

    ax = y_train.groupby(y_train.index.isocalendar().week).mean().plot.bar(width=.9, ec='black')
    plt.xticks(rotation=0)
    ax.set(title=f'Average {target} by Week', xlabel='Week', ylabel=target)
    plt.show()

    ax = y_train.groupby(y_train.index.month).mean().plot.bar(width=.9, ec='black')
    plt.xticks(rotation=0)
    ax.set(title=f'Average {target} by Month', xlabel='Month', ylabel=target)
    plt.show()

    ax = y_train.groupby(y_train.index.year).mean().plot.bar(width=.9, ec='black')
    plt.xticks(rotation=0)
    ax.set(title=f'Average {target} by Year', xlabel='Year', ylabel=target)
    plt.show()

    None

def explore_target_diff_time_series(y_train,target):
    ''' 
    input y_train and target
    prints out a few plots with diffent time spacing
    '''

    y_train.resample('W').mean().diff().plot(title=f'Average change in {target}',label="Week to Week")
    y_train.resample('M').mean().diff().plot(title=f'Average change in {target}',label="Month to Month")
    y_train.resample('Y').mean().diff().plot(title=f'Average change in {target}',label="Year to Year")
    plt.xlabel("Time")
    plt.ylabel(f"{target}")
    plt.legend()

    y_train.groupby([y_train.index.year, y_train.index.month]).mean().unstack(0).plot(title='Seasonal Plot',ylabel=f"{target}")
    table = y_train.groupby([y_train.index.year, y_train.index.month]).mean().unstack()

    fig, axs = plt.subplots(1, 12, sharey=True, sharex=True)
    for ax, (month, subset) in zip(axs, table.iteritems()):
        subset.plot(ax=ax, title=month)
        ax.hlines(subset.mean(), *ax.get_xlim())
        ax.set(xlabel='Date',ylabel=f"{target}")

    fig.suptitle('Seasonal Subseries Plot') # super-title for the overall figure
    fig.subplots_adjust(wspace=0)
    plt.show()
    None

def explore_target_weekly_time_series(train,target,corr_shift=-1,):
    ''' 
    input train dataframe(needs index as datetime) and target(as string), additional arguements optional
    makes a correlation plot, correlation lag plot, and two decomp plots
    creates a decomp dataframe
    returns the decomp df
    '''
    #makes a correlation with a shift of -1 unless otherwise specified
    weekly = train.resample('W').mean()
    weekly[f'Week {corr_shift*-1} Difference'] = weekly[target].shift(corr_shift)
    weekly = weekly.rename(columns={target: 'This Week'})
    weekly.plot.scatter(x='This Week', y=f'Week {corr_shift*-1} Difference')
    plt.title(f"This week compared with {corr_shift*-1} week Difference for {target}")
    plt.show()

    # plots correlation over changing lag
    pd.plotting.autocorrelation_plot(train[target].resample('W').mean())
    plt.title(f"Correlation of {target} over different Weekly Lags")
    plt.show()

    #creates dataframe using seasonal decompose to produce trend season and residual plots
    y_train_temp = train[target].resample('W').mean()
    result = sm.tsa.seasonal_decompose(y_train_temp)
    decomposition = pd.DataFrame({'y': result.observed,
                                'trend': result.trend,
                                'seasonal': result.seasonal,
                                'resid': result.resid,})
    #plots the results
    result.plot()
    plt.xlabel("Time")
    plt.ylabel(f"{target}")
    plt.show()

    decomposition.iloc[:, 1:].plot()
    plt.ylabel(f"{target} Compared")
    plt.title("Comparison - Trends (aggregated)")
    plt.show()

    decomposition.head()
    return decomposition

def explore_target_bin_compare_time_series(train,target,bin="D",scale="M",year="2015"):
   ''' 
   input train(index is datetime), 
           target (string), 
           bin(string-resample bin for integrals), 
           scale(string-resample bin for x axis)
           year(string-to examine further)
   creates a df, and cuts based on quartiles, regroups and resamples
   '''

   train_temp = train.resample(bin).mean()
   # create a categorical feature
   train_temp[f'{target}_bin'] = pd.qcut(train_temp[target], 4, labels=['lowest', 'low', 'high', 'highest'])
   train_temp.groupby(f'{target}_bin').mean()
     
   (train_temp.groupby(f'{target}_bin').resample(scale).size().unstack(0).apply(lambda row: row / row.sum(), axis=1).plot.area())
   plt.ylabel(f'% of {bin} in the {scale}')
   plt.title(f"Distribution of {target} by {bin} over Time")
   plt.show()

   if train_temp.loc[year].groupby(f'{target}_bin').resample(scale).size().shape[0] == train_temp[f'{target}_bin'].nunique():
      ax = (train_temp.loc[year].groupby(f'{target}_bin').resample(scale).size().T.plot.bar(stacked=True, width=.9, ec='black'))
      labels = [pd.to_datetime(t.get_text()).strftime('%B') for t in ax.get_xticklabels()]
      ax.set_xticklabels(labels)
      ax.set(ylabel=f"{target}",title=f"Mean {target} in {year} by {scale}")
   else:
      ax = (train_temp.loc[year].groupby(f'{target}_bin').resample(scale).size().unstack(0).plot.bar(stacked=True, width=.9, ec='black'))
      labels = [pd.to_datetime(t.get_text()).strftime('%B') for t in ax.get_xticklabels()]
      ax.set_xticklabels(labels)
      ax.set(ylabel=f"{target}",title=f"Mean {target} in {year} by {scale}")

   plt.show()
   ax = train[target].groupby(train[target].index.strftime('%m-%b')).mean().plot.bar(title=f"{target} by Month (mean)")
   ax.set_xticklabels([t.get_text()[3:] for t in ax.get_xticklabels()], rotation=0)
   plt.show()
   None