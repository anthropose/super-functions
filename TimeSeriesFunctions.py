# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 10:41:50 2019

@author: chari
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import preprocessing

#%%
# generate a time series line plot of each column of a dataframe
def timeseriesplotbycol(data, 
                        time_labels = [], 
                        xlabel = "Time", 
                        ylabel = "Observed Values"):
    """
    Create time series line plots for each column of a dataframe with time 
    along the x-axis and values from desired feature along the y-axis. Requires 
    user to provide a Pandas dataframe. Default x- and y-axis labels are
    provided and can be overwritten. Plot titles take the name of the dataframe
    column.
    """
    time_label = data.index.values
    
    if len(time_labels) > 0:
        time_label = time_labels
    
    # for each column in df, generate a line plot with time on x-axis
    for col in data.columns.values:
        
        # use length of df as units of time
        x = range(len(data))
        
        y = data[col].values
        
        plt.plot(x, y)
        plt.xlabel("Time")
        plt.xticks(ticks = x, labels = time_label, rotation = 20)
        plt.ylabel("Demand")
        plt.title(col)
        plt.show()


#%%
# generate a single time series line plot of all columns of a dataframe on the
# same plot
# requires that all data in the df be normalized to the same scale [0,1]

def timeseriesplotall(data,
                      time_labels = [],
                      xlabel = "Time", 
                      ylabel = "Observed Values", 
                      title = "Time Series",
                      figsize = (12, 6)):
    """
    Creates time series line plot for all columns of a dataframe on the same
    plot, with time along the x-axis and values from desired feature along the 
    y-axis. Requires users to provide a Pandas dataframe. Default x- and y-axis
    labels are provided and can be overwritten. Default plot title takes the 
    name of the df and can be overwritten. Each line plotted takes the name of 
    the df column.
    """
    
    time_label = data.index.values
    
    if len(time_labels) > 0:
        time_label = time_labels
    
    # normalize all data in dataframe
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    data_normalized = pd.DataFrame(np_scaled, 
                                   index = data.index, 
                                   columns = data.columns)
    
    # use length of df as units of time
    x = range(len(data))
    
    # plot each column as a line on figure
    for col in data_normalized.columns.values:
        col_name = data_normalized[col].values
        
        plt.figure(num = 1, figsize = figsize)
        plt.plot(x, col_name, label = col)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.xticks(ticks = x, labels = time_label, rotation = 20)
        plt.ylabel(ylabel)
        plt.legend(loc = "best", bbox_to_anchor = (1.04, 1))
            
    plt.show()
    
    
#%%
# decompose a time series based on cluster IDs; this could also be used for other
# user defined columns
def ts_decomposition_cluster(df):
    """
    Decomposes a time series in the form of a pandas dataframe. Note: index of
    the user-provided dataframe must be in a DateTime dtype. Returns a 
    dictionary of the decompositions. To access the individual values for each
    time unit simply call it in the following manner: 
    dict["Cluster_Number"].seasonal, dict["Cluster_Number"].trend
    """
    clusterID = list(df.columns.values)
    decompositions = {}
    for cluster in clusterID:
        result = sm.tsa.seasonal_decompose(df[cluster], model = "additive", freq = 11)
        decompositions[cluster] = result
        result.plot()
        plt.suptitle(cluster)
        plt.show()
    
    return decompositions
    
    
#%%
# save data from decomposition dictionary into df
def decomposition_dict2df(dictionary, attribute = "seasonal"):
    keys = list(dictionary.keys())
    decomposition = pd.DataFrame()
    
    if attribute == "seasonal":
        for key in keys:
            decomposition[key] = dictionary[key].seasonal
    elif attribute == "trend":
        for key in keys:
            decomposition[key] = dictionary[key].trend
            
    return decomposition
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    