# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 11:07:03 2019

@author: chari
"""

#%%
import numpy as np
import scipy.stats as st
import pandas as pd
   

#%%
# function for generating data from a scipy histogram

class GeneratedData:
    def __init__(self, bins, edges, histogram):
        """

        """
        self.Bins = bins
        self.Edges = edges
        self.Histogram = histogram
        
    def Generate(self, size = 10000):
        return self.Histogram.rvs(size = size)


# compute the histogram of observed data using numpy
def hist_compute(data, bins = -1):
    """

    """
    # initialize empty dictionary to hold computed bin counts and bin edges of
    # for each column of df
    hist_computation = {}
    
    if bins < 0:
        bins = len(data)

    # compute histograms for each column in df
    for col in data.columns.values:
        counts, edges = np.histogram(data[col], bins = bins, density = False)
        hist_computation[col] = GeneratedData(counts, edges, st.rv_histogram((counts, edges)))
        
    return hist_computation


#%%
# create df of newly generated data
def generate_df(data, bins = -1, size = 10000):
    """
    """
    new_df = pd.DataFrame()
    
    hist_dict = hist_compute(data, bins)
    for col in data.columns.values:
        new_df[col] = hist_dict[col].Generate(size)
    
    return new_df











































