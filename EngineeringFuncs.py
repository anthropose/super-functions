# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:15:29 2019

@author: chari
"""

#%%
# import libraries
import numpy as np
import pandas as pd


#%%
# function to find absolute differences between two columns of a dataframe and
# create a new column; does not drop old columns
def absolute_difference(df, col1, col2):
    for name1, name2 in zip(col1, col2):
        diff = abs(df[name1] - df[name2])
        df["Diff_" + name1 + name2] = diff
    return df


#def absolute_error(df, col1, col2):
#    absolute_difference = []    
#    for name1, name2 in zip(col1, col2):
#        arr1 = np.array(df(name1))
#        arr2 = np.array(df(name2))
#        absolute_difference.append(arr1 - arr2)
#    return absolute_difference

#arr1 = np.array(comps["DMD_01_2018"])
#arr2 = np.array(comps["FCST_01_2018"])
#arr3 = (arr1 - arr2)

#%%
# function to create lagged features
# https://stackoverflow.com/questions/20410312/how-to-create-a-lagged-data-structure-using-pandas-dataframe
def buildLaggedFeatures(df, lag = 2, dropna = True):
    """
    Builds a new DataFrame to facilitate regressing over all possible lagged features
    """
    if type(df) is pd.DataFrame:
        new_dict = {}
        for col_name in df:
            new_dict[col_name] = df[col_name]
            # create lagged Series
            for num in range(1, lag+1):
                new_dict['%s_lag%d' %(col_name, num)] = df[col_name].shift(num)
        res = pd.DataFrame(new_dict, index = df.index)
    
    elif type(df) is pd.Series:
        the_range = range(lag+1)
        res = pd.concat([df.shift(i) for i in the_range], axis = 1)
        res.columns = ['lag_%d' %i for i in the_range]
    else:
        print('Only works for DataFrame or Series')
        return None
    if dropna:
        return res.dropna()
    else:
        return res