# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:15:29 2019

@author: chari
"""

#%%
# import libraries
import numpy as np
#import pandas as pd

#%%
# function to find the forecast error between two columns of a dataframe and
# create a new column; does not drop old columns
# the difference between the realized value and the predicted value
def prediction_error(df, col1, col2):
    for name1, name2 in zip(col1, col2):
        diff = df[name1] - df[name2]
        df["Error_" + name1 + name2] = diff
        
    return df


#%%
# function to determine whether the forecast error for a particular column was
# beyond or equal to a user-defined threshold
def prediction_error_threshold(df, error_cols, threshold = 0, direction = "less"):
    for name in error_cols:
        if direction == "less":
            df[name + "<" + str(threshold)] = np.where(df[name] < 0, 1, 0)
        elif direction == "more":
            df[name + ">" + str(threshold)] = np.where(df[name] > 0, 1, 0)
        elif direction == "exact":
            df[name + "=" + str(threshold)] = np.where(df[name] == 0, 1, 0)
            
    return df
    

#%%
# function to get row average of a user-defined span for specific columns
def average_span(df, span_cols, span = 3):    
    temp_df = df[span_cols]
    new_df = temp_df.groupby(np.arange(len(temp_df.columns))//span, axis = 1).mean()
    
    return new_df     


#%%
# function to create a flag variable for the presence of missing values for
# specific columns of a dataframe
def is_missing_flag(df, missing_cols):
    for name in missing_cols:
        df["Is_Missing_" + name] = np.where(df[name].isnull(), 1, 0)
    
    return df


#%%
## function to create lagged features
## https://stackoverflow.com/questions/20410312/how-to-create-a-lagged-data-structure-using-pandas-dataframe
#def buildLaggedFeatures(df, lag = 2, dropna = True):
#    """
#    Builds a new DataFrame to facilitate regressing over all possible lagged features
#    """
#    if type(df) is pd.DataFrame:
#        new_dict = {}
#        for col_name in df:
#            new_dict[col_name] = df[col_name]
#            # create lagged Series
#            for num in range(1, lag+1):
#                new_dict['%s_lag%d' %(col_name, num)] = df[col_name].shift(num, axis = 0)
#        res = pd.DataFrame(new_dict, index = df.index)
#    
#    elif type(df) is pd.Series:
#        the_range = range(lag+1)
#        res = pd.concat([df.shift(i) for i in the_range], axis = 1)
#        res.columns = ['lag_%d' %i for i in the_range]
#    else:
#        print('Only works for DataFrame or Series')
#        return None
#    if dropna:
#        return res.dropna()
#    else:
#        return res
#
#
##%%
## build lagged features for desired list of columns in a dataframe
#def lag_features_col(df, lag_cols, lag = 2, dropna = True):
#    """
#    Builds a new DataFrame to facilitate regressing over all possible lagged features
#    """
#    if type(df) is pd.DataFrame:
#        new_dict = {}
#        for name in lag_cols:
#            # create lagged Series
#            for num in range(1, lag+1):
#                new_dict['%s_lag%d' %(name, num)] = df[name].shift(num)
#        res = pd.DataFrame(new_dict, index = df.index)
#    
#    elif type(df) is pd.Series:
#        the_range = range(lag+1)
#        res = pd.concat([df.shift(i) for i in the_range], axis = 1)
#        res.columns = ['lag_%d' %i for i in the_range]
#    else:
#        print('Only works for DataFrame or Series')
#        return None
#    if dropna:
#        return res.dropna()
#    else:
#        return res


#%%
#def lag(df, periods = 2):
#    
#test = pd.DataFrame(comps.iloc[23]).shift(periods = 2, axis = 0)        
        
        