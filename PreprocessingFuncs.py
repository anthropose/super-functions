# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:34:11 2019

@author: chari
"""

#%%
# import libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OrdinalEncoder


#%%
# function to drop unnecessary features
def drop_preprocessing(df, drop_cols):
    for name in drop_cols:
        df.drop(name, axis = 1, inplace = True)
        
    return df


#%%
# function to convert all date-time features into date-time objects
def datetime_preprocessing(df, date_time_cols):
    for name in date_time_cols:
        df[name] = pd.to_datetime(df[name]) 
    # parse all date-time features into new features for month, day, hour
        df[name + "_month"] = df[name].apply(lambda x: x.month)
        df[name + "_day"] = df[name].apply(lambda x: x.day)
        df[name + "_hour"] = df[name].apply(lambda x: x.hour) 
    # drop date-time objects
        df.drop(name, axis = 1, inplace = True)
        
    return df


#%%
# function to convert a time feature into a time object and get total seconds
def time_preprocessing(df, time_cols):
    for name in time_cols:
        df[name] = pd.to_datetime(
                df[name], errors = "raise", format = "%H:%M:%S").dt.time
    # calculate total seconds into new feature
        df[name + "_totalsec"] = df[name].apply(
                lambda x: x.hour * 3600 + x.minute * 60 + x.second)
    # drop time object
        df.drop(name, axis = 1, inplace = True)
        
    return df


#%%
# function to re-format standard US currencies to a float
def currency_preprocessing(df, currency_cols):
    for name in currency_cols:
        df[name] = df[name].str.replace("$", "")
        df[name] = df[name].str.replace(",", "")
        df[name] = df[name].astype("float64")
        
    return df



#%%
# function to take the average of two columns
def avg_by_cols(row, columns):
    total = 0 
    for name in columns:
        total = total + row[name]
    
    avg = total / len(columns)
    
    return avg


#%%
# function to deal with correlated numeric features
def correlated_preprocessing(df, corr_cols):
    separator = '_'    
    for cols in corr_cols:
        df[separator.join(cols) + "_avg"] = df.apply(lambda row: avg_by_cols(row, cols), axis = 1)
        df.drop(cols, axis = 1, inplace = True)
        
    return df
    

#%%
# function to fill missing values with a constant
def fill_missing_constant_preprocessor(df, missing_cols, fill_value = -9999):
    si = SimpleImputer(missing_values = np.nan, 
                       strategy = "constant", 
                       fill_value = fill_value)
    for col in missing_cols:
        df[col] = pd.DataFrame(si.fit_transform(df[[col]]), index = df.index)
        
    return si


#%%
# function to convert numeric features to kbinsdiscretized features
def numeric2kbins_preprocessor(df, continuous_cols, strategy = "quantile", nbins = 4):
    if strategy == "quantile":
        strategy = "quantile"
        kbd_norm_cols = {} 
        for col in continuous_cols:
            kbd_norm = KBinsDiscretizer(n_bins = nbins, 
                                        encode = "ordinal", 
                                        strategy = strategy)
            df[col + "_kbins"] = pd.DataFrame(kbd_norm.fit_transform(df[[col]]), index = df.index)
            df.drop(col, axis = 1, inplace = True)
            kbd_norm_cols[col] = kbd_norm
        return kbd_norm_cols

    elif strategy == "uniform":
        strategy = "uniform"
        kbd_uniform_cols = {} 
        for col in continuous_cols:    
            kbd_uniform = KBinsDiscretizer(n_bins = nbins, encode = "ordinal", strategy = strategy)
            df[col + "_kbins"] = pd.DataFrame(kbd_uniform.fit_transform(df[[col]]), index = df.index)
            df.drop(col, axis = 1, inplace = True)
            kbd_uniform_cols[col] = kbd_uniform
    
        return kbd_uniform_cols    


#%%
def numeric2binary_preprocessor(df, binary_cols, threshold = 0.0):  
    binarized_cols = {} 
    for col in binary_cols:
        binarizer = Binarizer(threshold = threshold, copy = False)
        df[col + "_binary"] = pd.DataFrame(binarizer.fit_transform(df[[col]]), index = df.index)
        df.drop(col, axis = 1, inplace = True)
        binarized_cols[col] = binarizer
    
    return binarized_cols


#%%
def cat2ordinal_preprocessor(df, ordinal_cols):
    ordinal_encoders = {}
    for col in ordinal_cols:
        ordinal_encoder = OrdinalEncoder(categories = "auto", dtype = int)
        df[col + "_ordinal"] = pd.DataFrame(ordinal_encoder.fit_transform(df[[col]]), index = df.index)
        df.drop(col, axis = 1, inplace = True)
        ordinal_encoders[col] = ordinal_encoder
    
    return ordinal_encoders


#%%
def fill_missing_constant_transformer(df, missing_cols, transformer):    
    for col in missing_cols:
        df[col] = pd.DataFrame(transformer.transform(df[[col]]), index = df.index)


#%%
def numeric2kbins_transformer(df, continuous_cols, transformer):     
    for col in continuous_cols:
        df[col + "_kbins"] = pd.DataFrame(transformer[col].transform(df[[col]]), index = df.index)
        df.drop(col, axis = 1, inplace = True)


#%%
def numeric2binary_transformer(df, binary_cols, transformer):
    for col in binary_cols:
        df[col + "_binary"] = pd.DataFrame(transformer[col].transform(df[[col]]), index = df.index)
        df.drop(col, axis = 1, inplace = True)


#%%
def cat2ordinal_transformer(df, ordinal_cols, transformer):
     for col in ordinal_cols:
        df[col + "_ordinal"] = pd.DataFrame(transformer[col].transform(df[[col]]), index = df.index)
        df.drop(col, axis = 1, inplace = True)


#%%
def get_column_names_from_ColumnTransformer(column_transformer):    
    col_name = []
    for transformer_in_columns in column_transformer.transformers_[:-1]:#the last transformer is ColumnTransformer's 'remainder'
        raw_col_name = transformer_in_columns[2]
        if isinstance(transformer_in_columns[1], Pipeline): 
            transformer = transformer_in_columns[1].steps[-1][1]
        else:
            transformer = transformer_in_columns[1]
        try:
            names = transformer.get_feature_names()
        except AttributeError: # if no 'get_feature_names' function, use raw column name
            names = raw_col_name
        if isinstance(names,np.ndarray): # eg.
            col_name += names.tolist()
        elif isinstance(names,list):
            col_name += names    
        elif isinstance(names,str):
            col_name.append(names)
    return col_name
