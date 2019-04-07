# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:34:11 2019

@author: chari
"""

#%%
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


#%%
# function to drop unnecessary features
def drop_preprocessing(df, drop_cols, axis = 1):
    for name in drop_cols:
        df.drop(name, axis = axis, inplace = True)
        
    return df


#%%
# function to clip values at a user-defined threshold
def clip_preprocessing(df, clip_cols, threshold = 0):
    for name in clip_cols:
        df[name] = df[name].clip(lower = threshold)
    
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
# function to min-max scale numeric features to be between 0 and 1
def zero_one_scale(df, scale_cols):
    scaled_cols = {}
    for col in scale_cols:
        scaler = MinMaxScaler(feature_range = (0,1))
        df[col + "_scaled"] = pd.DataFrame(scaler.fit_transform(df[[col]]), index = df.index)
        df.drop(col, axis = 1, inplace = True)
        scaled_cols[col] = scaler
    
    return scaled_cols


#%%
# function to z-score transform numeric features
def zscore(df, z_cols, with_mean = True, with_std = True):
    standardized_cols = {}
    for col in z_cols:
        z_standardizer = StandardScaler(with_mean = with_mean, with_std = with_std)
        df[col + "_z"] = pd.DataFrame(z_standardizer.fit_transform(df[[col]]), index = df.index)
        df.drop(col, axis = 1, inplace = True)
        standardized_cols[col] = z_standardizer
    
    return standardized_cols


#%%
# function to generate and plot cumulative sums of explained variance using PCA 
# to determine the best number of components
def pca_cumsum_plot(df, n_keep = "mle", solver = "full", seed = 42, name = "Plot"):
    pca = PCA(n_components = n_keep, svd_solver = solver, random_state = seed)
    pca.fit_transform(df)
    df.name = name
    
    var = pca.explained_variance_ratio_
    cum_var = np.cumsum(np.round(var, decimals = 4) * 100)
    plt.plot(cum_var)
    plt.title("PCs for %s" % name)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Percent Variance Explained")
    plt.show()    
    

#%%
# function to normalize the rows of a dataframe; allows the user to remove
# unnecessary columns prior to normalization and concat them back in
#def normalize(df, norm_cols):
#    normalized_cols = {}
#    for col in norm_cols:
#        norm = Normalizer(norm = "l2")
#        df[col + "_norm"] = pd.DataFrame(norm.fit_transform(df[[col]]), index = df.index)
#        df.drop(col, axis = 1, inplace = True)
#        normalized_cols[col] = norm
#    
#    return normalized_cols
#
#norm = Normalizer(norm='l2')
#dfnorm= pd.DataFrame(norm.fit_transform(data131_ASL[norm_cols]), 
#                      columns=['norm_'+x for x in


def normalize(df, norm_cols, norm_type = "l2"):
    df_temp = df[norm_cols]
    df.drop(norm_cols, axis = 1, inplace = True)
    norm = Normalizer(norm = norm_type)
    df_norm = pd.DataFrame(norm.fit_transform(df_temp[norm_cols]), 
                           columns = [x + '_norm' for x in df_temp.columns], 
                           index = df_temp.index)
    new_df = pd.concat((df, df_norm), axis = 1, join = "outer")
    
    return new_df


#%%
# function to get dummy/indicator variables for specific categorical variables
#def dummy(df, dummy_cols, prefix = None):
#    for col in dummy_cols:
#        pd.get_dummies()
#test = pd.get_dummies(data = data131_y["ABC"], prefix = "ABC", sparse = False, dtype = int)

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
#def cat2label_preprocessor(df, label_cols):
#    label_encoders = {}
#    for col in label_cols:
#        label_encoder = LabelEncoder()
#        df[col + "_labeled"] = pd.DataFrame(label_encoder.fit_transform(df[[col]]), index = df.index)
#        df.drop(col, axis = 1, inplace = True)
#        label_encoders[col] = label_encoder
#    
#    return label_encoders


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
