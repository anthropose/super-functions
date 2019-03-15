# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 13:02:02 2019

@author: chari
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

#%%
# my defaults
# set color palette for Seaborn
print("Setting Seaborn pallet color defaults")
sns.set_palette("deep")
sns.set(color_codes=True)

# set figure display settings for Seaborn
sns.set_context("paper", rc = {"font.size":8, "axes.titlesize":10, "axes.labelsize":8})
print("Setting Seaborn display default to [paper]")

# Console display settings
max_col = 50
print("Setting Pandas display.max_columns defaults to [%d]" % max_col)
pd.set_option("display.max_columns", max_col)

#%%
# load data function
def loadDF(file_path, ftype, resetmax = False):
    """
    This function 1) creates a file path in the current working directory
    to a csv or Excel file, 2) loads the file as Pandas dataframe, and 3) has 
    an option to reset the Pandas column display settings to the length of the 
    imported data frame. Required arguments are: 1) name of csv or Excel file, 
    and 2) type of file (i.e., csv or excel).
    """
    # initialize empty Pandas dataframe
    df = pd.DataFrame()
    
    # create file path only if user enters a string
    if type(file_path) is str:
        # create a file path only if user enters a string file type (i.e, csv
        # or Excel)
        if type(ftype) is str:
            fpath = os.path.join(os.getcwd(), file_path)
            
            if(ftype == "csv"):
                df = pd.read_csv(fpath)
            elif (ftype == "excel"):
                df = pd.read_excel(fpath)      
            else:
                print("Invalid file type [%s]" % ftype)
        else:
            print("Invalid file type. Expects string (i.e., csv or excel)")
    else:
        print("File path expects a string")
    
    # reset console display settings to length of newly-created data frame; 
    # optional argument
    if resetmax:
        col_len = len(df.columns)
        if pd.get_option("display.max_columns") <= col_len:
            print("Setting new max columns for Pandas to [%d] columns" 
                  % col_len)
            pd.set_option("display.max_columns", col_len)
    
    # return newly-created dataframe    
    return df


#%%
# quick view function
def QView(df):
    """
    This function generates quick views on the shape and information of a 
    Pandas dataframe. It also saves the column names to a new object for later
    use. Required argument is: name of Pandas dataframe.
    """
    if type(df) is pd.DataFrame:
        print("Shape of DataFrame")
        print(df.shape)
        print()
        print()
        print("Information on DataFrame")
        print()
        print(df.info())
        print()
        print()
        print("Head of DataFrame")
        print()
        print(df.head())
        print()
        
    else:
        print("Not a Pandas dataframe")

#%%
# remove NaNs function
def elimNan(df):
    """
    This function removes NaNs from a dataframe so it can be called inside 
    other functions. For example, many Seaborn plotting functions do not handle
    NaNs. This function removes NaNs row-wise for any NaN that is present and 
    returns a NEW dataframe.
    """
    # initialize a new Pandas dataframe
    new_df = pd.DataFrame()
    
    # create a new dataframe only if input is a Pandas dataframe
    if type(df) is pd.DataFrame:
        new_df = df.dropna()
        
    else:
        print("Not a Pandas dataframe")
    
    # return newly-created dataframe without NaNs    
    return new_df

#%%
# remove outliers function
def elimOuts(df):
    """
    This function removes outliers from a dataframe so it can be called inside 
    other functions. Used in EDA; not for pre-processing.
    """
    # initialize a new Pandas dataframe
    new_df = pd.DataFrame()
    
    # create a new dataframe only if input is a Pandas dataframe
    if type(df) is pd.DataFrame:
        # can only remove outliers for numeric features without NaNs
        num_df = elimNan(df.select_dtypes(include = "number"))
        new_df = num_df[(np.abs(stats.zscore(num_df)) < 3).all(axis = 1)]
        
    else:
        print("Not a Pandas dataframe")
    
    # return newly-created dataframe without outliers
    return new_df


#%%
# percentage of missing values for each feature
def percentMissing(df):
    """
    This function generates the percentage of missing values for each feature
    in a dataframe over the total number of rows, sorts the resulting
    percentages, and prints the resulting numpy array. Required argument is a
    Pandas dataframe
    """
    if type(df) is pd.DataFrame:
        # sum all missing values for each feature and save to new dataframe
        missing = df.isna().sum()
        
        # calculate the percentage of missing values
        per_missing = missing/len(df)
        
        # sort the percentages from highest to lowest
        sorted_missing = per_missing.sort_values(ascending = False)
        print(sorted_missing)
        
        return sorted_missing
        
    else:
        print("Not a Pandas dataframe")
        
    
#%%
# display plots to visualize the location or counts of missing values using
# Seaborn, Missingno, or pyplot
def visMissing(df, method):
    """
    This function creates plots to help visualize missing values and their
    relationship(s) between features. Required arguments are the following: 1)
    a Pandas dataframe, and 2) the desired plotting method entered as a string
    (i.e., seaborn, missingno, or pyplot). The pyplot method returns sorted
    results.
    """
    # display heatmaps only for Pandas dataframes
    if type(df) is pd.DataFrame:
        if type(method) is str:
            
            # Seaborn method
            if method == "seaborn":
                sns.heatmap(df.isnull(), cbar = False, yticklabels = False)
                plt.title("Missing Values Locations")
                plt.show()
            
            # Missingno method
            if method == "missingno":
                msno.matrix(df)
                plt.show()
            
            # pyplot method
            if method == "pyplot":
                # plot using sorted values and horizontal bar chart
                df.isnull().sum().sort_values(ascending = False).plot(
                        kind = "barh", 
                        title = "Missing Value Counts",
                        figsize = (10,10),
                        fontsize = 8)
                plt.show()
                
        else:
            print(
            "Method type requires string (i.e., seaborn, missingno, pyplot)")
    else:
        print("Not a Pandas dataframe")

#%%
# visualize correlation(s) of missing values using msno package
def corrMissing(df, method):
    """
    Assess how strongly the presence or absence of one variable affects the
    presence or absence of another. For heatmaps, nullity correlations range
    from -1 (if one variable appears and the other does not) to 0 (variables
    appearing or not have no effect on one another) to 1 (if one variable
    appears the other also appears)). Entries marked <1 or >-1 point to records 
    in the dataset which may be erroneous. For dendrograms, variables are binned
    against one another by their nullity correlation (measured in terms of 
    binary distance). Read the graph from top-down. Cluster leaves which are 
    linked together at a distance of zero fully predict one another's presence 
    - whether negatively or positively. Cluster leaves which split close to 
    zero, but not at it, predict one another well, but not perfectly. These 
    examples may indicate erroneous data, especially if those particular 
    columns actually are or ought to match each other perfectly in nullity. See 
    missingno documentation for more information.
    """
    if type(df) is pd.DataFrame:
        if type(method) is str:
            if method == "heatmap":
                msno.heatmap(df, labels = True, fontsize = 8, cmap = "copper",
                             figsize = (10,10))
                plt.title("Missing Values Correlations")
                plt.show()
                
            if method == "dendrogram":
                msno.dendrogram(df, orientation = "right", fontsize = 8, 
                                figsize = (10,10))
                plt.title("Missing Values Correlations")
                plt.show()
        else:
            print("Method type requires string (i.e., heatmap, dendrogram)")
    else:
        print("Not a Pandas dataframe")
    
        
#%%
# display Pandas plots functions for numeric features only
def Numplots(df, ptype):
    """
    This function creates EDA plots for Pandas dataframes. Required arguments
    are: 1) the name of the dataframe, and 2) a string input for the type of 
    plot desired (i.e., hist, box, area, kde, line).
    """
    # display plot(s) only for numeric features
    if type(df) is pd.DataFrame:
        if type(ptype) is str:
            # step 1: pull all column names where the type is numeric
            cols = df.select_dtypes(include = "number").columns.values
            
            # step 2: plot each numeric feature from the original dataframe
            for name in cols:
                df[name].plot(kind = ptype, title = name)
                plt.xlabel(xlabel = "Bins")
                plt.ylabel(ylabel = "Frequency")
                plt.show()
                
        else: 
            print("Plot type requires string, (i.e., hist, box, area")
    else:
        print("Not a Pandas dataframe")

# add functionality for scatterplots??

#%%
# display distributions and outliers for numeric features using Seaborn
def snsNumplots(df, ptype):
    """
    This function displays EDA plots using Seaborn methods of visualization.
    Required arguments are: 1) the name of a Pandas dataframe, and 2) a string
    input for the type of Seaborn plot desired (i.e., "box", "dist")
    """
        # display plot(s) only for numeric features
    if type(df) is pd.DataFrame:
            # step 1: extract only numeric data from dataframe and ignore NaNs
            num_df = elimNan(df.select_dtypes(include = "number"))
            
            # step 2: pull all column names from new numeric dataframe
            col_names = num_df.columns.values
            
            # step 3: produce desired plots for each feature in numeric df
            if type(ptype) is str:
                if ptype == "box":                              
                    for name in col_names:
                        sns.boxplot(num_df[name], 
                                    color = "green", 
                                    orient = "v")
                        plt.show()
                        
                if ptype == "dist":
                    for name in col_names:
                        sns.distplot(num_df[name], 
                                     kde = True)
                        plt.show()
            else: 
                print("Plot type requires string, (i.e., box, dist)")
    else:
        print("Not a Pandas dataframe")
        

#%%
# display heatmaps to visualize correlations between numeric features
def corrNum(df, ptype, annot = False):
    """
    This function creates heatmaps or clustermaps to better visualize the 
    correlations between numeric features. Required arguments are: 1) the name
    of a Pandas dataframe, and 2) a string input for the type of heatmap 
    desired (i.e., "heatmap", "clustermap"). An optional boolean argument is 
    whether to annotate each cell with its corresponding Pearson coefficient. 
    The default is set to False.
    """
    if type(df) is pd.DataFrame:
        if type(ptype) is str:
            if annot == False:
                if ptype == "heatmap":
                    sns.heatmap(df.corr(),
                                cmap = "Blues")
                    plt.title("Correlation Matrix")
                    plt.show()
                    
                if ptype == "clustermap":
                    sns.clustermap(df.corr(), 
                                   metric = "correlation", 
                                   cmap = "Blues")
                    plt.title("Clustered Correlation Matrix")
                    plt.show()
                    
            else:
                if annot == True:
                    if ptype == "heatmap":
                        sns.heatmap(df.corr(),
                                    cmap = "Blues",
                                    annot = True)
                        plt.title("Correlation Matrix")
                        plt.show()
                        
                    if ptype == "clustermap":
                        sns.clustermap(df.corr(), 
                                       metric = "correlation", 
                                       cmap = "Blues",
                                       annot = True)
                        plt.title("Clustered Correlation Matrix")
                        plt.show()
        else:
            print("ptype requires string (i.e., heatmap or clustermap)")
    else:
        print("Not a Pandas dataframe")
   
     
#%%
# describe object features          
def describeCat(df, include = None):
    """
    Describe string/object features in a dataframe. Takes two arguments: 1) the
    name of the Pandas dataframe, and 2) an optional argument that passes in a
    list of desired features for which value counts and counts of distinct
    observations are generated. Returns counts and percentages of unique values
    for each categorical feature and counts of their distinct observations.
    """
    if type(df) is pd.DataFrame:
        # extract string features into new dataframe
        obj_df = df.select_dtypes(include = "object")
        
        if include is None:
            # extract string feature names to new object
            obj_cols = obj_df.columns.values
            
            # distinct values for all string features
            for name in obj_cols:                
                counts = obj_df[name].value_counts(dropna = False)
                percents = counts / len(obj_df)
                new_df = pd.DataFrame({"counts": counts,
                                       "percent of total": percents})
                print(new_df)
                print("Count of distinct observations including NaNs: ",
                      obj_df[name].nunique(dropna = False))
                print("Count of distinct observations without NaNs: ",
                      obj_df[name].nunique(dropna = True))
                print("\n")
                obj_df[name].value_counts(dropna = False).plot(kind = "bar", 
                      title = name)
                plt.show()
                print("\n")
                print("\n")
                
            return new_df
            
        else:
            obj_cols = obj_df[include].columns.values
            for name in obj_cols:
                counts = obj_df[name].value_counts(dropna = False)
                percents = counts / len(obj_df)
                new_df = pd.DataFrame({"counts": counts,
                                       "percent of total": percents})
                print(new_df)
                print("Count of distinct observations including NaNs: ",
                      obj_df[name].nunique(dropna = False))
                print("Count of distinct observations without NaNs: ",
                      obj_df[name].nunique(dropna = True))
                print("\n")
                obj_df[name].value_counts(dropna = False).plot(kind = "bar", 
                      title = name)
                plt.show()
                print("\n")
                print("\n")
            
            return new_df

    else:
        print("Not a Pandas dataframe")
                 
#%%
# visualize categorical features using scatter, distribution, bar, and count 
# plots in Seaborn
def Catplots(df, ptype, include = None):
    """
    This function displays EDA plots for categorical features of a Pandas
    dataframe. It has two required arguments: 1) the name of a Pandas dataframe,
    and 2) a string input for the type of plot desired (i.e., "strip", "box",
    "violin", "bar"). There is an optional argument for passing in a list of 
    desired string/categorical features.
    """
    # display plot(s) only for categorical features of a Pandas dataframe
    if type(df) is pd.DataFrame:
        if type(ptype) is str:
            # extract only object data from dataframe
            obj_cols = df.select_dtypes(include = "object").columns.values
            num_cols = df.select_dtypes(include = "number").columns.values
            
            # generate plots for every string feature by each numerical feature
            if include is None:
                
                # Seaborn scatterplot
                if ptype == "strip":
                    for name1 in obj_cols:
                        for name2 in num_cols:
                            sns.catplot(x = name2,
                                        y=name1, 
                                        data = df)
                            plt.show()
                            
                # Seaborn boxplot; good for outlier detection
                elif ptype == "box":
                    for name1 in obj_cols:
                        for name2 in num_cols:
                            sns.catplot(x = name2, 
                                        y=name1, 
                                        kind = "box", 
                                        data = df)
                            plt.show()
                            
                # Seaborn violin plot; good for distributions            
                elif ptype == "violin":
                    for name1 in obj_cols:
                        for name2 in num_cols:
                            sns.catplot(x = name2, 
                                        y=name1, 
                                        kind = "violin", 
                                        data = df)
                            plt.show()
                    
                # Seaborn bar plot; good for distributions
                elif ptype == "bar":
                    for name1 in obj_cols:
                        for name2 in num_cols:
                            sns.catplot(x = name2, 
                                        y=name1, 
                                        kind = "bar", 
                                        data = df)
                            plt.show()
                
                           
            else:
                # generate plots for only the desired string feature(s) by each
                # numerical feature
                obj_cols = df[include].columns.values
                
                # Seaborn scatterplot
                if ptype == "strip":
                    for name1 in obj_cols:
                        for name2 in num_cols:
                            sns.catplot(x = name2, 
                                        y=name1, 
                                        data = df)
                            plt.show()
                
                # Seaborn boxplot; good for outlier detection            
                elif ptype == "box":
                    for name1 in obj_cols:
                        for name2 in num_cols:
                            sns.catplot(x = name2, 
                                        y=name1, 
                                        kind = "box", 
                                        data = df)
                            plt.show()
                
                # Seaborn violin plot; good for distributions
                elif ptype == "violin":
                    for name1 in obj_cols:
                        for name2 in num_cols:
                            sns.catplot(x = name2, 
                                        y=name1, 
                                        kind = "violin", 
                                        data = df)
                            plt.show()
                
                # Seaborn bar plot; good for distributions
                elif ptype == "bar":
                    for name1 in obj_cols:
                        for name2 in num_cols:
                            sns.catplot(x = name2, 
                                        y=name1, 
                                        kind = "bar", 
                                        data = df)
                            plt.show()
        
        else:
            print("ptype requires string")
    else:
        print("Not a Pandas dataframe")        

   
    
   
        
        
        
        
        
        
        
        
        
        
        