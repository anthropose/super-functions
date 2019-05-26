# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 14:51:01 2019

@author: chari
"""


#%%
import MyFuncs as mf
import PreprocessingFuncs as pf
import GenerateFuncs as genfuncs
import pandas as pd


#%%
# load data into df
comps = mf.loadDF("Data\data_comps.csv", "csv")


#%%
# first, groupby ABC and find mean of other features
comps_groupABC = comps.groupby("ABC").mean()
# index of comps_group is already set to ABC codes

# drop un-wanted features
drop_cols = ["FCST_01_2018", 
             "FCST_02_2018", 
             "FCST_03_2018", 
             "FCST_04_2018", 
             "FCST_05_2018", 
             "FCST_06_2018", 
             "FCST_07_2018",
             "FCST_08_2018", 
             "FCST_09_2018", 
             "FCST_10_2018", 
             "FCST_11_2018", 
             "FCST_12_2018",
             "COST",
             "LEADTIME",
             "SAFETY_STK",
             "ONHANDNEW"]

pf.drop_preprocessing(comps_groupABC, drop_cols)

# take transpose of comps_group
comps_groupABC_transpose = comps_groupABC.transpose()
comps_groupABC_transpose.index

## keep only desired rows of new df
#keep = ["DMD_12", "DMD_11", "DMD_10", "DMD_9", "DMD_8", "DMD_7", "DMD_6", 
#         "DMD_5", "DMD_4", "DMD_3", "DMD_2", "DMD_1"]
#
#comps_groupABC = comps_groupABC_transpose.loc[keep]


#%%
# groupby ABC and filter by ASL "y"
# generate df without any rows where ASL is "n" or NaN
comps_y = comps.loc[comps["ASL"] == "y"]
# reset the index
comps_y = comps_y.reset_index(drop = True)

# groupby ABC and find mean of other features
comps_groupABCy = comps_y.groupby("ABC").mean()

# drop unwanted features
pf.drop_preprocessing(comps_groupABCy, drop_cols)

# take transpose of comps_groupABCy
comps_groupABCy_transpose = comps_groupABCy.transpose()
# verify index of transposed df
comps_groupABCy_transpose.index

## keep only desired rows of new df
#comps_groupABCy = comps_groupABCy_transpose.loc[keep]


#%%
# create dataframe(s) of newly generated demands for each ABC group
# number of years of new data
num_years = 2

# number of random variates; this is 12, because we have 12 months of demand
size = 12

# initialize an empty list to hold new dataframe(s)
new_years = []

# create index for new dataframe(s)
index_list = list(comps_groupABCy.columns.values)

# calls the genfuncs.generate_df function, which calls the 
# genfuncs.GeneratedData class
for year in range(num_years):
    yeardf = genfuncs.generate_df(comps_groupABCy_transpose, size = size)
    yeardf.set_index(keys = pd.Series(index_list), drop = True, inplace = True)
    new_years.append(yeardf)
    

#%%
# simulate 2017 demands
year_2017_ABC = new_years[0]

# simulate 2016 demands
year_2016_ABC = new_years[1]
    

#%%
# compare simulated demand data histograms to original histograms
# original data
mf.Numplots(comps_groupABCy, "hist")

# new data
mf.Numplots(year_2017_ABC, "hist")
mf.Numplots(year_2016_ABC, "hist")


#%%
# load data into df
clusters = mf.loadDF("Data\comps_kmeans_clusterID.csv", "csv")


#%%
# merge comps and clusters dfs together
# set index
clusters.set_index(keys = "Original_Index", drop = True, inplace = True)

# merge both dataframes
comps_cluster = comps.merge(clusters, 
                            how = "outer", 
                            left_index = True, 
                            right_index = True)


#%%
# view data
comps_cluster.info()


#%%
# drop products where ASL = n
# generate df without any rows where ASL is "n" or NaN
comps_cluster = comps_cluster.loc[comps_cluster["ASL"] == "y"]
# reset the index
comps_cluster = comps_cluster.reset_index(drop = True)


#%%
# for now, drop rows with missing values
comps_cluster.dropna(axis = 0, how = "any", inplace = True)
# verify missing values are dropped
comps_cluster.isna().sum()


#%%
# groupby ABC and find mean of other features
comps_clustergroup = comps_cluster.groupby("ClusterID").mean()

# set index to ClusterID using the following map:
index = pd.Series(comps_clustergroup.index.values)
index = index.apply(lambda x: int(x))
index = index.apply(lambda x: str(x))
index = index.apply(lambda x: "Cluster"+ x)

comps_clustergroup["ClusterIDs"] = index
comps_clustergroup.set_index(keys = "ClusterIDs", drop = True, inplace = True)


# drop unwanted features
pf.drop_preprocessing(comps_clustergroup, drop_cols)

# take transpose of comps_clustergroup
comps_clustergroup_transpose = comps_clustergroup.transpose()
# verify index of transposed df
comps_clustergroup_transpose.index


#%%
# create dataframe(s) of newly generated demands for each Cluster
# number of years of new data
num_years = 2

# number of random variates; this is 12, because we have 12 months of demand
size = 12

# initialize an empty list to hold new dataframe(s)
new_years = []

# create index for new dataframe(s)
index_list = list(comps_clustergroup.columns.values)

# calls the genfuncs.generate_df function, which calls the 
# genfuncs.GeneratedData class
for year in range(num_years):
    yeardf = genfuncs.generate_df(comps_clustergroup_transpose, size = size)
    yeardf.set_index(keys = pd.Series(index_list), drop = True, inplace = True)
    new_years.append(yeardf)


#%%
# simulate 2017 demands
year_2017_clusters = new_years[0]

# simulate 2016 demands
year_2016_clusters = new_years[1]
    



