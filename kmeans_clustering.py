# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 12:08:16 2019

@author: chari
"""

#%%
# import libraries
import MyFuncs as mf
import PreprocessingFuncs as pf
import ClusteringFuncs as cf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#%%
# load data into df
comps = mf.loadDF("data_comps.csv", "csv")


#%%
# drop unnecessary columns
drop_cols = ["LOCNAME", "HOSTPARTID", "TODAY"]
comps = comps.drop(labels = drop_cols, axis = 1)


#%%
# filter by ASL "y"
# generate df without any rows where ASL is "n" or NaN
comps_y = comps.loc[comps["ASL"] == "y"]
# reset the index
comps_y = comps_y.reset_index(drop = True)
# drop ASL
comps_ASL = comps_y.drop(labels = "ASL", axis = 1)


#%%
# verify df
comps_ASL.head()
comps_ASL.info()
comps_ASL.isna().sum()


#%%
# drop rows with missing values
comps_ASL.dropna(axis = 0, how = "any", inplace = True)
comps_ASL.isna().sum()


#%%
# factorize strings to ordinals
ABC_codes = pd.DataFrame(comps_ASL["ABC"])
ABC_factor_labels, ABC_factor_uniques = pd.factorize(comps_ASL["ABC"])
comps_ASL["ABC_ID"] = ABC_factor_labels
comps_ASL.drop(labels = "ABC", axis = 1, inplace = True)

# encode ABC codes as ordinal variables
#label_cols = ["ABC"]
#pf.cat2label_preprocessor[comps_ASL, label_cols]


#%%
# K-means clustering as an unsupervised technique
# search for best number of k
kclusters = list(range(2, 30))
sse_ASL = cf.kmeans_parameter_search(comps_ASL, kclusters)

# plot SSEs and use "elbow criterion"
# this is an heuristic approach; lo0k for where the SSE decreases abruptly
cf.plot_kmeans_sse(sse_ASL)

# optimal number of clusters looks like k = 7
kmeans_ASL = KMeans(n_clusters = 7, max_iter = 1000, n_init = 25).fit(comps_ASL)

# get clusterIDs
clusterID_ASL = kmeans_ASL.labels_

# put clusterID back into df
comps_ASL["ClusterID"] = clusterID_ASL


#%%
# re-run the above code, but without any of the forecast columns to see if they
# affect the number of k
FCST = ["FCST_01_2018", 
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
        "FCST_12_2018"]

# drop FCST
comps_FCST = comps_y.drop(labels = FCST, axis = 1)

# drop ASL
comps_FCST.drop(labels = "ASL", axis = 1, inplace = True)

# drop rows with missing values
comps_FCST.dropna(axis = 0, how = "any", inplace = True)
comps_FCST.isna().sum()

# factorize strings to ordinals
comps_FCST["ABC_ID"] = pd.factorize(comps_FCST["ABC"])[0]
comps_FCST.drop(labels = "ABC", axis = 1, inplace = True)

# kmeans clustering, checking for best k
sse_FCST = cf.kmeans_parameter_search(comps_FCST, kclusters)
cf.plot_kmeans_sse(sse_FCST)

# optimal number of clusters looks like k = 7
kmeans_FCST = KMeans(n_clusters = 7, max_iter = 1000, n_init = 25).fit(comps_FCST)

# get clusterIDs
clusterID_FCST = kmeans_FCST.labels_

# put clusterID back into df
comps_FCST["ClusterID"] = clusterID_FCST


#%%
# how do these clusterIDs compare to the ABC codes?
ABC_codes["ClusterID_ASL"] = clusterID_ASL
ABC_codes["ClusterID_FCST"] = clusterID_FCST
# lots of overlap, but not always consistent


#%%
# use comps_FCST, drop ABC code, and use ClusterID for next steps 
# groupby ClusterID and find mean of other features
comps_group = comps_FCST.groupby("ClusterID").mean()

# drop ABCID
comps_group.drop(labels = "ABC_ID", axis = 1, inplace = True)

# set index to ClusterID using the following map:
# set index to ClusterID using the following map:
index = pd.Series(comps_group.index.values)
index = index.apply(lambda x: str(x))
index = index.apply(lambda x: "Cluster"+ x)

comps_group["ClusterIDs"] = index
comps_group.set_index(keys = "ClusterIDs", drop = True, inplace = True)


#%%
# export comps_group
comps_group.to_csv("comps_kmeans_clusterID.csv", index_label = False)



