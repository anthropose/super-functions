# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 09:42:29 2019

@author: chari
"""

#%%
import pandas as pd
import MyFuncs as mf
import TimeSeriesFunctions as tsfuncs

#%%
# load data into df
comps = mf.loadDF("Data\Forecast POC Components.xlsx", "excel")


#%%
# rename demand and forecast features to be more readable
rename_DMD = {"DMD_12": "DMD_01_2018", 
          "DMD_11": "DMD_02_2018", 
          "DMD_10": "DMD_03_2018", 
          "DMD_9": "DMD_04_2018", 
          "DMD_8": "DMD_05_2018", 
          "DMD_7": "DMD_06_2018", 
          "DMD_6": "DMD_07_2018",
          "DMD_5": "DMD_08_2018", 
          "DMD_4": "DMD_09_2018", 
          "DMD_3": "DMD_10_2018", 
          "DMD_2": "DMD_11_2018", 
          "DMD_1": "DMD_12_2018"}

comps.rename(columns = rename_DMD, inplace = True)

rename_FCST = {"FCST_12": "FCST_01_2018", 
          "FCST_11": "FCST_02_2018", 
          "FCST_10": "FCST_03_2018", 
          "FCST_9": "FCST_04_2018", 
          "FCST_8": "FCST_05_2018", 
          "FCST_7": "FCST_06_2018", 
          "FCST_6": "FCST_07_2018",
          "FCST_5": "FCST_08_2018", 
          "FCST_4": "FCST_09_2018", 
          "FCST_3": "FCST_10_2018", 
          "FCST_2": "FCST_11_2018", 
          "FCST_1": "FCST_12_2018"}

comps.rename(columns = rename_FCST, inplace = True)


#%%
# groupby ABC and filter by ASL "y"
# generate df without any rows where ASL is "n" or NaN
comps_y = comps.loc[comps["ASL"] == "y"]

# reset the index
comps_y = comps_y.reset_index(drop = True)

# groupby ABC and find mean of other features
comps_groupABCy = comps_y.groupby("ABC").mean()

# take transpose of comps_groupABCy
comps_groupABCy_transpose = comps_groupABCy.transpose()

# verify index of transposed df
comps_groupABCy_transpose.index

# keep only desired rows of new df
keep = ["DMD_01_2018", 
        "DMD_02_2018", 
        "DMD_03_2018", 
        "DMD_04_2018", 
        "DMD_05_2018", 
        "DMD_06_2018", 
        "DMD_07_2018",
        "DMD_08_2018", 
        "DMD_09_2018", 
        "DMD_10_2018", 
        "DMD_11_2018", 
        "DMD_12_2018"]

comps_groupABCy = comps_groupABCy_transpose.loc[keep]


#%%
# generate time series line plots of mean demand by ABC group by month
months = ["Jan", 
          "Feb", 
          "Mar", 
          "Apr", 
          "May", 
          "June", 
          "July", 
          "Aug", 
          "Sep", 
          "Oct", 
          "Nov", 
          "Dec"]
tsfuncs.timeseriesplotbycol(comps_groupABCy, time_labels = months)

#%%
# generate a single time series line plot of mean demand by group by month
tsfuncs.timeseriesplotall(comps_groupABCy,
                          time_labels = months,
                          xlabel = "Month", 
                          ylabel = "Demand")

#%%
# view smaller numbers of ABC groups as time series line plots
A = ["A1", "A2", "A3", "A4"]
tsfuncs.timeseriesplotall(comps_groupABCy[A],
                          time_labels = months,
                          xlabel = "Month", 
                          ylabel = "Demand")

B = ["B1", "B2", "B3", "B4"]
tsfuncs.timeseriesplotall(comps_groupABCy[B], 
                          time_labels = months,
                          xlabel = "Month", 
                          ylabel = "Demand")

C = ["C2", "C3", "C4"]
tsfuncs.timeseriesplotall(comps_groupABCy[C],
                          time_labels = months,
                          xlabel = "Month", 
                          ylabel = "Demand")

M = ["M1", "M2", "M4"]
tsfuncs.timeseriesplotall(comps_groupABCy[C], 
                          time_labels = months,
                          xlabel = "Month", 
                          ylabel = "Demand")

Z = ["ZC", "ZL"]
tsfuncs.timeseriesplotall(comps_groupABCy[Z], 
                          time_labels = months,
                          xlabel = "Month", 
                          ylabel = "Demand")

one = ["A1", "B1", "M1"]
tsfuncs.timeseriesplotall(comps_groupABCy[one], 
                          time_labels = months,
                          xlabel = "Month", 
                          ylabel = "Demand")

two = ["A2", "B2", "M2"]
tsfuncs.timeseriesplotall(comps_groupABCy[two], 
                          time_labels = months,
                          xlabel = "Month", 
                          ylabel = "Demand")
        
three = ["A3", "B3", "C3"]
tsfuncs.timeseriesplotall(comps_groupABCy[three], 
                          time_labels = months,
                          xlabel = "Month", 
                          ylabel = "Demand")

four = ["A4", "B4", "C4", "M4"]
tsfuncs.timeseriesplotall(comps_groupABCy[four], 
                          time_labels = months,
                          xlabel = "Month", 
                          ylabel = "Demand")


#%%
# decompose the time series
# must first convert index into DateTimeIndex
date_time = ["2018-Jan", 
             "2018-Feb", 
             "2018-Mar", 
             "2018-Apr", 
             "2018-May", 
             "2018-Jun", 
             "2018-Jul",
             "2018-Aug", 
             "2018-Sep", 
             "2018-Oct", 
             "2018-Nov", 
             "2018-Dec"]

comps_groupABCy["Date"] = date_time
comps_groupABCy.reset_index(inplace = True, drop = True)
comps_groupABCy["Date"] = pd.to_datetime(comps_groupABCy["Date"])
comps_groupABCy = comps_groupABCy.set_index("Date", drop = True)


# decompose time series
comps_decompositions_groupABCy = tsfuncs.ts_decomposition_cluster(comps_groupABCy)

# access the seasonal and trend data from the decomposition dictionary
comps_seasonal_groupABCy = tsfuncs.decomposition_dict2df(comps_decompositions_groupABCy, 
                                               attribute = "seasonal")
# there is no seasonal information! 
# probably because there is only one year's worth of data

comps_trend_groupABCy = tsfuncs.decomposition_dict2df(comps_decompositions_groupABCy, 
                                            attribute = "trend")
# there is a trend of varying degrees for every cluster in June and July
# it's possible that this trend is actually seasonality hidden by the fact there
# is only one year's worth of data


#%%
# the above approach is using an adhoc method for unsupervised "learning"
# this would be a logical place to use clustering analysis, because cluster ID
# might reveal better group(s) among the data than ABC group does


#%%
# load the kmeans clusterID comps df
comps_cluster_kmeans = mf.loadDF("comps_kmeans_clusterID.csv", "csv")


#%%
# for timeseries plotting, drop unnecessary columns
drop = ["COST", "LEADTIME", "SAFETY_STK", "ONHANDNEW"]
comps_cluster_kmeans.drop(labels = drop, axis = 1, inplace = True)

# take transpose of comps_cluster
comps_cluster_kmeans_transpose = comps_cluster_kmeans.transpose()

# verify index of transposed df
comps_cluster_kmeans_transpose.index


#%%
# generate time series line plots of mean demand by cluster by month
tsfuncs.timeseriesplotbycol(comps_cluster_kmeans_transpose, time_labels = months)

# generate a single time series line plot of mean demand by cluster by month
tsfuncs.timeseriesplotall(comps_cluster_kmeans_transpose, 
                          time_labels = months, 
                          xlabel = "Month", 
                          ylabel = "Demand")


#%%
# decompose the time series
# must first convert index into DateTimeIndex
comps_cluster_kmeans_transpose["Date"] = date_time
comps_cluster_kmeans_transpose.reset_index(inplace = True, drop = True)
comps_cluster_kmeans_transpose["Date"] = pd.to_datetime(comps_cluster_kmeans_transpose["Date"])
comps_cluster_kmeans_transpose = comps_cluster_kmeans_transpose.set_index("Date", drop = True)


# decompose time series
comps_decompositions_kmeans = tsfuncs.ts_decomposition_cluster(comps_cluster_kmeans_transpose)

# access the seasonal and trend data from the decomposition dictionary
comps_seasonal_kmeans = tsfuncs.decomposition_dict2df(comps_decompositions_kmeans, 
                                               attribute = "seasonal")
# there is no seasonal information! 
# probably because there is only one year's worth of data

comps_trend_kmeans = tsfuncs.decomposition_dict2df(comps_decompositions_kmeans, 
                                            attribute = "trend")
# there is a trend of varying degrees for every cluster in June and July
# it's possible that this trend is actually seasonality hidden by the fact there
# is only one year's worth of data


#%%
# load the affinity propagation clusterID comps df
comps_cluster_affinity = mf.loadDF("comps_affinity_clusterID.csv", "csv")


#%%
# for timeseries plotting, drop unnecessary columns
comps_cluster_affinity.drop(labels = drop, axis = 1, inplace = True)

# take transpose of comps_cluster
comps_cluster_affinity_transpose = comps_cluster_affinity.transpose()

# verify index of transposed df
comps_cluster_affinity_transpose.index


#%%
# generate time series line plots of mean demand by cluster by month
#tsfuncs.timeseriesplotbycol(comps_cluster_affinity_transpose, time_labels = months)

# generate a single time series line plot of mean demand by cluster by month
tsfuncs.timeseriesplotall(comps_cluster_affinity_transpose, 
                          time_labels = months, 
                          xlabel = "Month", 
                          ylabel = "Demand")


#%%
# decompose the time series
# must first convert index into DateTimeIndex
comps_cluster_affinity_transpose["Date"] = date_time
comps_cluster_affinity_transpose.reset_index(inplace = True, drop = True)
comps_cluster_affinity_transpose["Date"] = pd.to_datetime(comps_cluster_affinity_transpose["Date"])
comps_cluster_affinity_transpose = comps_cluster_affinity_transpose.set_index("Date", drop = True)

# decompose time series
comps_decompositions_affinity = tsfuncs.ts_decomposition_cluster(comps_cluster_affinity_transpose)

# access the seasonal and trend data from the decomposition dictionary
comps_seasonal_affinity = tsfuncs.decomposition_dict2df(comps_decompositions_affinity, 
                                               attribute = "seasonal")
# there is no seasonal information! 
# probably because there is only one year's worth of data

comps_trend_affinity = tsfuncs.decomposition_dict2df(comps_decompositions_affinity, 
                                            attribute = "trend")
# there is a trend of varying degrees for every cluster in June and July
# it's possible that this trend is actually seasonality hidden by the fact there
# is only one year's worth of data


#%%
# there is no difference in time series decomposition results based on adhoc 
# (groupby_ABC) or unsupervised clustering technique (Kmeans, Affinity Propagation)


















