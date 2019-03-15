# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:59:06 2019

@author: chari
"""

#%%
# import libraries
import MyFuncs as mf
import pandas as pd

#%%
# import customer history data
cust_hist = mf.loadDF("CustomerHistory.csv", "csv", resetmax = True)

#%%
# quick view of data
cust_hist.shape
# Rows: 12,010
# Columns: 51

# get info on each feature
cust_hist.info()

# get head of df
cust_hist.head()
# many features are binary variables {0,1}

# describe numeric features
cust_hist.describe()
# userid should be index or key that is used to concat/merge dfs

# AttributedEmail is binary {0,1}
# EmailVerified is binary {0,1}
# GeneratedTerms is binary {0,1}
# HasBureauError is binary {0,1}
# HasStartedPurchase is binary {0,1}
# InboundCall is binary {0,1}
# InboundChat is binary {0,1}
# InboundEmail is binary {0,1}
# IsDealSeeker is binary {0,1}
# IsDirtLover is binary {0,1}
# IsDreamer is binary {0,1}
# IsGreen is binary {0,1}
# IsResearcher is binary {0,1}
# MultipleEmail is binary {0,1}
# OutboundEmail is binary {0,1}
# ReferralAttachment is binary {0,1}
# TradeInValueGenerated is binary {0,1}

# InboundCallCount is discrete count; may need to be binarized
# InboundChatCount is discrete count; may need to be binarized
# InboundEmailCount is discrete count; may need to be binarized
# OutboundCallCount is discrete count; may need to be binarized
# OutboundEmailCount is discrete count; may need to be binarized
# DeviceCount is discrete count
# NumberSavedVehicles is discrete count
# NumberSearches is discrete count
# UniqueModels is discrete count
# VehicleVelocity1 is discrete count
# VehicleVelocity15 is discrete count
# VehicleVelocity30 is discrete count

# TotalInboundCallLength is continuous; likely in seconds
# BureauIncome is continuous
# CreditScore1 is continuous
# CreditScore2 is continuous
# FraudScore is continuous
# Income is continuous; shows some negative values, which may be valid
# MedianVehicleFuelEcon is continuous
# MedianVehiclePrice is continuous
# TradeInValueAmount is continuous
# MedianVehicleMileage is continuous

# ClicksQuintile is categorical


# describe object features
cust_hist.select_dtypes("object").describe()
# CensusGeoRegionAddress is categorical
# LeadSource is categorical
# ModeDeviceType is categorical
# ModeVehicleType is categorical
# PhoneType is categorical
# ServiceProvider is categorical
# AgeCategory is categorical
# InboundCallContact is categorical

# PhoneNumber_hashed is not a usable object because every row is unique; could
# be used to merge/concate dfs
# accountcreationdatetime is not a true object; it is a date-time object and 
# should be parsed/converted


#%%
# any missing values?
cust_missing_count = cust_hist.isna().sum()
cust_missing_percent = mf.percentMissing(cust_hist)
# only PhoneNumber_hashed has missing values


#%%
# create new datetime column by converting accountcreationdatetime to datetime 
# object 
cust_hist["Date_Time_account"] = pd.to_datetime(
        cust_hist["accountcreationdatetime"],
        errors = "raise",
        format = "%d%b%Y:%H:%M:%S.%f")

# drop accountcreatedatetime column
cust_hist.drop(columns = "accountcreationdatetime", inplace = True)

# confirm new datetime object in df
cust_hist.info()


#%%
# get counts and percentages of unique values for object features of interest
# object features of interest
cat_vars = ['CensusGeoRegionAddress', 
            'LeadSource', 
            'ModeDeviceType', 
            'ModeVehicleType', 
            'PhoneType', 
            'ServiceProvider', 
            'AgeCategory', 
            'InboundCallContact']

# descriptions and histograms of each categorical/object feature
mf.describeCat(cust_hist, include = cat_vars)
# most categorical features are well-balanced, with the exception of PhoneType
# and LeadSource
# could consider re-casting PhoneType as a binary {0,1}, Postpaid vs. Other
# could consider re-casting LeadSource as a binary {0,1}, Website vs. Other


#%%
# visualize numeric features for later preprocessing steps

# numeric variables of interest
num_vars = ["TotalInboundCallLength", 
            "BureauIncome", 
            "CreditScore1", 
            "CreditScore2",
            "FraudScore",
            "Income",
            "MedianVehicleFuelEcon",
            "MedianVehiclePrice",
            "TradeInValueAmount",
            "MedianVehicleMileage",
            "NumberSavedVehicles",
            "NumberSearches",
            "UniqueModels",
            "VehicleVelocity1",
            "VehicleVelocity15",
            "VehicleVelocity30",
            "ClicksQuintile",
            "DeviceCount"]

# view histograms to check for normality, skew, etc.
mf.Numplots(cust_hist[num_vars], ptype = "hist")
# num_vars features appear to be normally distributed
# large scale differences between features
# will be an issue for any parametric modeling, like linear regression

# view boxplots to check for outliers
mf.Numplots(cust_hist[num_vars], ptype = "box")
# lots of outliers for all num_vars features
# will be an issue for any parametric modeling, like linear regression

# view heatmaps to check for correlations
cust_corr = cust_hist.corr()
mf.corrNum(cust_hist, ptype = "heatmap")
mf.corrNum(cust_hist, ptype = "clustermap")
# CreditScore1 and CreditScore2 are highly positively correlated
# VehicleVelocity1, 15, and 30 are somewhat positively correlated
# BureauIncome and Income are positively correlated
 

#%%
# visualize discrete features for later preprocessing steps

# discrete variables of interest
discrete_vars = ["InboundCallCount",
                 "InboundChatCount",
                 "InboundEmailCount",
                 "OutboundCallCount",
                 "OutboundEmailCount"]

# view histograms to check for normality, etc.
mf.Numplots(cust_hist[discrete_vars], ptype = "hist")
# these discrete counts should be binarized


#%%
# export df to facilitate merging
cust_hist.to_csv("cust_hist.csv", index = False)


#%%
# import call history data
call_hist = mf.loadDF("OBCallHistory.csv", "csv", resetmax = True)

#%%
# quick view of data
call_hist.shape
# Rows: 7,998
# Columns: 5

# get info on each feature
call_hist.info()

# get head of df
call_hist.head()

# describe numeric features
call_hist.describe()
# Contacted is binary {0,1}
# advocate_id is likely categorical

# describe object features
call_hist.select_dtypes("object").describe()
# PhoneNumber_hashed is not a usable object because all rows are unique; could
# be used to merge/concate dfs
# OBCallDateTime is not a true object; it is a date-time object and should 
# be parsed
# CallDuration is not a true object; it is a time object and should be parsed/
# converted


#%%
# any missing values?
call_missing_count = call_hist.isna().sum()
call_missing_percent = mf.percentMissing(call_hist)
# only CallDuration has missing values
# it's likely that if Contacted is 0, then CallDuration is NaN/missing
# might need to fill these NaNs with some kind of indicator value


#%%
# create new datetime column by converting OBCallDateTime to datetime object 
call_hist["Date_Time_call"] = pd.to_datetime(call_hist["OBCallDateTime"],
         errors = "raise",
         format = "%d%b%Y:%H:%M:%S.%f")

# drop OBCallDateTime column
call_hist.drop(columns = "OBCallDateTime", inplace = True)

# confirm new datetime object in df
call_hist.info()


#%%
# convert call duration into time object
call_hist["Call_Duration"] = pd.to_datetime(call_hist["CallDuration"],
         errors = "raise",
         format = "%H:%M:%S").dt.time

# drop CallDuration column
call_hist.drop(columns = "CallDuration", inplace = True)

# confirm new time object in df
call_hist.info()



#%%
# export df to facilitate merging
call_hist.to_csv("call_hist.csv", index = False)


#%%
# import sales history data
sale_hist = mf.loadDF("SaleHistory.csv", "csv", resetmax = True)

#%%
# quick view of data
sale_hist.shape
# Rows: 5,610
# Columns: 4

# get info on each feature
sale_hist.info()

# get head of df
sale_hist.head()

# describe numeric features
sale_hist.describe()
# # userid should be index or key that is used to concat/merge dfs
# Odometer is continuous

# describe object features
sale_hist.select_dtypes("object").describe()
# saledatetime is not a true object; it is a date-time object and should be 
# parsed/converted
# StickerPrice is not a true object; it is a continuous feature that reads as
# an object due to the $-sign


#%%
# any missing values?
call_missing_count = sale_hist.isna().sum()
call_missing_percent = mf.percentMissing(sale_hist)
# no missing values


#%%
# visualize numeric features for later preprocessing steps
# numeric variables of interest
num_vars = ["Odometer"]

# view histograms to check for normality, skew, etc.
mf.Numplots(sale_hist[num_vars], ptype = "hist")
# Odometer is a normal distribution


#%%
# create new datetime column by converting saledatetime to datetime object
sale_hist["Date_Time_sale"] = pd.to_datetime(sale_hist["saledatetime"],
         errors = "raise",
         format = "%d%b%Y:%H:%M:%S.%f")

# drop saledatetime column
sale_hist.drop(columns = "saledatetime", inplace = True)
 
# confirm new datetime object in df
sale_hist.info()


#%%
# export df to facilitate merging
sale_hist.to_csv("sale_hist.csv", index = False)
